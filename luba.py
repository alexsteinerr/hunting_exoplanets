import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import lightkurve as lk
from numpy.polynomial import Chebyshev, Polynomial
import os

# Optional: fetch stellar radius from TIC (MAST) to compute absolute R_p
FETCH_CATALOG_PARAMS = True
try:
   if FETCH_CATALOG_PARAMS:
       from astroquery.mast import Catalogs
except Exception as e:
   FETCH_CATALOG_PARAMS = False
   print("[WARN] astroquery not available; will skip TIC stellar radius. Rp will be in units of Rs.")

# ===================== Config =====================
TARGET_NAME = "WASP-18"
MISSION = "TESS"
# period kept fixed here for folding; t0 & tau will be estimated from data
PERIOD_DAYS = 0.94145299

USE_ALL_SECTORS = False
REMOVE_NANS = True
NORMALIZE = True

USE_BINNING = True
NBINS = 400

# NN & training
SEED = 42
H1, H2 = 256, 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10000
PATIENCE = 500
ALPHA_TRANSIT = 10.0

# Outputs
CSV_OUT = "wasp18_folded_mlp.csv"
CSV_OUT_BINNED = "wasp18_folded_mlp_binned.csv"
LC_CSV_FILE = f"{TARGET_NAME.replace(' ', '_').lower()}_folded_lc.csv"

# ================ Reproducibility =================
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ================= Download & fold =================
def download_and_process_lc():
    """Download and process light curve data"""
    print(f"[INFO] Searching TESS light curves for {TARGET_NAME} …")
    search_result = lk.search_lightcurve(TARGET_NAME, mission=MISSION)
    if len(search_result) == 0:
        raise RuntimeError("No TESS lightcurves found for target.")
    print(search_result)

    print("[INFO] Downloading light curve …")
    lc = search_result.download_all().stitch() if USE_ALL_SECTORS else search_result.download()
   
    if REMOVE_NANS:
        lc = lc.remove_nans()
    if NORMALIZE:
        lc = lc.normalize()
   
    return lc

def save_lc_to_csv(phase, flux, flux_err, filename):
    """Save folded light curve data to CSV file"""
    df = pd.DataFrame({
        'phase': phase,
        'flux': flux,
        'flux_err': flux_err
    })
    df.to_csv(filename, index=False)
    print(f"[INFO] Folded light curve saved to: {os.path.abspath(filename)}")


def load_lc_from_csv(filename):
    """Load folded light curve data from CSV file"""
    try:
        df = pd.read_csv(filename)
        print(f"[INFO] Loaded folded light curve from CSV: {filename}")
        return df['phase'].values, df['flux'].values, df['flux_err'].values
    except FileNotFoundError:
        print(f"[INFO] CSV file not found: {filename}")
        return None, None, None


def find_tight_transit_boundaries(phase, flux):
    """Find transit boundaries that capture the full dip but minimize baseline inclusion"""
    # Sort the data by phase
    sorted_indices = np.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux[sorted_indices]
   
    # Calculate robust baseline using mode-like approach
    from scipy import stats
    flux_hist, flux_bins = np.histogram(flux, bins=50)
    mode_bin = np.argmax(flux_hist)
    baseline_flux = (flux_bins[mode_bin] + flux_bins[mode_bin + 1]) / 2
    baseline_std = np.std(flux[np.abs(flux - baseline_flux) < 0.01])  # Tight std around mode
   
    # Find the minimum flux point (deepest part of transit)
    min_flux_idx = np.argmin(sorted_flux)
    transit_center = sorted_phase[min_flux_idx]
    min_flux = sorted_flux[min_flux_idx]
   
    print(f"  - Baseline (mode) flux: {baseline_flux:.6f} ± {baseline_std:.6f}")
    print(f"  - Minimum flux: {min_flux:.6f}")
    print(f"  - Transit depth: {baseline_flux - min_flux:.6f}")
   
    # Use a tighter threshold to exclude more baseline points
    # Focus on capturing the parabolic shape, not the flat baseline
    threshold = baseline_flux - 0.8 * baseline_std  # Tighter threshold
   
    # Find transit start (moving left from center)
    transit_start = transit_center
    in_transit = False
   
    # Move left from center until we hit baseline
    for i in range(min_flux_idx, -1, -1):
        current_flux = sorted_flux[i]
       
        if not in_transit and current_flux < threshold:
            # Entering transit region
            in_transit = True
            transit_start = sorted_phase[i]
       
        elif in_transit and current_flux >= threshold:
            # Found the edge - transit starts here
            transit_start = sorted_phase[i]
            break
       
        # Stop if we've gone too far
        if i == 0:
            transit_start = sorted_phase[0]
            break
   
    # Find transit end (moving right from center)
    transit_end = transit_center
    in_transit = False
   
    for i in range(min_flux_idx, len(sorted_flux)):
        current_flux = sorted_flux[i]
       
        if not in_transit and current_flux < threshold:
            # Entering transit region
            in_transit = True
            transit_end = sorted_phase[i]
       
        elif in_transit and current_flux >= threshold:
            # Found the edge - transit ends here
            transit_end = sorted_phase[i]
            break
       
        # Stop if we've gone too far
        if i == len(sorted_flux) - 1:
            transit_end = sorted_phase[-1]
            break
   
    # Ensure we capture the full dip by checking if boundaries are too tight
    transit_mask_initial = (sorted_phase >= transit_start) & (sorted_phase <= transit_end)
    if transit_start > transit_end:  # Handle phase wrapping
        transit_mask_initial = (sorted_phase >= transit_start) | (sorted_phase <= transit_end)
   
    # Check if we're missing part of the dip
    min_in_transit = np.min(sorted_flux[transit_mask_initial])
    if abs(min_in_transit - min_flux) > 0.0001:  # If we're missing significant depth
        # Expand boundaries slightly to capture full dip
        depth_threshold = min_flux + 0.1 * (baseline_flux - min_flux)
       
        # Expand left
        for i in range(np.argmin(np.abs(sorted_phase - transit_start)), -1, -1):
            if sorted_flux[i] <= depth_threshold:
                transit_start = sorted_phase[i]
            else:
                break
       
        # Expand right
        for i in range(np.argmin(np.abs(sorted_phase - transit_end)), len(sorted_flux)):
            if sorted_flux[i] <= depth_threshold:
                transit_end = sorted_phase[i]
            else:
                break
   
    # Calculate transit duration
    transit_duration = transit_end - transit_start
    if transit_duration < 0: 
        transit_duration += 1.0
   
    return transit_start, transit_end

# Check if CSV file exists
phase, flux, flux_err = load_lc_from_csv(LC_CSV_FILE)

if phase is None:
    # Download and process light curve
    lc = download_and_process_lc()
   
    # Fold the light curve
    folded_lc = lc.fold(period=PERIOD_DAYS)

    # Extract arrays
    phase = folded_lc.phase.value.astype(np.float32)
    flux = folded_lc.flux.value.astype(np.float32)
    flux_err_attr = getattr(folded_lc, "flux_err", None)
    flux_err = (flux_err_attr.value.astype(np.float32)
               if flux_err_attr is not None else np.full_like(flux, np.nan, dtype=np.float32))
   
    # Save to CSV for future use
    save_lc_to_csv(phase, flux, flux_err, LC_CSV_FILE)
else:
    print("[INFO] Using folded light curve data from CSV file")