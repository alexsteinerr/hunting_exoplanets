import numpy as np
from scipy import stats

def find_tight_transit_boundaries(phase, flux):
    """Find transit boundaries that capture the full dip but minimize baseline inclusion"""
    # Sort the data by phase
    sorted_indices = np.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux[sorted_indices]
   
    # Calculate robust baseline using mode-like approach
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
    threshold = baseline_flux - 0.8 * baseline_std  # Tighter threshold
   
    # Find transit start (moving left from center)
    transit_start = transit_center
    in_transit = False
   
    for i in range(min_flux_idx, -1, -1):
        current_flux = sorted_flux[i]
        if not in_transit and current_flux < threshold:
            in_transit = True
            transit_start = sorted_phase[i]
        elif in_transit and current_flux >= threshold:
            transit_start = sorted_phase[i]
            break
        if i == 0:
            transit_start = sorted_phase[0]
            break
   
    # Find transit end (moving right from center)
    transit_end = transit_center
    in_transit = False
    for i in range(min_flux_idx, len(sorted_flux)):
        current_flux = sorted_flux[i]
        if not in_transit and current_flux < threshold:
            in_transit = True
            transit_end = sorted_phase[i]
        elif in_transit and current_flux >= threshold:
            transit_end = sorted_phase[i]
            break
        if i == len(sorted_flux) - 1:
            transit_end = sorted_phase[-1]
            break
   
    # Ensure we capture the full dip by checking if boundaries are too tight
    transit_mask_initial = (sorted_phase >= transit_start) & (sorted_phase <= transit_end)
    if transit_start > transit_end:  # Handle phase wrapping
        transit_mask_initial = (sorted_phase >= transit_start) | (sorted_phase <= transit_end)
   
    min_in_transit = np.min(sorted_flux[transit_mask_initial])
    if abs(min_in_transit - min_flux) > 0.0001:
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