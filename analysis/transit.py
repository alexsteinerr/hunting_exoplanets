import numpy as np
import math
import pandas as pd
from config.settings import *

def analyze_transit_dip(model, phase, flux, p_dense, y_nn_dense, period_days):
    P_SECONDS = period_days * 86400.0
    
    # Find dip center
    dip_center = float(phase[np.nanargmin(flux)])
    
    # Calculate area under the dip
    below = y_nn_dense < BASELINE
    if not np.any(below):
        # No transit detected
        return {
            'dip_center': dip_center,
            'area_phase': 0.0,
            'area_time': 0.0,
            'width_phase': 0.0,
            'width_time_seconds': 0.0,  # Consistent naming
            'depth_eq': 0.0,
            'rp_over_rs_est': 0.0,
            't0_phase': dip_center,
            't0_seconds': dip_center * P_SECONDS,
            'transit_detected': False
        }
    
    p_shade = p_dense[below]
    y_shade = y_nn_dense[below]
    drop = np.maximum(0.0, BASELINE - y_shade)

    area_phase = np.trapz(drop, p_shade)
    area_time = area_phase * P_SECONDS
    width_phase = (p_shade.max() - p_shade.min()) if p_shade.size >= 2 else 0.0
    width_time_seconds = width_phase * P_SECONDS  # Consistent naming

    # Equivalent box depth & Rp/Rs
    depth_eq = (area_phase / width_phase) if width_phase > 0 else 0.0
    rp_over_rs_est = math.sqrt(max(depth_eq, 0.0))

    # Estimate t0
    idx_min = np.argmin(y_nn_dense)
    t0_phase = float(p_dense[idx_min])
    t0_seconds = t0_phase * P_SECONDS

    return {
        'dip_center': dip_center,
        'area_phase': area_phase,
        'area_time': area_time,  # Keep for backward compatibility
        'width_phase': width_phase,
        'width_time_seconds': width_time_seconds,  # Use consistent naming
        'depth_eq': depth_eq,
        'rp_over_rs_est': rp_over_rs_est,
        't0_phase': t0_phase,
        't0_seconds': t0_seconds,
        'transit_detected': True
    }

def fetch_stellar_radius(target_name):
    """Fetch stellar radius from TIC catalog"""
    try:
        from astroquery.mast import Catalogs
        print(f"[INFO] Querying TIC for stellar radius of {target_name}…")
        tic_tab = Catalogs.query_object(target_name, catalog="TIC")
        tic_row = tic_tab.to_pandas().sort_values("Tmag").iloc[0]
        rstar_rsun = float(tic_row.get("rad", np.nan))
        R_SUN_M = 6.957e8
        
        if math.isfinite(rstar_rsun):
            R_s_m = rstar_rsun * R_SUN_M
            return R_s_m, rstar_rsun
        else:
            return float("nan"), float("nan")
    except Exception as e:
        print(f"[WARN] TIC query failed for {target_name}: {e}")
        return float("nan"), float("nan")