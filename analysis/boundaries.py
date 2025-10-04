import numpy as np
from scipy import stats

def find_tight_transit_boundaries(phase, flux):
    # Sort the data by phase
    sorted_indices = np.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux[sorted_indices]
    
    # Calculate precise baseline using mode
    flux_hist, flux_bins = np.histogram(flux, bins=100)
    mode_bin = np.argmax(flux_hist)
    baseline_flux = (flux_bins[mode_bin] + flux_bins[mode_bin + 1]) / 2
    
    # Calculate standard deviation around mode with balanced window
    balanced_mask = np.abs(flux - baseline_flux) < 0.01  # Balanced window
    baseline_std = np.std(flux[balanced_mask]) if np.sum(balanced_mask) > 10 else 0.001
    
    # Find the minimum flux point (deepest part of transit)
    min_flux_idx = np.argmin(sorted_flux)
    transit_center = sorted_phase[min_flux_idx]
    min_flux = sorted_flux[min_flux_idx]
    
    print(f"  - Baseline (mode) flux: {baseline_flux:.6f} ± {baseline_std:.6f}")
    print(f"  - Minimum flux: {min_flux:.6f}")
    print(f"  - Transit depth: {baseline_flux - min_flux:.6f}")
    
    # Use balanced threshold - wider than before but still reasonable
    threshold = baseline_flux - 0.5 * baseline_std  # Balanced threshold (was 0.3)
    
    print(f"  - Detection threshold: {threshold:.6f} (balanced approach)")
    
    # Find transit start (moving left from center)
    transit_start = transit_center
    
    # Move left from center with wider search
    for i in range(min_flux_idx, -1, -1):
        if sorted_flux[i] >= threshold:
            transit_start = sorted_phase[i]
            # Look a bit further to ensure we capture the full ingress
            look_ahead = max(0, i - 5)  # Look 5 more points ahead
            if look_ahead >= 0:
                transit_start = sorted_phase[look_ahead]
            break
        elif i == 0:
            transit_start = sorted_phase[0]
    
    # Find transit end (moving right from center)
    transit_end = transit_center
    
    for i in range(min_flux_idx, len(sorted_flux)):
        if sorted_flux[i] >= threshold:
            transit_end = sorted_phase[i]
            # Look a bit further to ensure we capture the full egress
            look_ahead = min(len(sorted_flux) - 1, i + 5)  # Look 5 more points ahead
            if look_ahead < len(sorted_flux):
                transit_end = sorted_phase[look_ahead]
            break
        elif i == len(sorted_flux) - 1:
            transit_end = sorted_phase[-1]
    
    # Additional expansion to ensure we capture the full parabolic shape
    # Expand boundaries by a small fixed amount to make transit wider
    expansion_factor = 0.005  # Expand by 0.005 in phase units
    
    transit_start = max(-0.5, transit_start - expansion_factor)
    transit_end = min(0.5, transit_end + expansion_factor)
    
    # Ensure we capture the full depth
    transit_mask_initial = (sorted_phase >= transit_start) & (sorted_phase <= transit_end)
    if transit_start > transit_end:
        transit_mask_initial = (sorted_phase >= transit_start) | (sorted_phase <= transit_end)
    
    min_in_transit = np.min(sorted_flux[transit_mask_initial])
    if abs(min_in_transit - min_flux) > 0.0001:
        # Expand more aggressively if we're missing depth
        depth_threshold = min_flux + 0.2 * (baseline_flux - min_flux)  # More generous
        
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
    
    return transit_start, transit_end