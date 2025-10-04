import numpy as np
import math

def featurize(p_numpy: np.ndarray) -> np.ndarray:
    p = p_numpy.astype(np.float32)
    x1 = 2.0 * p
    w = 2 * math.pi * p
    feats = [x1,
             np.sin(w), np.cos(w),
             np.sin(2*w), np.cos(2*w),
             np.sin(3*w), np.cos(3*w)]
    return np.vstack(feats).T.astype(np.float32)

def prepare_training_data(phase, flux, flux_err, use_binning=True):
    from data.processor import phase_bin
    from config.settings import USE_BINNING, NBINS
    
    if use_binning and USE_BINNING:
        phase_b, flux_b, flux_err_b, counts_b = phase_bin(phase, flux, flux_err, NBINS)
        X_np = featurize(phase_b)
        y_np = flux_b.astype(np.float32)
        err_base = np.nanmedian(flux_err_b) if np.any(np.isfinite(flux_err_b)) else 1.0
        err_np = np.where(np.isfinite(flux_err_b), flux_err_b, err_base).astype(np.float32)
        return X_np, y_np, err_np, phase_b, flux_b, flux_err_b
    else:
        X_np = featurize(phase)
        y_np = flux.astype(np.float32)
        err_base = np.nanmedian(flux_err) if np.any(np.isfinite(flux_err)) else 1.0
        err_np = np.where(np.isfinite(flux_err), flux_err, err_base).astype(np.float32)
        return X_np, y_np, err_np, phase, flux, flux_err