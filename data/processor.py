import numpy as np
import math
from config.settings import *

def phase_bin(phase_arr, y, yerr, nbins=NBINS):
    edges = np.linspace(-0.5, 0.5, nbins + 1)
    idx = np.digitize(phase_arr, edges) - 1
    valid = (idx >= 0) & (idx < nbins)

    bin_phase = np.zeros(nbins, dtype=np.float32)
    bin_y = np.zeros(nbins, dtype=np.float32)
    bin_yerr = np.zeros(nbins, dtype=np.float32)
    counts = np.zeros(nbins, dtype=np.int32)

    for b in range(nbins):
        mask = valid & (idx == b)
        bin_phase[b] = 0.5 * (edges[b] + edges[b + 1])
        if not np.any(mask):
            bin_y[b] = np.nan; bin_yerr[b] = np.nan; counts[b] = 0; continue
        bin_phase[b] = np.median(phase_arr[mask])
        bin_y[b] = np.nanmean(y[mask])
        bin_yerr[b] = (np.nanstd(y[mask]) / math.sqrt(mask.sum())) if mask.sum() >= 2 else np.nan
        counts[b] = mask.sum()

    keep = np.isfinite(bin_y)
    return bin_phase[keep], bin_y[keep], bin_yerr[keep], counts[keep]