import matplotlib.pyplot as plt
from config.settings import *

def create_overlay_plot(phase, flux, phase_b, flux_b, flux_err_b, p_dense, y_nn_dense, 
                       p_shade, y_shade, results, use_binning=True):
    plt.figure(figsize=(10,6))
    plt.scatter(phase, flux, s=5, alpha=0.20, label="Raw folded flux")

    if use_binning:
        plt.errorbar(phase_b, flux_b, yerr=flux_err_b, fmt="o",
                     ms=4, capsize=2, alpha=0.9, label="Binned flux")

    plt.axhline(BASELINE, lw=1.2, ls="--", alpha=0.7, label="Baseline (y=1)")
    plt.plot(p_dense, y_nn_dense, lw=2.0, label="NN fit (dip window)")
    plt.fill_between(p_shade, y_shade, BASELINE, alpha=0.35, label="Area ∫(1 - y_nn) dp")

    # Window guides
    dip_center = results['dip_center']
    plt.axvline(dip_center - HALF_WINDOW, ls="--", lw=1.2, alpha=0.6)
    plt.axvline(dip_center + HALF_WINDOW, ls="--", lw=1.2, alpha=0.6)

    txt = (
        f"τ (s): {results['width_time']:.3e}\n"
        f"t0 (s): {results['t0_seconds']:.3e}\n"
        f"Rp/Rs:  {results['rp_over_rs_est']:.3e}"
    )
    plt.gca().text(0.02, 0.05, txt, transform=plt.gca().transAxes,
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.title("WASP-18b — NN Dip Area, Polynomial Approx., and Derived Parameters")
    plt.xlabel("Orbital Phase (days / P)")
    plt.ylabel("Normalized Flux")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    return plt