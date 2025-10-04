# wasp18_nn_dip_area_poly.py
# TESS WASP-18 -> fold -> NN fit -> dip area & Chebyshev polynomial
# Derive & PRINT: R_s, R_p, tau, t0 from data (R_s via TIC; Rp via Rp/Rs).

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

# Dip window & area
HALF_WINDOW = 0.06
BASELINE = 1.0
DENSE_SAMPLES = 4000

# Polynomial fit
CHEB_DEGREE = 8

# Outputs
CSV_OUT = "wasp18_folded_mlp.csv"
CSV_OUT_BINNED = "wasp18_folded_mlp_binned.csv"

# ================ Reproducibility =================
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ================= Download & fold =================
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

folded_lc = lc.fold(period=PERIOD_DAYS)  # phase ~ [-0.5, 0.5], transit near 0

# Arrays
phase = folded_lc.phase.value.astype(np.float32)
flux = folded_lc.flux.value.astype(np.float32)
flux_err_attr = getattr(folded_lc, "flux_err", None)
flux_err = (flux_err_attr.value.astype(np.float32)
            if flux_err_attr is not None else np.full_like(flux, np.nan, dtype=np.float32))

# ================== Phase binning ==================
def phase_bin(phase_arr, y, yerr, nbins=400):
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

if USE_BINNING:
    print(f"[INFO] Binning to {NBINS} phase bins …")
    phase_b, flux_b, flux_err_b, counts_b = phase_bin(phase, flux, flux_err, NBINS)

# ============ Fourier features for NN ============
def featurize(p_numpy: np.ndarray) -> np.ndarray:
    p = p_numpy.astype(np.float32)
    x1 = 2.0 * p
    w = 2 * math.pi * p
    feats = [x1,
             np.sin(w), np.cos(w),
             np.sin(2*w), np.cos(2*w),
             np.sin(3*w), np.cos(3*w)]
    return np.vstack(feats).T.astype(np.float32)

# Training tensors (use binned if available)
if USE_BINNING:
    X_np = featurize(phase_b)
    y_np = flux_b.astype(np.float32)
    err_base = np.nanmedian(flux_err_b) if np.any(np.isfinite(flux_err_b)) else 1.0
    err_np = np.where(np.isfinite(flux_err_b), flux_err_b, err_base).astype(np.float32)
else:
    X_np = featurize(phase)
    y_np = flux.astype(np.float32)
    err_base = np.nanmedian(flux_err) if np.any(np.isfinite(flux_err)) else 1.0
    err_np = np.where(np.isfinite(flux_err), flux_err, err_base).astype(np.float32)

X_train = torch.from_numpy(X_np)
y_train = torch.from_numpy(y_np).unsqueeze(1)
err_train = torch.from_numpy(err_np).unsqueeze(1)

# ===================== NN model =====================
class MLP2(nn.Module):
    def __init__(self, in_dim=7, h1=256, h2=256, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),    nn.ReLU(),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x): return self.net(x)

model = MLP2(in_dim=X_train.shape[1], h1=H1, h2=H2, out_dim=1)

# Weighted loss
eps = 1e-8
w_iv = 1.0 / torch.clamp(err_train**2, min=eps)
if torch.any(torch.isfinite(w_iv)):
    w_iv = torch.clamp(w_iv, max=torch.quantile(w_iv[torch.isfinite(w_iv)], 0.99))
else:
    w_iv = torch.ones_like(w_iv)
w_transit = 1.0 + ALPHA_TRANSIT * torch.clamp(1.0 - y_train, min=0.0)
weights = (w_iv * w_transit).detach()

def weighted_mse(pred, target, w): return torch.mean(w * (pred - target)**2)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 1000))

# ===================== Train NN =====================
best_loss = float("inf"); wait = 0
best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
print("[INFO] Training NN …")
model.train()
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = weighted_mse(pred, y_train, weights)
    loss.backward()
    optimizer.step(); scheduler.step()
    cur = loss.item()
    if cur + 1e-10 < best_loss:
        best_loss, wait = cur, 0
        best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | wMSE {cur:.8e} | best {best_loss:.8e}")
    if wait >= PATIENCE:
        print(f"[INFO] Early stopping at epoch {epoch} (best wMSE {best_loss:.8e})")
        break

model.load_state_dict(best_state)

# ================ Predictions & CSVs ================
with torch.no_grad():
    y_hat_train = model(X_train).squeeze(1).cpu().numpy()

if USE_BINNING:
    pd.DataFrame({
        "phase": phase_b, "flux": flux_b, "flux_err": flux_err_b,
        "mlp_pred": y_hat_train, "residual": flux_b - y_hat_train
    }).to_csv(CSV_OUT_BINNED, index=False)
    print(f"[SAVE] {CSV_OUT_BINNED}")

X_full = torch.from_numpy(featurize(phase))
with torch.no_grad():
    y_full_pred = model(X_full).squeeze(1).cpu().numpy()

pd.DataFrame({
    "phase": phase, "flux": flux, "flux_err": flux_err,
    "mlp_pred": y_full_pred, "residual": flux - y_full_pred
}).sort_values("phase").to_csv(CSV_OUT, index=False)
print(f"[SAVE] {CSV_OUT}")

# ============ Dip center & dense NN in window ============
if USE_BINNING:
    dip_center = float(phase_b[np.nanargmin(flux_b)])
else:
    dip_center = float(phase[np.nanargmin(flux)])
half_window = float(HALF_WINDOW)

p_dense = np.linspace(dip_center - half_window, dip_center + half_window,
                      DENSE_SAMPLES, dtype=np.float32)
with torch.no_grad():
    y_nn_dense = model(torch.from_numpy(featurize(p_dense))).squeeze(1).cpu().numpy()

# ============== Area under (1 - y_nn) where y_nn < 1 ==============
below = y_nn_dense < BASELINE
p_shade = p_dense[below]
y_shade = y_nn_dense[below]
drop = np.maximum(0.0, BASELINE - y_shade)

area_phase = np.trapz(drop, p_shade)                 # phase units
P_SECONDS = PERIOD_DAYS * 86400.0
area_time = area_phase * P_SECONDS                    # seconds
width_phase = (p_shade.max() - p_shade.min()) if p_shade.size >= 2 else 0.0
width_time = width_phase * P_SECONDS

# Equivalent box depth & Rp/Rs
depth_eq = (area_phase / width_phase) if width_phase > 0 else 0.0
rp_over_rs_est = math.sqrt(max(depth_eq, 0.0))

# ======== Estimate t0 and tau directly from the NN curve ========
# Mid-transit phase ~ position of minimum of NN within window
idx_min = np.argmin(y_nn_dense)
t0_phase = float(p_dense[idx_min])                    # phase
t0_seconds = t0_phase * P_SECONDS                     # seconds relative to fold 0

# τ estimate from NN: (i) width_time from area/box-depth (robust),
# (ii) also a crossing-based width at 50% of depth for reference.
tau_est_seconds = width_time

# Crossing-based (optional): where y crosses baseline - 0.5*depth
if depth_eq > 0:
    level = BASELINE - 0.5 * depth_eq
    # find indices around left/right crossings
    left = None; right = None
    for i in range(1, len(p_dense)):
        if (y_nn_dense[i-1] > level) and (y_nn_dense[i] <= level):
            left = i
            break
    for i in range(len(p_dense)-2, -1, -1):
        if (y_nn_dense[i+1] > level) and (y_nn_dense[i] <= level):
            right = i+1
            break
    if left is not None and right is not None and right > left:
        tau50_phase = p_dense[right] - p_dense[left]
        tau50_seconds = tau50_phase * P_SECONDS
    else:
        tau50_seconds = float("nan")
else:
    tau50_seconds = float("nan")

# ============== Fit Chebyshev to NN dip & print equation ==========
domain = [dip_center - half_window, dip_center + half_window]
cheb = Chebyshev.fit(p_dense, y_nn_dense, deg=CHEB_DEGREE, domain=domain)
poly: Polynomial = cheb.convert(kind=Polynomial)      # power-basis polynomial

terms = []
for k, a_k in enumerate(poly.coef):
    coeff = f"{a_k:.10e}"
    if k == 0:   terms.append(f"{coeff}")
    elif k == 1: terms.append(f"{coeff}·p")
    else:        terms.append(f"{coeff}·p^{k}")
equation_str = "y(p) = " + " + ".join(terms)

# ============== Fetch stellar radius from TIC (optional) ==========
R_s_m = float("nan"); R_p_m = float("nan")
if FETCH_CATALOG_PARAMS:
    try:
        # Prefer exact TIC record for the target if available
        # 1) query by target name; fall back to targetid if present
        print("[INFO] Querying TIC for stellar radius …")
        tic_tab = Catalogs.query_object(TARGET_NAME, catalog="TIC")
        # Filter to the brightest/closest match
        tic_row = tic_tab.to_pandas().sort_values("Tmag").iloc[0]
        rstar_rsun = float(tic_row.get("rad", np.nan))  # stellar radius in solar radii
        R_SUN_M = 6.957e8
        if math.isfinite(rstar_rsun):
            R_s_m = rstar_rsun * R_SUN_M
            if depth_eq > 0:
                R_p_m = R_s_m * rp_over_rs_est
        else:
            print("[WARN] TIC radius not available; R_s will be NaN and R_p cannot be in meters.")
    except Exception as e:
        print(f"[WARN] TIC query failed: {e}")

# ====================== Printouts ======================
print("\n=== NN Dip Approximation — Polynomial on [{:.6f}, {:.6f}] ==="
      .format(domain[0], domain[1]))
print(equation_str)
print("\nChebyshev coefficients on the same domain:")
for i, ak in enumerate(cheb.coef):
    print(f"  a[{i}] (T_{i}) = {ak:.10e}")

print("\n=== Derived transit metrics from light curve ===")
print(f"Area ∫(1 - y_nn) dp (phase)   : {area_phase:.10e}")
print(f"Area ∫(1 - y_nn) dt (seconds) : {area_time:.10e}")
print(f"Dip width (phase)              : {width_phase:.10e}")
print(f"Dip width (sec) [τ estimate]   : {tau_est_seconds:.10e}")
print(f"Dip width 50%-depth (sec)      : {tau50_seconds:.10e}")
print(f"Equivalent box depth (NN)      : {depth_eq:.10e}")
print(f"Rp/Rs (from NN depth)          : {rp_over_rs_est:.10e}")
print(f"t0 (phase)                     : {t0_phase:.10e}")
print(f"t0 (seconds rel. to fold 0)    : {t0_seconds:.10e}")

print("\n=== Requested variables (derived from data) ===")
print(f"R_s (meters, from TIC)         : {R_s_m:.10e}")
print(f"R_p (meters, via Rp/Rs*R_s)    : {R_p_m:.10e}")
print(f"tau (seconds, from NN)         : {tau_est_seconds:.10e}")
print(f"t0  (seconds, from NN mid)     : {t0_seconds:.10e}")

# ===================== UNIFIED OVERLAY PLOT =====================
plt.figure(figsize=(10,6))
plt.scatter(phase, flux, s=5, alpha=0.20, label="Raw folded flux")

if USE_BINNING:
    plt.errorbar(phase_b, flux_b, yerr=flux_err_b, fmt="o",
                 ms=4, capsize=2, alpha=0.9, label="Binned flux")

plt.axhline(BASELINE, lw=1.2, ls="--", alpha=0.7, label="Baseline (y=1)")
plt.plot(p_dense, y_nn_dense, lw=2.0, label="NN fit (dip window)")

# Shaded area where NN < 1
plt.fill_between(p_shade, y_shade, BASELINE, alpha=0.35, label="Area ∫(1 - y_nn) dp")

# Window guides
plt.axvline(dip_center - half_window, ls="--", lw=1.2, alpha=0.6)
plt.axvline(dip_center + half_window, ls="--", lw=1.2, alpha=0.6)

txt = (
    f"τ (s): {tau_est_seconds:.3e}\n"
    f"t0 (s): {t0_seconds:.3e}\n"
    f"Rp/Rs:  {rp_over_rs_est:.3e}\n"
    f"R_s (m): {R_s_m:.3e}\n"
    f"R_p (m): {R_p_m:.3e}"
)
plt.gca().text(0.02, 0.05, txt, transform=plt.gca().transAxes,
               fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

plt.title("WASP-18b — NN Dip Area, Polynomial Approx., and Derived Parameters")
plt.xlabel("Orbital Phase (days / P)")
plt.ylabel("Normalized Flux")
plt.grid(alpha=0.3)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
