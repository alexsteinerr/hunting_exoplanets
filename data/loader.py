import os
import re
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
from astropy.utils.exceptions import AstropyWarning
from lightkurve.utils import LightkurveWarning
import lightkurve as lk
from astropy.time import Time

warnings.filterwarnings("ignore", category=LightkurveWarning)
warnings.filterwarnings("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", message="column .* has a unit but is kept as a Column")
warnings.filterwarnings("ignore", message="The following columns will be excluded from stitching")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
try:
    lk.log.setLevel("WARNING")
except Exception:
    pass

CACHE_DIR = os.environ.get("LC_CACHE_DIR") or os.path.join(os.path.dirname(__file__), "..", "cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def _norm_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (x or "").strip())

def _npz_path(kind: str, key: str) -> str:
    return os.path.join(CACHE_DIR, f"{kind}_{_norm_name(key)}.npz")

def _is_tic(x: Optional[str]) -> Optional[int]:
    if not x:
        return None
    s = str(x).upper().replace("TIC", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None

def _is_kic(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    m = re.search(r"\bKIC[-\s_]*([0-9]{4,})\b", str(name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _as_float(a, default=np.nan):
    try:
        return np.asarray(a, dtype=float)
    except Exception:
        arr = np.array(a)
        out = np.empty(arr.shape, dtype=float)
        out[:] = default
        return out

def _clean_lightcurve(lc: lk.LightCurve) -> lk.LightCurve:
    t = getattr(lc, "time", None)
    if hasattr(t, "value"):
        time = _as_float(t.value)
    elif isinstance(t, Time):
        time = _as_float(t.to_value("jd"))
    else:
        time = _as_float(t)
    f = getattr(lc, "flux", None)
    flux = _as_float(f.value if hasattr(f, "value") else f)
    fe_raw = getattr(lc, "flux_err", None)
    if fe_raw is None:
        med = np.nanmedian(flux)
        scatter = 1.4826 * np.nanmedian(np.abs(flux - med))
        flux_err = np.full_like(flux, scatter if np.isfinite(scatter) and scatter > 0 else 1e-4)
    else:
        flux_err = _as_float(fe_raw.value if hasattr(fe_raw, "value") else fe_raw)
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[mask], flux[mask], flux_err[mask]
    return lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

def _download_stitched_pdcsap(sr, mission_hint=None) -> Optional[lk.LightCurve]:
    if len(sr) == 0:
        return None
    def _filter_by_author(sr, authors):
        if "author" in sr.table.colnames:
            mask = np.isin(np.array(sr.table["author"]).astype(str), authors)
            return sr[mask] if mask.any() else sr
        return sr
    if mission_hint == "TESS":
        sr = _filter_by_author(sr, ["SPOC", "QLP"])
    elif mission_hint == "Kepler":
        sr = _filter_by_author(sr, ["Kepler"])
    elif mission_hint == "K2":
        sr = _filter_by_author(sr, ["K2"])
    lcs = []
    for r in sr:
        lc = None
        try:
            lc = r.download(quality_bitmask="hard", flux_column="pdcsap_flux")
        except Exception:
            pass
        if lc is None:
            try:
                lc = r.download(quality_bitmask="hard", flux_column="sap_flux")
            except Exception:
                lc = None
        if lc is None:
            continue
        try:
            lc = lc.normalize()
        except Exception:
            pass
        lc = _clean_lightcurve(lc)
        lcs.append(lc)
    if not lcs:
        return None
    stitched = lk.LightCurveCollection(lcs).stitch()
    stitched = _clean_lightcurve(stitched)
    return stitched

def _search_and_download_lightcurve(name: Optional[str], tic_id: Optional[str]) -> lk.LightCurve:
    tic = _is_tic(tic_id) or _is_tic(name)
    if tic:
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        if len(sr) == 0:
            sr = lk.search_lightcurve(f"TIC {tic}")
        lc = _download_stitched_pdcsap(sr, mission_hint="TESS")
        if lc is not None:
            return lc
    if name:
        sr = lk.search_lightcurve(name, mission="TESS")
        if len(sr) == 0:
            sr = lk.search_lightcurve(name)
        lc = _download_stitched_pdcsap(sr, mission_hint="TESS")
        if lc is not None:
            return lc
    if name:
        sr = lk.search_lightcurve(name, mission="Kepler")
        if len(sr) == 0:
            kic = _is_kic(name)
            if kic:
                sr = lk.search_lightcurve(f"KIC {kic}", mission="Kepler")
        lc = _download_stitched_pdcsap(sr, mission_hint="Kepler")
        if lc is not None:
            return lc
    if name:
        sr = lk.search_lightcurve(name, mission="K2")
        if len(sr) == 0:
            kic = _is_kic(name)
            if kic:
                sr = lk.search_lightcurve(f"EPIC {kic}", mission="K2")
        lc = _download_stitched_pdcsap(sr, mission_hint="K2")
        if lc is not None:
            return lc
    raise ValueError("No light curve found in TESS, Kepler, or K2 for the given target.")

def _fold_lc_if_needed(lc: lk.LightCurve, period_days: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(lc.flux, dtype=float)
    if hasattr(lc.time, "value"):
        t = np.asarray(lc.time.value, dtype=float)
    else:
        t = _as_float(lc.time)
    fe = getattr(lc, "flux_err", None)
    if fe is None:
        med = np.nanmedian(f)
        scatter = 1.4826 * np.nanmedian(np.abs(f - med))
        fe = np.full_like(f, scatter if np.isfinite(scatter) and scatter > 0 else 1e-4)
    else:
        fe = _as_float(fe)
    if period_days and np.isfinite(period_days) and period_days > 0:
        folded = lc.fold(period=period_days)
        phase = np.asarray(folded.phase.value, dtype=float)
        return phase, np.asarray(folded.flux, dtype=float), _as_float(getattr(folded, "flux_err", fe))
    lc_flat = lc.flatten(window_length=301, polyorder=2, break_tolerance=5).remove_outliers(sigma=6)
    tt = np.asarray(lc_flat.time.value, dtype=float)
    yy = np.asarray(lc_flat.flux, dtype=float)
    yyerr = getattr(lc_flat, "flux_err", None)
    if yyerr is None:
        med = np.nanmedian(yy)
        scatter = 1.4826 * np.nanmedian(np.abs(yy - med))
        yyerr = np.full_like(yy, scatter if np.isfinite(scatter) and scatter > 0 else 1e-4)
    else:
        yyerr = _as_float(yyerr)
    period_grid = np.geomspace(0.3, 50.0, 5000)
    duration_grid = np.linspace(0.5/24, 6.0/24, 16)
    bls = lk.periodogram.BoxLeastSquaresPeriodogram(tt, yy, yyerr, minimum_period=period_grid.min(), maximum_period=period_grid.max())
    res = bls.compute_stats()
    p_est = float(res.period_at_max_power)
    folded = lc.fold(period=p_est)
    return np.asarray(folded.phase.value, dtype=float), np.asarray(folded.flux, dtype=float), _as_float(getattr(folded, "flux_err", fe))

def get_folded_light_curve(name: Optional[str], tic_id: Optional[str], period_days: Optional[float]):
    key = json.dumps({"name": name or "", "tic": tic_id or "", "period": float(period_days) if period_days else None})
    fp = _npz_path("folded", key)
    if os.path.exists(fp):
        try:
            data = np.load(fp, allow_pickle=False)
            return data["phase"], data["flux"], data["flux_err"], True
        except Exception:
            pass
    lc = _search_and_download_lightcurve(name, tic_id)
    phase, flux, flux_err = _fold_lc_if_needed(lc, period_days)
    np.savez_compressed(fp, phase=phase, flux=flux, flux_err=flux_err)
    return phase, flux, flux_err, False

def get_disposition_label(name: str, tic_id: Optional[str]) -> str:
    if name and "toi" in name.lower():
        return "TESS_CANDIDATE"
    return "CONFIRMED_OR_OTHER"
