import lightkurve as lk
import numpy as np
import pandas as pd
import os
import requests
from config.settings import *

CACHE_DIR = "lightcurve_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _tap_get(query, timeout=15):
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        params={"query": query, "format": "json"},
        timeout=timeout,
    )
    if r.status_code == 200:
        return r.json()
    return None

def _is_nonempty(rows):
    return isinstance(rows, list) and len(rows) > 0

def _norm_tic(tic_id):
    if not tic_id:
        return None
    s = tic_id.upper().replace("TIC", "").replace(" ", "")
    return int(s) if s.isdigit() else None

def _norm_toi_name(name):
    t = name.upper().replace("TOI", "").replace("-", " ").strip()
    return f"TOI {t}"

def query_nasa_exoplanet_archive(target_name, tic_id=None):
    tname = target_name.strip()
    tic_num = _norm_tic(tic_id)
    q_ps_exact = f"select pl_name,hostname from ps where lower(pl_name)=lower('{tname}')"
    rows = _tap_get(q_ps_exact)
    if _is_nonempty(rows):
        return "Confirmed Exoplanet"
    q_ps_like = f"select pl_name,hostname from ps where lower(pl_name) like lower('%{tname}%')"
    rows = _tap_get(q_ps_like)
    if _is_nonempty(rows):
        return "Confirmed Exoplanet"
    q_pc_exact = f"select pl_name,hostname from pscomppars where lower(pl_name)=lower('{tname}')"
    rows = _tap_get(q_pc_exact)
    if _is_nonempty(rows):
        return "Confirmed Exoplanet"
    q_pc_like = f"select pl_name,hostname from pscomppars where lower(pl_name) like lower('%{tname}%')"
    rows = _tap_get(q_pc_like)
    if _is_nonempty(rows):
        return "Confirmed Exoplanet"
    if "TOI" in tname.upper():
        toi_name = _norm_toi_name(tname)
        q_toi_by_name = (
            "select toi,toi_name,tfopwg_disp,tid "
            f"from toi where lower(toi_name)=lower('{toi_name}')"
        )
        rows = _tap_get(q_toi_by_name)
        if _is_nonempty(rows):
            disp = (rows[0].get("tfopwg_disp") or "").upper()
            if disp in ("CP", "KP"):
                return "Confirmed Exoplanet"
            if disp == "PC":
                return "Planet Candidate"
            if disp == "FP":
                return "False Positive"
    if tic_num is not None:
        q_toi_by_tid = f"select toi,toi_name,tfopwg_disp,tid from toi where tid={tic_num}"
        rows = _tap_get(q_toi_by_tid)
        if _is_nonempty(rows):
            disp = (rows[0].get("tfopwg_disp") or "").upper()
            if disp in ("CP", "KP"):
                return "Confirmed Exoplanet"
            if disp == "PC":
                return "Planet Candidate"
            if disp == "FP":
                return "False Positive"
    if any(x in tname.upper() for x in ("KEPLER", "KOI", "KIC")):
        q_koi_like = (
            "select kepid,kepoi_name,koi_disposition "
            f"from q1_q17_dr25_koi where lower(kepoi_name) like lower('%{tname}%')"
        )
        rows = _tap_get(q_koi_like)
        if _is_nonempty(rows):
            disp = (rows[0].get("koi_disposition") or "").upper()
            if "CONFIRMED" in disp:
                return "Confirmed Exoplanet"
            if "CANDIDATE" in disp:
                return "Planet Candidate"
            if "FALSE POSITIVE" in disp:
                return "False Positive"
    return None

def query_exofop_tess(tic_id):
    try:
        if tic_id and tic_id.upper().startswith("TIC"):
            tic_clean = tic_id.upper().replace("TIC", "").strip()
            url = f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_clean}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content = response.text.upper()
                if "FALSE POSITIVE" in content or " FP " in content or ">FP<" in content:
                    return "False Positive"
                if "CONFIRMED" in content:
                    return "Confirmed Exoplanet"
                if "CANDIDATE" in content or "PLANET CANDIDATE" in content or " PC " in content:
                    return "Planet Candidate"
        return None
    except Exception:
        return None

def query_kepler_fp_catalog(target_name):
    try:
        if any(x in target_name.upper() for x in ("KIC", "KEPLER", "KOI")):
            return None
        return None
    except Exception:
        return None

def get_disposition_heuristic(target_name, tic_id=None):
    false_positive_indicators = [
        "FP", "FALSE POSITIVE", "EB", "ECLIPSING BINARY", "V", "VARIABLE STAR", "BINARY", "B", "VARIABLE"
    ]
    target_name_upper = target_name.upper()
    for indicator in false_positive_indicators:
        if indicator.upper() in target_name_upper:
            return "False Positive (Heuristic)"
    confirmed_prefixes = [
        "TOI", "TIC", "KELT", "WASP", "HAT", "HD", "TRAPPIST", "KEPLER", "K2", "GJ", "LHS", "LTT", "L", "SCR"
    ]
    for prefix in confirmed_prefixes:
        if prefix.upper() in target_name_upper:
            return "Confirmed Exoplanet (Heuristic)"
    if "TOI" in target_name_upper:
        return "Planet Candidate (Heuristic)"
    if tic_id and tic_id.upper().startswith("TIC"):
        return "Planet Candidate (Heuristic)"
    return "Unknown Disposition"

def get_disposition_label(target_name, tic_id=None):
    print(f"[QUERY] Checking NASA Exoplanet Archive for {target_name}...")
    nasa_disp = query_nasa_exoplanet_archive(target_name, tic_id)
    if nasa_disp:
        print(f"[QUERY] NASA Archive disposition: {nasa_disp}")
        return nasa_disp
    if tic_id and tic_id.upper().startswith("TIC"):
        print(f"[QUERY] Checking ExoFOP-TESS for {tic_id}...")
        exofop_disp = query_exofop_tess(tic_id)
        if exofop_disp:
            print(f"[QUERY] ExoFOP-TESS disposition: {exofop_disp}")
            return exofop_disp
    if any(x in target_name.upper() for x in ("KIC", "KEPLER", "KOI")):
        print(f"[QUERY] Checking Kepler False Positive Catalog for {target_name}...")
        kepler_disp = query_kepler_fp_catalog(target_name)
        if kepler_disp:
            print(f"[QUERY] Kepler FP Catalog disposition: {kepler_disp}")
            return kepler_disp
    print(f"[QUERY] No database match found, using heuristic classification for {target_name}")
    return get_disposition_heuristic(target_name, tic_id)

def get_cache_filename(target_name, tic_id, period_days):
    if tic_id and tic_id.upper().startswith("TIC"):
        base_name = tic_id.replace(" ", "_")
    else:
        base_name = target_name.replace(" ", "_").replace("/", "_")
    filename = f"{base_name}_P{period_days:.6f}.csv"
    return os.path.join(CACHE_DIR, filename)

def load_cached_light_curve(target_name, tic_id, period_days):
    cache_file = get_cache_filename(target_name, tic_id, period_days)
    if os.path.exists(cache_file):
        try:
            print(f"[CACHE] Loading cached light curve from {cache_file}")
            df = pd.read_csv(cache_file)
            phase = df["phase"].values.astype(np.float32)
            flux = df["flux"].values.astype(np.float32)
            flux_err = df["flux_err"].values.astype(np.float32)
            return phase, flux, flux_err, True
        except Exception as e:
            print(f"[CACHE] Error loading cached file {cache_file}: {e}")
            return None, None, None, False
    return None, None, None, False

def save_light_curve_to_cache(target_name, tic_id, period_days, phase, flux, flux_err):
    cache_file = get_cache_filename(target_name, tic_id, period_days)
    try:
        df = pd.DataFrame({"phase": phase, "flux": flux, "flux_err": flux_err})
        df.to_csv(cache_file, index=False)
        print(f"[CACHE] Saved light curve to {cache_file}")
        return True
    except Exception as e:
        print(f"[CACHE] Error saving to cache {cache_file}: {e}")
        return False

def download_light_curve(target_name, tic_id=None):
    search_str = tic_id if (tic_id and tic_id.upper().startswith("TIC")) else target_name
    print(f"[INFO] Searching TESS light curves for {search_str} …")
    search_result = lk.search_lightcurve(search_str, mission=MISSION)
    if len(search_result) == 0:
        raise RuntimeError(f"No TESS lightcurves found for {search_str}.")
    print(f"[INFO] Downloading light curve for {search_str} …")
    try:
        lc = search_result.download_all().stitch() if USE_ALL_SECTORS else search_result.download()
    except Exception as e:
        print(f"[WARN] Stitching failed, downloading first sector: {e}")
        lc = search_result[0].download()
    if REMOVE_NANS:
        lc = lc.remove_nans()
    if NORMALIZE:
        lc = lc.normalize()
    return lc

def fold_light_curve(lc, period_days):
    folded_lc = lc.fold(period=period_days)
    phase = folded_lc.phase.value.astype(np.float32)
    flux = folded_lc.flux.value.astype(np.float32)
    flux_err_attr = getattr(folded_lc, "flux_err", None)
    flux_err = (
        flux_err_attr.value.astype(np.float32)
        if flux_err_attr is not None
        else np.full_like(flux, np.nan, dtype=np.float32)
    )
    return phase, flux, flux_err

def get_folded_light_curve(target_name, tic_id, period_days):
    phase, flux, flux_err, from_cache = load_cached_light_curve(target_name, tic_id, period_days)
    if from_cache:
        return phase, flux, flux_err, True
    print(f"[DOWNLOAD] No cache found for {target_name}, downloading...")
    lc = download_light_curve(target_name, tic_id)
    phase, flux, flux_err = fold_light_curve(lc, period_days)
    save_light_curve_to_cache(target_name, tic_id, period_days, phase, flux, flux_err)
    return phase, flux, flux_err, False

def fetch_tess_targets(n=200):
    adql = f"""
    SELECT TOP {n}
        pl_name,
        hostname,
        tic_id,
        pl_orbper AS period_days,
        pl_tranmid AS transit_epoch,
        pl_trandur AS transit_duration,
        pl_rade AS radius_earth,
        pl_radj AS radius_jupiter,
        pl_trandep AS transit_depth_ppm,
        st_rad AS stellar_radius,
        st_teff AS stellar_teff,
        sy_dist AS distance_pc,
        ra,
        dec,
        sy_tmag AS tess_mag
    FROM pscomppars
    WHERE default_flag=1
      AND tran_flag=1
      AND pl_orbper IS NOT NULL
      AND tic_id IS NOT NULL
      AND pl_trandep IS NOT NULL
      AND disc_facility LIKE '%TESS%'
    ORDER BY tic_id
    """
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        params={"query": adql, "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()
