from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import numpy as np
import json, time, traceback
from pathlib import Path
import lightkurve as lk

OUTDIR = Path("tess_folded_single_best")
MISSION = "TESS"
AUTHOR  = "SPOC"
NORMALIZE = True
SIGMA_CLIP = 6.0
NBINS = 400
USE_BINNING = True
ALLOWED_DISPOSITIONS = {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
ONLY_TRANSITING = True
MIN_PERIOD_DAYS, MAX_PERIOD_DAYS = 0.1, 1000.0
MAX_TARGETS = None
SLEEP_BETWEEN = 0.4
RESUME_SKIP_IF_PRESENT = True

R_SUN_M   = 6.957e8
R_EARTH_M = 6.371e6
R_JUP_M   = 6.9911e7

def sanitize(s: str) -> str:
    keep = "-_.() "
    return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in s).strip()

def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _normalize_pscomppars():
    ps = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,hostname,pl_orbper,tran_flag,st_rad,pl_rade,pl_radj,pl_trandur,pl_tranmid"
    ).to_pandas()
    ps = _to_numeric(ps, ["pl_orbper","tran_flag","st_rad","pl_rade","pl_radj","pl_trandur","pl_tranmid"])
    ps = ps.rename(columns={
        "pl_orbper":"orbper",
        "tran_flag":"tranflag",
        "pl_trandur":"tran_dur",
        "pl_tranmid":"tran_mid",
    })
    ps["disposition_norm"] = "CONFIRMED"
    ps = ps.dropna(subset=["pl_name","hostname","orbper"])
    return ps

def _normalize_toi():
    toi = NasaExoplanetArchive.query_criteria(table="toi", select="*").to_pandas()
    name_col = _pick(toi, "full_toi_id", "toi", "toi_name")
    tic_col  = _pick(toi, "tic_id", "ticid", "tic", "host_tic_id")
    per_col  = _pick(toi, "pl_orbper", "orbital_period", "period")
    t0_col   = _pick(toi, "pl_tranmid", "tranmid", "t0", "epoch", "transit_epoch")
    dur_col  = _pick(toi, "pl_trandurh", "pl_trandur", "tran_dur", "transit_duration", "duration")
    st_col   = _pick(toi, "st_rad", "st_radius")
    rade_col = _pick(toi, "pl_rade", "planet_radius")
    disp_col = _pick(toi, "tfopwg_disp", "disposition", "toi_disposition")

    cols_to_num = [c for c in [per_col, t0_col, dur_col, st_col, rade_col] if c]
    toi = _to_numeric(toi, cols_to_num)

    pl_name = toi[name_col].astype(str) if name_col else pd.Series(dtype=str)
    pl_name = pl_name.apply(lambda s: s if s.startswith("TOI") else f"TOI {s}")

    if tic_col:
        hostname = "TIC " + toi[tic_col].astype(str)
    else:
        hostname = pl_name

    u = pd.DataFrame({
        "pl_name": pl_name,
        "hostname": hostname,
        "orbper": toi[per_col] if per_col else pd.NA,
        "tran_mid": toi[t0_col] if t0_col else pd.NA,
        "tran_dur": toi[dur_col] if dur_col else pd.NA,
        "st_rad": toi[st_col] if st_col else pd.NA,
        "pl_rade": toi[rade_col] if rade_col else pd.NA,
        "tranflag": 1,
    })

    def norm_disp(x):
        s = "" if pd.isna(x) else str(x).upper()
        if "FP" in s or "FALSE" in s: return "FALSE POSITIVE"
        if "CP" in s or "CONFIRM" in s or "KP" in s: return "CONFIRMED"
        return "CANDIDATE"

    if disp_col:
        u["disposition_norm"] = toi[disp_col].apply(norm_disp)
    else:
        u["disposition_norm"] = "CANDIDATE"

    u = u.dropna(subset=["pl_name","hostname","orbper"])
    return u

def fetch_catalog():
    ps = _normalize_pscomppars()
    toi = _normalize_toi()
    df = pd.concat([ps, toi], ignore_index=True, sort=False)
    df = df[df["orbper"].between(MIN_PERIOD_DAYS, MAX_PERIOD_DAYS)]
    if ONLY_TRANSITING and "tranflag" in df.columns:
        df = df[df["tranflag"] == 1]
    df = df[df["disposition_norm"].isin(ALLOWED_DISPOSITIONS)]
    df = df.dropna(subset=["pl_name","hostname","orbper"]).reset_index(drop=True)

    df["__k1"] = df["pl_name"].str.upper().str.strip()
    df["__k2"] = (df["hostname"].str.upper().str.strip() + "|" + df["orbper"].round(6).astype(str))

    def _prefer_confirmed(g):
        if (g["disposition_norm"] == "CONFIRMED").any():
            return g[g["disposition_norm"] == "CONFIRMED"].iloc[0]
        return g.iloc[0]

    a = df.groupby("__k1", as_index=False, group_keys=False).apply(_prefer_confirmed)
    b = a.groupby("__k2", as_index=False, group_keys=False).apply(_prefer_confirmed)
    b = b.drop(columns=["__k1","__k2"], errors="ignore").reset_index(drop=True)
    return b

def search_tess(target: str):
    sr = lk.search_lightcurve(target, mission=MISSION, author=AUTHOR)
    if len(sr) == 0:
        sr = lk.search_lightcurve(target, mission=MISSION)
    return sr

def clean_norm(lc):
    if SIGMA_CLIP: lc = lc.remove_outliers(sigma=SIGMA_CLIP)
    if NORMALIZE:
        norm = lc.normalize(unit="ppm").remove_nans()
        flux = 1.0 + norm.flux.value/1e6
        return norm.copy(flux=flux)
    return lc.remove_nans()

def robust_scatter(y):
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    return mad / 0.67448975

def choose_best_sector(sr):
    if len(sr) == 0: return None, None
    best, best_meta, best_sc = None, None, np.inf
    for i in range(len(sr)):
        try:
            lc = sr[i].download(quality_bitmask="default")
            if lc is None: continue
            lc = clean_norm(lc)
            sc = robust_scatter(lc.flux.value)
            if np.isfinite(sc) and sc < best_sc:
                best_sc = sc
                best = lc
                row = sr.table[i]
                best_meta = {k: row[k] for k in row.colnames if k in ("sector","author","exptime","target_name","description")}
        except Exception:
            continue
    return best, best_meta

def fold_df(lc, period_days):
    f = lc.fold(period=period_days)
    phase = f.time.value / period_days
    return pd.DataFrame({"phase": phase, "flux": f.flux.value, "flux_err": getattr(f,"flux_err", None)})

def bin_df(raw_df, nbins):
    edges = np.linspace(-0.5, 0.5, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    idx = np.digitize(raw_df["phase"].values, edges) - 1
    bf, be = np.full_like(centers, np.nan, dtype=float), np.full_like(centers, np.nan, dtype=float)
    for i in range(nbins):
        sel = idx == i
        if np.any(sel):
            vals = raw_df.loc[sel,"flux"].values
            bf[i] = np.nanmean(vals)
            be[i] = np.nanstd(vals) / max(1, np.sum(sel))**0.5
    return pd.DataFrame({"phase": centers, "flux_binned": bf, "flux_binned_err": be})

def planet_radius_m(row):
    rade = pd.to_numeric(row.get("pl_rade"), errors="coerce")
    radj = pd.to_numeric(row.get("pl_radj"), errors="coerce")
    if pd.notna(rade): return float(rade) * R_EARTH_M
    if pd.notna(radj): return float(radj) * R_JUP_M
    return None

def star_radius_m(row):
    st = pd.to_numeric(row.get("st_rad"), errors="coerce")
    return None if pd.isna(st) else float(st) * R_SUN_M

def safe_float(x):
    v = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(v) else float(v)

def process(row, outdir: Path):
    host, pl, per = str(row["hostname"]), str(row["pl_name"]), float(row["orbper"])
    disp = row.get("disposition_norm","UNKNOWN")
    tgt_dir = outdir / f"{sanitize(host)}__{sanitize(pl)}"
    tgt_dir.mkdir(parents=True, exist_ok=True)
    raw_csv, bin_csv, meta_json = tgt_dir/"raw_folded.csv", tgt_dir/"binned_folded.csv", tgt_dir/"meta.json"
    for query in (pl, host):
        sr = search_tess(query)
        if len(sr): break
    if len(sr) == 0: raise RuntimeError("No TESS search results")
    best_lc, best_meta = choose_best_sector(sr)
    if best_lc is None: raise RuntimeError("No downloadable sector")
    raw = fold_df(best_lc, per)
    raw.to_csv(raw_csv, index=False)
    if USE_BINNING:
        bnd = bin_df(raw, NBINS)
        bnd.to_csv(bin_csv, index=False)
    else:
        bnd = pd.DataFrame()
    meta = {
        "host": host,
        "planet": pl,
        "disposition": disp,
        "orbital_period_days": per,
        "stellar_radius_Rsun": safe_float(row.get("st_rad")),
        "stellar_radius_m": star_radius_m(row),
        "planet_radius_Rearth": safe_float(row.get("pl_rade")),
        "planet_radius_Rjup": safe_float(row.get("pl_radj")),
        "planet_radius_m": planet_radius_m(row),
        "transit_duration_hours": safe_float(row.get("tran_dur")),
        "mid_transit_time_bjd": safe_float(row.get("tran_mid")),
        "selected_sector_meta": best_meta,
        "n_points_raw": int(len(raw)),
        "n_points_binned": int(len(bnd)) if USE_BINNING else 0,
        "files": {"raw_csv": str(raw_csv), "binned_csv": str(bin_csv)},
    }
    with open(meta_json, "w") as f: json.dump(meta, f, indent=2)
    return {"status":"ok","host":host,"planet":pl,"disposition":disp,"period_days":per,"raw_csv":str(raw_csv),"binned_csv":str(bin_csv),"meta_json":str(meta_json),"n_raw":len(raw),"n_binned":len(bnd)}

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cat = fetch_catalog()
    if MAX_TARGETS: cat = cat.head(MAX_TARGETS)
    print(f"Processing {len(cat)} targets into '{OUTDIR}' ...")
    recs = []
    for i, row in cat.iterrows():
        tag = f"{row['hostname']} — {row['pl_name']} ({row.get('disposition_norm','UNKNOWN')})"
        try:
            r = process(row, OUTDIR); r["i"] = i
            print(f"[{i+1}/{len(cat)}] {tag}: {r['status']}  (raw={r['n_raw']}, binned={r['n_binned']})")
        except Exception as e:
            r = {"i":i,"host":row.get('hostname'),"planet":row.get('pl_name'),"disposition":row.get('disposition_norm',"UNKNOWN"),"period_days": safe_float(row.get('orbper')),"status":"error","error":f"{type(e).__name__}: {e}","trace":traceback.format_exc(),"raw_csv":"","binned_csv":"","meta_json":"","n_raw":0,"n_binned":0}
            print(f"[{i+1}/{len(cat)}] {tag}: ERROR -> {e}")
        recs.append(r); time.sleep(SLEEP_BETWEEN)
    man = pd.DataFrame(recs)
    man.to_csv(OUTDIR/"MANIFEST.csv", index=False)
    print("\nSaved manifest:", (OUTDIR/"MANIFEST.csv").resolve())

if __name__ == "__main__":
    main()
