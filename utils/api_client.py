import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple

class NASAExoplanetAPI:
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    def __init__(self, pause_s: float = 0.5):
        self.session = requests.Session()
        self.pause_s = pause_s

    def _run_adql(self, adql: str, timeout: int = 120) -> pd.DataFrame:
        r = self.session.get(self.BASE_URL, params={"query": adql, "format": "json"}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data) if isinstance(data, list) and len(data) else pd.DataFrame()

    @staticmethod
    def _select_block() -> str:
        return """
            pl_name,
            hostname,
            tic_id,
            pl_orbper AS period_days,
            pl_tranmid AS transit_epoch,
            pl_trandur AS transit_duration,
            pl_rade AS radius_earth,
            pl_radj AS radius_jupiter,
            pl_ratdor AS semi_major_axis,
            pl_imppar AS impact_parameter,
            pl_trandep AS transit_depth_ppm,
            st_rad AS stellar_radius,
            st_teff AS stellar_teff,
            st_mass AS stellar_mass,
            sy_dist AS distance_pc,
            ra,
            dec,
            sy_vmag AS v_mag,
            sy_tmag AS tess_mag
        """

    @staticmethod
    def _keyset_clause(last_key: Optional[Tuple[int, str]]) -> str:
        if last_key is None:
            return ""
        last_tic, last_name = last_key
        last_name_esc = last_name.replace("'", "''")
        return f"""
            AND (
                (tic_id > {last_tic})
                OR (tic_id = {last_tic} AND pl_name > '{last_name_esc}')
            )
        """

    def _paged_fetch(self, base_where: str, page_size: int = 50000, max_pages: Optional[int] = None) -> pd.DataFrame:
        all_chunks = []
        last_key: Optional[Tuple[int, str]] = None
        page = 0
        total = 0

        while True:
            page += 1
            if max_pages is not None and page > max_pages:
                break
            keyset = self._keyset_clause(last_key)
            adql = f"""
            SELECT TOP {page_size}
                {self._select_block()}
            FROM ps
            WHERE {base_where}
            {keyset}
            ORDER BY tic_id, pl_name
            """
            df = self._run_adql(adql)
            n = len(df)
            if n == 0:
                print(f"[API] No more rows. Done. Total rows: {total}")
                break
            all_chunks.append(df)
            total += n
            print(f"[API] Page {page}: fetched {n} rows (cumulative {total})")
            last_row = df.iloc[-1]
            try:
                last_tic_val = int(str(last_row["tic_id"]).replace("TIC", "").strip()) if str(last_row["tic_id"]).upper().startswith("TIC") else int(last_row["tic_id"])
            except Exception:
                last_tic_val = int(float(last_row["tic_id"]))
            last_name_val = str(last_row["pl_name"])
            last_key = (last_tic_val, last_name_val)
            time.sleep(self.pause_s)
            if n < page_size:
                print(f"[API] Final partial page reached. Total rows: {total}")
                break

        return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

    def get_transiting_exoplanets_paged(self, min_period: float = 0.5, max_period: float = 50.0, page_size: int = 50000, max_pages: Optional[int] = None) -> pd.DataFrame:
        base_where = f"""
            default_flag=1
            AND tran_flag=1
            AND pl_orbper BETWEEN {min_period} AND {max_period}
            AND pl_trandep IS NOT NULL
            AND tic_id IS NOT NULL
        """
        print("[API] Fetching transiting exoplanets (paged)...")
        return self._paged_fetch(base_where=base_where, page_size=page_size, max_pages=max_pages)

    def get_tess_targets_paged(self, page_size: int = 50000, max_pages: Optional[int] = None) -> pd.DataFrame:
        base_where = """
            default_flag=1
            AND tran_flag=1
            AND pl_orbper IS NOT NULL
            AND pl_trandep IS NOT NULL
            AND tic_id IS NOT NULL
            AND disc_facility LIKE '%TESS%'
        """
        print("[API] Fetching TESS targets (paged)...")
        return self._paged_fetch(base_where=base_where, page_size=page_size, max_pages=max_pages)


def _format_tic(val):
    if pd.isna(val):
        return None
    s = str(val).upper().strip()
    if s.startswith("TIC"):
        s = s[3:].strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"TIC {digits}" if digits else None

def _f(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None

def create_target_list_from_api(use_tess: bool = True, page_size: int = 50000, max_pages: Optional[int] = None) -> List[Dict]:
    api = NASAExoplanetAPI()
    try:
        df = api.get_tess_targets_paged(page_size=page_size, max_pages=max_pages) if use_tess else api.get_transiting_exoplanets_paged(page_size=page_size, max_pages=max_pages)
        targets: List[Dict] = []
        for _, row in df.iterrows():
            tic_str = _format_tic(row.get("tic_id"))
            target = {
                "name": row.get("pl_name", "Unknown"),
                "hostname": row.get("hostname", "Unknown"),
                "tic_id": tic_str,
                "period_days": _f(row.get("period_days")),
                "transit_epoch": _f(row.get("transit_epoch")),
                "transit_duration": _f(row.get("transit_duration")),
                "radius_earth": _f(row.get("radius_earth")),
                "radius_jupiter": _f(row.get("radius_jupiter")),
                "transit_depth_ppm": _f(row.get("transit_depth_ppm")),
                "stellar_radius": _f(row.get("stellar_radius")),
                "stellar_teff": _f(row.get("stellar_teff")),
                "distance_pc": _f(row.get("distance_pc")),
                "ra": _f(row.get("ra")),
                "dec": _f(row.get("dec")),
                "tess_mag": _f(row.get("tess_mag")),
            }
            if target["period_days"] is not None and target["period_days"] > 0.1 and target["tic_id"] is not None:
                targets.append(target)
        print(f"[INFO] Created {len(targets)} valid targets from API")
        return targets
    except Exception as e:
        print(f"[ERROR] Failed to fetch from API: {e}")
        from targets.target_list import EXOPLANET_TARGETS
        return EXOPLANET_TARGETS

def save_targets_to_csv(targets: List[Dict], filename: str = "api_exoplanet_targets.csv"):
    df = pd.DataFrame(targets)
    df.to_csv(filename, index=False)
    print(f"[SAVE] Target list saved to {filename}")
    return df
