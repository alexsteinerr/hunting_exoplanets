import time
import requests
import pandas as pd
from typing import List, Dict, Optional

class NASAExoplanetAPI:
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    def __init__(self, pause_s: float = 0.5):
        self.session = requests.Session()
        self.pause_s = pause_s

    def _run_adql(self, adql: str, timeout: int = 120) -> pd.DataFrame:
        r = self.session.get(self.BASE_URL, params={"query": adql, "format": "json"}, timeout=timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise requests.HTTPError(f"{e}\n\n[TAP ERROR BODY]\n{r.text}") from None
        data = r.json()
        return pd.DataFrame(data) if isinstance(data, list) and len(data) else pd.DataFrame()

    # -------- schema helper --------
    def _list_columns(self, table: str) -> set:
        df = self._run_adql(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{table}'")
        return set(df["column_name"].astype(str)) if not df.empty else set()

    # ------------------ TESS false positives (schema-aware) ------------------
    def get_tess_false_positives(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch TOIs with false-positive disposition using the real columns present in `toi`.
        It prefers TIC ID as key and gracefully skips any missing columns.
        """
        cols = self._list_columns("toi")

        # always try to include these if present
        wanted = [
            ("toi", "toi"),
            ("tic_id", "tic_id"),
            ("tfopwg_disp", "tfopwg_disp"),
            ("disposition", "disposition"),
            ("ra", "ra"),
            ("dec", "dec"),
            ("pl_tranmid", "pl_tranmid"),
            ("pl_orbper", "pl_orbper"),
            ("pl_trandur", "pl_trandur"),
            ("pl_trandep", "pl_trandep"),
            ("pl_rade", "pl_rade"),
            ("pl_insol", "pl_insol"),
            ("pl_eqt", "pl_eqt"),
            ("tmag", "tmag"),       # <-- this is the correct TESS magnitude field
            ("st_dist", "st_dist"),
            ("st_teff", "st_teff"),
            ("st_logg", "st_logg"),
            ("st_rad", "st_rad"),
        ]

        select_parts = []
        for c, alias in wanted:
            if c in cols:
                select_parts.append(f"{c} AS {alias}")

        if not select_parts:
            raise RuntimeError("No selectable columns found in `toi` (unexpected).")

        # Prefer TFOPWG disposition; fall back to general disposition text
        where_clause = "tfopwg_disp = 'FP'" if "tfopwg_disp" in cols else "disposition = 'FALSE POSITIVE'"
        order_col = "tic_id" if "tic_id" in cols else ("toi" if "toi" in cols else list(cols)[0])

        adql = f"""
        SELECT TOP {int(limit)}
            {', '.join(select_parts)}
        FROM toi
        WHERE {where_clause}
        ORDER BY {order_col} ASC
        """
        print("[API] Fetching TESS false positives (schema-aware, using TIC ID if available)…")
        return self._run_adql(adql)

    # ------------------ Kepler false positives ------------------
    def get_kepler_false_positives(self, limit: int = 1000) -> pd.DataFrame:
        adql = f"""
        SELECT TOP {int(limit)}
            kepid,
            kepoi_name,
            koi_disposition,
            koi_period,
            koi_duration,
            koi_depth,
            koi_prad,
            koi_srho,
            ra, dec,
            koi_kepmag
        FROM q1_q17_dr25_koi
        WHERE koi_disposition = 'FALSE POSITIVE'
        ORDER BY kepid ASC
        """
        print("[API] Fetching Kepler false positives…")
        return self._run_adql(adql)

# ------------------ quick demo ------------------
if __name__ == "__main__":
    api = NASAExoplanetAPI()

    # TESS FPs (uses tmag, not tess_mag)
    tess_fp = api.get_tess_false_positives(limit=100)
    print(f"[TESS] {len(tess_fp)} rows")
    print(tess_fp.head())

    # Kepler FPs
    kepler_fp = api.get_kepler_false_positives(limit=100)
    print(f"[Kepler] {len(kepler_fp)} rows")
    print(kepler_fp.head())

    # Save
    tess_fp.to_csv("tess_false_positives.csv", index=False)
    kepler_fp.to_csv("kepler_false_positives.csv", index=False)
    print("[SAVE] Wrote tess_false_positives.csv and kepler_false_positives.csv")
