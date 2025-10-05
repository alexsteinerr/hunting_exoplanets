from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import time
import requests
import pandas as pd
import numpy as np
import torch
import random
import os
import sys
import logging
from typing import Dict, Any, Optional, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config.settings import *
    from data.loader import CACHE_DIR, get_disposition_label, get_folded_light_curve
    from utils.features import prepare_training_data, featurize
    from models.mlp import MLP
    from models.trainer import train_model
    from analysis.transit import analyze_transit_dip, fetch_stellar_radius
    from analysis.derivation import compare_theoretical_nn_area
    from analysis.boundaries import find_tight_transit_boundaries
    from targets.target_list import get_exoplanet_targets
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_param(req, key: str, default: str = "") -> str:
    json_data = req.get_json(silent=True) or {}
    val = json_data.get(key) or req.form.get(key) or req.args.get(key) or default
    return val

def _format_tic(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        s = str(val).strip()
        if not s:
            return None
        if s.upper().startswith('TIC'):
            return s
        digits = ''.join(ch for ch in s if ch.isdigit())
        return f"TIC {digits}" if digits else None
    except Exception:
        return None

def _f(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, str):
            s = val.strip()
            if not s or s in {"-", "--"}:
                return None
            return float(s)
        if isinstance(val, (int, float, np.floating, np.integer)):
            if isinstance(val, float) and np.isnan(val):
                return None
            return float(val)
        return None
    except (TypeError, ValueError):
        return None

def _format_value(val: Any, unit: str = "") -> str:
    if val is None or (isinstance(val, str) and not val.strip()):
        return "-"
    try:
        num_val = float(val)
        if num_val < 0.01:
            s = f"{num_val:.2e}"
        elif num_val < 1:
            s = f"{num_val:.3f}"
        elif num_val < 1000:
            s = f"{num_val:.2f}"
        else:
            s = f"{num_val:.1f}"
        return f"{s} {unit}".strip()
    except (TypeError, ValueError):
        return str(val).strip() if val else "-"

def _round_sf(x, sf=3):
    try:
        if x == 0 or (isinstance(x, float) and np.isnan(x)):
            return float(x)
        magnitude = 10 ** (np.floor(np.log10(abs(x))) - sf + 1)
        return round(x / magnitude) * magnitude
    except Exception:
        return x

def _json_safe(x):
    import numpy as _np
    import pandas as _pd
    if isinstance(x, (_np.floating, _np.integer)):
        return _round_sf(float(x))
    if isinstance(x, float):
        return _round_sf(x)
    if isinstance(x, (_np.bool_)):
        return bool(x)
    if isinstance(x, (_np.ndarray,)):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (_pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    return x

def _parse_query_type(q: str) -> Dict[str, Any]:
    s = (q or "").strip()
    if not s:
        return {"kind": "empty"}
    s_up = s.upper()
    if s_up.startswith("TIC"):
        digits = "".join(ch for ch in s_up if ch.isdigit())
        if digits:
            return {"kind": "tic", "value": int(digits)}
    if s.isdigit():
        return {"kind": "tic", "value": int(s)}
    if s_up.startswith("TOI-"):
        core = s_up.replace("TOI-", "")
        try:
            float(core)
            return {"kind": "toi_full", "value": f"TOI-{core}"}
        except:
            pass
    try:
        float(s)
        if "." in s:
            return {"kind": "toi_short", "value": s}
    except:
        pass
    return {"kind": "name", "value": s}

class NASAExoplanetAPI:
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    def __init__(self, pause_s: float = 0.5, retries: int = 2, timeout: int = 120):
        self.session = requests.Session()
        self.pause_s = pause_s
        self.retries = max(0, retries)
        self.timeout = timeout

    def _run_adql(self, adql: str, timeout: Optional[int] = None, maxrec: int = 1000) -> pd.DataFrame:
        timeout = timeout or self.timeout
        adql = ' '.join(adql.split())
        params = {"query": adql, "format": "json", "MAXREC": str(maxrec)}
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(self.BASE_URL, params=params, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data):
                    return pd.DataFrame(data)
                return pd.DataFrame()
            except Exception as e:
                last_err = e
                time.sleep(self.pause_s)
        return pd.DataFrame()

    def _list_columns(self, table: str) -> set:
        df = self._run_adql(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name = '{table}'", maxrec=10000)
        return set(df["column_name"].astype(str)) if not df.empty else set()

    def search_planet_by_name(self, planet_name: str) -> pd.DataFrame:
        planet_name_clean = (planet_name or "").replace("'", "''")
        adql = f"""
        SELECT TOP 10
            pl_name,
            hostname,
            tic_id,
            pl_orbper,
            pl_tranmid,
            pl_trandur,
            pl_rade,
            pl_radj,
            pl_trandep,
            st_rad,
            st_teff,
            st_mass,
            sy_dist,
            ra,
            dec,
            sy_tmag
        FROM ps
        WHERE default_flag = 1
          AND UPPER(pl_name) LIKE UPPER('%{planet_name_clean}%')
        ORDER BY pl_name
        """
        return self._run_adql(adql, maxrec=10)

    def get_popular_exoplanets(self, limit: int = 50) -> pd.DataFrame:
        adql = f"""
        SELECT TOP {limit}
            pl_name,
            hostname,
            tic_id,
            pl_rade,
            pl_orbper
        FROM ps
        WHERE default_flag = 1
          AND pl_rade IS NOT NULL
          AND pl_orbper IS NOT NULL
        ORDER BY pl_name
        """
        return self._run_adql(adql, maxrec=limit)

    def get_ps_by_tic_id(self, tic_id: int) -> pd.DataFrame:
        adql = f"""
        SELECT TOP 1
            pl_name, hostname, tic_id, pl_orbper, pl_tranmid, pl_trandur, pl_rade, pl_radj,
            pl_trandep, st_rad, st_teff, st_mass, sy_dist, ra, dec, sy_tmag
        FROM ps
        WHERE default_flag = 1 AND tic_id = {int(tic_id)}
        """
        return self._run_adql(adql, maxrec=1)

    def get_ps_by_name(self, planet_name: str) -> pd.DataFrame:
        planet_name_clean = (planet_name or "").replace("'", "''")
        adql = f"""
        SELECT TOP 1
            pl_name, hostname, tic_id, pl_orbper, pl_tranmid, pl_trandur, pl_rade, pl_radj,
            pl_trandep, st_rad, st_teff, st_mass, sy_dist, ra, dec, sy_tmag
        FROM ps
        WHERE default_flag = 1
          AND (UPPER(pl_name) LIKE UPPER('%{planet_name_clean}%')
               OR UPPER(hostname) LIKE UPPER('%{planet_name_clean}%'))
        ORDER BY pl_name
        """
        return self._run_adql(adql, maxrec=1)

    def get_tess_false_positives(self, limit: int = 1000) -> pd.DataFrame:
        cols = self._list_columns("toi")
        wanted = [
            ("toi","toi"),("tic_id","tic_id"),("full_toi_id","full_toi_id"),
            ("tfopwg_disp","tfopwg_disp"),("disposition","disposition"),
            ("ra","ra"),("dec","dec"),("tmag","tmag"),
            ("pl_tranmid","pl_tranmid"),("pl_orbper","pl_orbper"),("pl_trandur","pl_trandur"),
            ("pl_trandep","pl_trandep"),("pl_rade","pl_rade"),("pl_insol","pl_insol"),("pl_eqt","pl_eqt"),
            ("st_dist","st_dist"),("st_teff","st_teff"),("st_logg","st_logg"),("st_rad","st_rad")
        ]
        select_parts = [f"{c} AS {alias}" for c, alias in wanted if c in cols]
        where_clause = "tfopwg_disp = 'FP'" if "tfopwg_disp" in cols else "disposition = 'FALSE POSITIVE'"
        order_col = "tic_id" if "tic_id" in cols else ("toi" if "toi" in cols else list(cols)[0])
        adql = f"""
        SELECT TOP {int(limit)}
            {', '.join(select_parts)}
        FROM toi
        WHERE {where_clause}
        ORDER BY {order_col} ASC
        """
        return self._run_adql(adql, maxrec=limit)

    def get_toi_by_full_id(self, full_toi_id: str) -> pd.DataFrame:
        full_toi_id = full_toi_id.replace("'", "''")
        cols = self._list_columns("toi")
        wanted = [
            "toi","full_toi_id","tic_id","tfopwg_disp","disposition","ra","dec","tmag",
            "pl_tranmid","pl_orbper","pl_trandur","pl_trandep","pl_rade","pl_insol","pl_eqt",
            "st_dist","st_teff","st_logg","st_rad"
        ]
        select_cols = [c for c in wanted if c in cols]
        adql = f"""
        SELECT TOP 1 {', '.join(select_cols)}
        FROM toi
        WHERE full_toi_id = '{full_toi_id}'
        """
        return self._run_adql(adql, maxrec=1)

    def get_toi_by_short(self, toi_short: str) -> pd.DataFrame:
        cols = self._list_columns("toi")
        wanted = [
            "toi","full_toi_id","tic_id","tfopwg_disp","disposition","ra","dec","tmag",
            "pl_tranmid","pl_orbper","pl_trandur","pl_trandep","pl_rade","pl_insol","pl_eqt",
            "st_dist","st_teff","st_logg","st_rad"
        ]
        select_cols = [c for c in wanted if c in cols]
        adql = f"""
        SELECT TOP 1 {', '.join(select_cols)}
        FROM toi
        WHERE toi = {toi_short}
        """
        return self._run_adql(adql, maxrec=1)

    def get_toi_by_tic_id(self, tic_id: int) -> pd.DataFrame:
        cols = self._list_columns("toi")
        wanted = [
            "toi","full_toi_id","tic_id","tfopwg_disp","disposition","ra","dec","tmag",
            "pl_tranmid","pl_orbper","pl_trandur","pl_trandep","pl_rade","pl_insol","pl_eqt",
            "st_dist","st_teff","st_logg","st_rad"
        ]
        select_cols = [c for c in wanted if c in cols]
        adql = f"""
        SELECT TOP 1 {', '.join(select_cols)}
        FROM toi
        WHERE tic_id = {int(tic_id)}
        """
        return self._run_adql(adql, maxrec=1)

class MLExoplanetAnalyzer:
    def __init__(self):
        self.SEED = 42
        self.initialized = ML_AVAILABLE
        if not self.initialized:
            logger.warning("ML analysis system not available - using simulation mode")

    def analyze_planet(self, target: Dict[str, Any], target_index: int = 0) -> Dict[str, Any]:
        if not self.initialized:
            return self._simulate_analysis(target)
        try:
            disposition = get_disposition_label(target['name'], target.get('tic_id'))
            random.seed(self.SEED + target_index)
            np.random.seed(self.SEED + target_index)
            torch.manual_seed(self.SEED + target_index)
            phase, flux, flux_err, from_cache = get_folded_light_curve(
                target['name'], target.get('tic_id'), target['period_days']
            )
            X_np, y_np, err_np, phase_data, flux_data, flux_err_data = prepare_training_data(
                phase, flux, flux_err, USE_BINNING
            )
            if len(X_np) < 10:
                return self._create_fallback_result(target, disposition, "INSUFFICIENT_DATA")
            X_train = torch.from_numpy(X_np)
            y_train = torch.from_numpy(y_np).unsqueeze(1)
            err_train = torch.from_numpy(err_np).unsqueeze(1)
            model = MLP(in_dim=X_train.shape[1], h1=H1, h2=H2, out_dim=1)
            best_loss = train_model(model, X_train, y_train, err_train, EPOCHS, PATIENCE, LR, WEIGHT_DECAY, ALPHA_TRANSIT)
            dip_center = float(phase_data[np.nanargmin(flux_data)])
            p_dense = np.linspace(dip_center - HALF_WINDOW, dip_center + HALF_WINDOW, DENSE_SAMPLES, dtype=np.float32)
            with torch.no_grad():
                _ = model(torch.from_numpy(featurize(p_dense))).squeeze(1).cpu().numpy()
            transit_start, transit_end = find_tight_transit_boundaries(phase_data, flux_data)
            if transit_start < transit_end:
                p_dense = np.linspace(transit_start, transit_end, DENSE_SAMPLES, dtype=np.float32)
            else:
                p_dense = np.concatenate([
                    np.linspace(transit_start, 1.0, DENSE_SAMPLES // 2, dtype=np.float32),
                    np.linspace(0.0, transit_end, DENSE_SAMPLES // 2, dtype=np.float32)
                ])
            with torch.no_grad():
                y_nn_dense = model(torch.from_numpy(featurize(p_dense))).squeeze(1).cpu().numpy()
            results = analyze_transit_dip(model, phase_data, flux_data, p_dense, y_nn_dense, target['period_days'])
            R_s_m, _ = fetch_stellar_radius(target['name'])
            R_p_m = R_s_m * results['rp_over_rs_est'] if results['transit_detected'] and not np.isnan(R_s_m) else float("nan")
            theoretical_area_seconds = float('nan')
            absolute_difference = float('nan')
            relative_difference_percent = float('nan')
            agreement_score = float('nan')
            final_label = "NO_COMPARISON"
            if (results['transit_detected'] and not np.isnan(R_s_m) and not np.isnan(R_p_m)
                and 'width_time_seconds' in results and results['width_time_seconds'] > 0):
                try:
                    tc = compare_theoretical_nn_area(
                        R_s=R_s_m, R_p=R_p_m, tau=results['width_time_seconds'],
                        t0=results['t0_seconds'], nn_area_seconds=results['area_time'],
                        target_name=target['name'], disposition=disposition
                    )
                    if tc:
                        theoretical_area_seconds = tc['theoretical_area_seconds']
                        absolute_difference = tc['absolute_difference']
                        relative_difference_percent = tc['relative_difference_percent']
                        agreement_score = tc['agreement_score']
                        final_label = tc['final_label']
                except Exception:
                    pass
            ml_results = {
                'transit_detected': bool(results['transit_detected']),
                'transit_confidence': self._calculate_confidence(results, agreement_score),
                'agreement_score': 0.0 if np.isnan(agreement_score) else round(float(agreement_score), 3),
                'final_label': final_label,
                'model_loss': float(best_loss),
                'area_time_seconds': float(results['area_time']),
                'width_time_seconds': float(results.get('width_time_seconds', 0)),
                'equivalent_depth': float(results['depth_eq']),
                'rp_over_rs': float(results['rp_over_rs_est']),
                'stellar_radius_m': None if np.isnan(R_s_m) else float(R_s_m),
                'planet_radius_m': None if np.isnan(R_p_m) else float(R_p_m),
                'data_points': int(len(phase_data)),
                'from_cache': bool(from_cache),
                'disposition': disposition,
                'analysis_status': 'success',
                'model_used': 'pinn_transit',
            }
            return ml_results
        except Exception as e:
            return self._create_fallback_result(target, 'Unknown', f"ANALYSIS_ERROR: {str(e)}")

    def _calculate_confidence(self, results: Dict, agreement_score: float) -> float:
        base_confidence = 50.0
        if results.get('transit_detected'):
            base_confidence += 30.0
        if agreement_score is not None and not np.isnan(agreement_score):
            base_confidence += (float(agreement_score) / 100.0) * 20.0
        if results.get('equivalent_depth', 0) and results['equivalent_depth'] > 0.01:
            base_confidence += 10.0
        return float(min(100.0, max(0.0, base_confidence)))

    def _simulate_analysis(self, target: Dict) -> Dict:
        name = (target.get('name') or '').lower()
        if any(known in name for known in ['kepler', 'trappist', 'proxima', 'hd', 'wasp', 'hat']):
            transit_detected = True
            confidence = 85 + random.random() * 15
            agreement = 80 + random.random() * 20
            final_label = "CONFIRMED_EXOPLANET"
        else:
            transit_detected = random.random() > 0.3
            confidence = 40 + random.random() * 50
            agreement = 50 + random.random() * 40
            final_label = "CANDIDATE_EXOPLANET" if transit_detected else "UNCERTAIN_SIGNAL"
        return {
            'transit_detected': transit_detected,
            'transit_confidence': round(confidence, 1),
            'agreement_score': round(agreement, 1),
            'final_label': final_label,
            'model_loss': float(random.random() * 0.1),
            'area_time_seconds': float(random.random() * 10000),
            'width_time_seconds': float(random.random() * 1000),
            'equivalent_depth': float(random.random() * 0.1),
            'rp_over_rs': float(random.random() * 0.1),
            'analysis_status': 'simulated',
            'model_used': 'simulation_fallback',
            'disposition': 'Unknown'
        }

    def _create_fallback_result(self, target: Dict, disposition: str, status: str) -> Dict:
        return {
            'transit_detected': False,
            'transit_confidence': 0.0,
            'agreement_score': 0.0,
            'final_label': 'ANALYSIS_FAILED',
            'model_loss': float('nan'),
            'area_time_seconds': float('nan'),
            'width_time_seconds': float('nan'),
            'equivalent_depth': float('nan'),
            'rp_over_rs': float('nan'),
            'analysis_status': status,
            'model_used': 'error_fallback',
            'disposition': disposition
        }

nasa_api = NASAExoplanetAPI()
ml_analyzer = MLExoplanetAnalyzer()

popular_exoplanets_cache = None
CACHE_DURATION = 3600
cache_timestamp = 0

def get_popular_exoplanets():
    global popular_exoplanets_cache, cache_timestamp
    current_time = time.time()
    if popular_exoplanets_cache is None or (current_time - cache_timestamp > CACHE_DURATION):
        df = nasa_api.get_popular_exoplanets(limit=50)
        popular_exoplanets_cache = []
        if not df.empty:
            seen = set()
            for _, row in df.iterrows():
                planet_name = (row.get("pl_name") or "").strip()
                if planet_name and planet_name not in seen:
                    seen.add(planet_name)
                    popular_exoplanets_cache.append({
                        "name": planet_name,
                        "name_lc": planet_name.lower(),
                        "hostname": row.get("hostname", "") or "",
                        "radius_earth": _f(row.get("pl_rade"))
                    })
        else:
            popular_exoplanets_cache = [
                {"name": "Kepler-186f", "name_lc": "kepler-186f", "hostname": "Kepler-186", "radius_earth": 1.17},
                {"name": "TRAPPIST-1e", "name_lc": "trappist-1e", "hostname": "TRAPPIST-1", "radius_earth": 0.92},
                {"name": "Proxima Centauri b", "name_lc": "proxima centauri b", "hostname": "Proxima Centauri", "radius_earth": 1.07},
            ]
        cache_timestamp = current_time
    return popular_exoplanets_cache

@app.route('/search', methods=['POST'])
def search_planet():
    try:
        q = _get_param(request, 'planet_name', '').strip() or _get_param(request, 'q', '').strip()
        if not q:
            return jsonify(_json_safe({'error': 'Provide planet name, TOI (e.g., TOI-1000.01), or TIC ID'})), 400
        mode = _parse_query_type(q)
        source = None
        row = None
        if mode['kind'] == 'tic':
            df_ps = nasa_api.get_ps_by_tic_id(mode['value'])
            if not df_ps.empty:
                source = 'ps'
                row = df_ps.iloc[0]
            else:
                df_toi = nasa_api.get_toi_by_tic_id(mode['value'])
                if not df_toi.empty:
                    source = 'toi'
                    row = df_toi.iloc[0]
        elif mode['kind'] == 'toi_full':
            df_toi = nasa_api.get_toi_by_full_id(mode['value'])
            if not df_toi.empty:
                source = 'toi'
                row = df_toi.iloc[0]
        elif mode['kind'] == 'toi_short':
            df_toi = nasa_api.get_toi_by_short(mode['value'])
            if not df_toi.empty:
                source = 'toi'
                row = df_toi.iloc[0]
        elif mode['kind'] == 'name':
            df_ps = nasa_api.get_ps_by_name(mode['value'])
            if not df_ps.empty:
                source = 'ps'
                row = df_ps.iloc[0]
            else:
                digits = "".join(ch for ch in mode['value'] if ch.isdigit())
                if digits:
                    df_toi = nasa_api.get_toi_by_tic_id(int(digits))
                    if not df_toi.empty:
                        source = 'toi'
                        row = df_toi.iloc[0]
        if row is None:
            return jsonify(_json_safe({'is_planet': False, 'query': q, 'reason': 'NO_MATCH'}))
        if source == 'ps':
            name = row.get('pl_name') or 'Unknown'
            tic = _format_tic(row.get('tic_id'))
            target = {
                'name': name,
                'hostname': row.get("hostname", "Unknown") or "Unknown",
                'tic_id': tic,
                'period_days': _f(row.get("pl_orbper")),
                'radius_earth': _f(row.get("pl_rade")),
                'radius_jupiter': _f(row.get("pl_radj")),
                'transit_depth_ppm': _f(row.get("pl_trandep")),
                'stellar_radius': _f(row.get("st_rad")),
                'stellar_teff': _f(row.get("st_teff")),
                'distance_pc': _f(row.get("sy_dist")),
            }
            display = {
                'source': 'ps',
                'is_planet': True,
                'name': target['name'],
                'hostname': target['hostname'],
                'tic_id': target['tic_id'] or '-',
                'period_days': _format_value(target['period_days'], "days"),
                'radius_earth': _format_value(target['radius_earth'], "R⊕"),
                'radius_jupiter': _format_value(target['radius_jupiter'], "R♃"),
                'transit_depth_ppm': _format_value(target['transit_depth_ppm'], "ppm"),
                'stellar_radius': _format_value(target['stellar_radius'], "R☉"),
                'stellar_teff': _format_value(target['stellar_teff'], "K"),
                'distance_pc': _format_value(target['distance_pc'], "pc"),
                'tess_mag': _format_value(_f(row.get("sy_tmag"))),
            }
        else:
            name = str(row.get('full_toi_id') or f"TOI {row.get('toi') or ''}").strip()
            tic = _format_tic(row.get('tic_id'))
            target = {
                'name': name,
                'hostname': 'Unknown',
                'tic_id': tic,
                'period_days': _f(row.get("pl_orbper")),
                'radius_earth': _f(row.get("pl_rade")),
                'radius_jupiter': None,
                'transit_depth_ppm': _f(row.get("pl_trandep")),
                'stellar_radius': _f(row.get("st_rad")),
                'stellar_teff': _f(row.get("st_teff")),
                'distance_pc': _f(row.get("st_dist")),
            }
            display = {
                'source': 'toi',
                'is_planet': True,
                'name': target['name'],
                'hostname': target['hostname'],
                'tic_id': target['tic_id'] or '-',
                'period_days': _format_value(target['period_days'], "days"),
                'radius_earth': _format_value(target['radius_earth'], "R⊕"),
                'radius_jupiter': "-",
                'transit_depth_ppm': _format_value(target['transit_depth_ppm'], "ppm"),
                'stellar_radius': _format_value(target['stellar_radius'], "R☉"),
                'stellar_teff': _format_value(target['stellar_teff'], "K"),
                'distance_pc': _format_value(target['distance_pc'], "pc"),
                'tess_mag': _format_value(_f(row.get("tmag"))),
                'tfopwg_disp': row.get("tfopwg_disp"),
                'disposition': row.get("disposition"),
            }
        parts = []
        if display.get('hostname') and display['hostname'] != "Unknown":
            parts.append(f"orbiting the star {display['hostname']}")
        if display.get('period_days') and display['period_days'] != "-":
            parts.append(f"with an orbital period of {display['period_days']}")
        if display.get('radius_earth') and display['radius_earth'] != "-":
            parts.append(f"and a radius of {display['radius_earth']}")
        if display.get('distance_pc') and display['distance_pc'] != "-":
            parts.append(f"located approximately {display['distance_pc']} from Earth")
        desc = f"{display['name']} is an exoplanet candidate" if source=='toi' else f"{display['name']} is an exoplanet"
        if parts:
            desc += " " + ". ".join(parts)
        display['description'] = desc.strip() + "."
        if source == 'toi' and str(display.get('tfopwg_disp','')).upper() == 'FP':
            final = {**display, 'is_planet': False, 'analysis_status': 'skipped_fp', 'final_label': 'FALSE_POSITIVE'}
            return jsonify(_json_safe(final))
        ml_results = ml_analyzer.analyze_planet(target)
        final_result = {**display, **ml_results}
        return jsonify(_json_safe(final_result))
    except Exception as e:
        return jsonify(_json_safe({'error': 'Internal server error', 'is_planet': False})), 500

@app.route('/analyze', methods=['POST'])
def analyze_planet():
    try:
        planet_data = request.get_json(silent=True) or {}
        if not planet_data:
            return jsonify(_json_safe({'error': 'Planet data is required'})), 400
        analysis_target = {
            'name': planet_data.get('name'),
            'hostname': planet_data.get('hostname'),
            'tic_id': planet_data.get('tic_id'),
            'period_days': _f(planet_data.get('period_days')),
            'radius_earth': _f(planet_data.get('radius_earth')),
            'transit_depth_ppm': _f(planet_data.get('transit_depth_ppm')),
            'stellar_radius': _f(planet_data.get('stellar_radius')),
        }
        ml_results = ml_analyzer.analyze_planet(analysis_target)
        return jsonify(_json_safe({
            'analysis_id': f"ml_{int(time.time())}",
            'analysis_timestamp': time.time(),
            **ml_results
        }))
    except Exception as e:
        return jsonify(_json_safe({'error': f'Analysis failed: {str(e)}'})), 500

@app.route('/tess/fp', methods=['GET'])
def tess_false_positives():
    try:
        limit = int(request.args.get('limit', 100))
        df = nasa_api.get_tess_false_positives(limit=limit)
        return jsonify(_json_safe(df.to_dict(orient='records')))
    except Exception:
        return jsonify(_json_safe({'error': 'Failed to fetch TESS false positives'})), 500

@app.route('/tess/toi', methods=['GET'])
def tess_toi_by_tic():
    try:
        tic_id = request.args.get('tic_id')
        if not tic_id:
            return jsonify(_json_safe({'error': 'tic_id is required'})), 400
        tic_id_int = int(''.join(ch for ch in str(tic_id) if ch.isdigit()))
        df = nasa_api.get_toi_by_tic_id(tic_id_int)
        if df.empty:
            return jsonify(_json_safe({'found': False, 'tic_id': tic_id_int}))
        return jsonify(_json_safe({'found': True, 'data': df.iloc[0].to_dict()}))
    except Exception:
        return jsonify(_json_safe({'error': 'Failed to fetch TOI by TIC ID'})), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    try:
        query = (request.args.get('q') or '').lower().strip()
        if len(query) < 2:
            return jsonify(_json_safe([]))
        popular_planets = get_popular_exoplanets()
        suggestions = [p['name'] for p in popular_planets if query in p['name_lc']]
        return jsonify(_json_safe(suggestions[:10]))
    except Exception:
        return jsonify(_json_safe([]))

@app.route('/popular', methods=['GET'])
def get_popular():
    try:
        popular_planets = get_popular_exoplanets()
        return jsonify(_json_safe([planet['name'] for planet in popular_planets[:20]]))
    except Exception:
        return jsonify(_json_safe(["Kepler-186f", "TRAPPIST-1e", "Proxima Centauri b"]))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(_json_safe({
        'status': 'healthy',
        'timestamp': time.time(),
        'ml_system_available': ml_analyzer.initialized,
        'nasa_api_available': True,
        'cache_size': len(popular_exoplanets_cache) if popular_exoplanets_cache else 0
    }))

@app.route('/system-info', methods=['GET'])
def system_info():
    return jsonify(_json_safe({
        'ml_system': {
            'available': ml_analyzer.initialized,
            'description': 'Neural Network Exoplanet Transit Analyzer',
            'version': '2.1',
            'features': ['light_curve_analysis', 'transit_detection', 'theoretical_validation'],
            'status': 'operational' if ml_analyzer.initialized else 'simulation_mode'
        },
        'data_sources': {
            'nasa_exoplanet_archive': True,
            'light_curve_cache': True,
            'stellar_parameters': True
        }
    }))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    get_popular_exoplanets()
    app.run(debug=True, host='0.0.0.0', port=3000)