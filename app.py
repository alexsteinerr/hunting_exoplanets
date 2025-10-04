from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
import re

from flask_cors import CORS
import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
import re
import urllib.parse

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class NASAExoplanetAPI:
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    def __init__(self, pause_s: float = 0.5):
        self.session = requests.Session()
        self.pause_s = pause_s

    def _run_adql(self, adql: str, timeout: int = 120) -> pd.DataFrame:
        """Execute ADQL query with proper error handling"""
        try:
            # Clean up the query - remove extra whitespace and ensure proper formatting
            adql = ' '.join(adql.split())  # Normalize whitespace
            
            params = {
                "query": adql,
                "format": "json"
            }
            
            print(f"[ADQL] Executing query: {adql[:100]}...")
            
            r = self.session.get(self.BASE_URL, params=params, timeout=timeout)
            r.raise_for_status()
            
            data = r.json()
            return pd.DataFrame(data) if isinstance(data, list) and len(data) else pd.DataFrame()
            
        except requests.exceptions.HTTPError as e:
            print(f"[ADQL ERROR] HTTP Error: {e}")
            print(f"[ADQL ERROR] Response: {e.response.text if e.response else 'No response'}")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ADQL ERROR] General error: {e}")
            return pd.DataFrame()

    def search_planet_by_name(self, planet_name: str) -> pd.DataFrame:
        """Search for a specific planet by name with proper ADQL syntax"""
        # Clean the planet name for SQL
        planet_name_clean = planet_name.replace("'", "''")
        
        adql = f"""
        SELECT 
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
        AND LOWER(pl_name) LIKE LOWER('%{planet_name_clean}%')
        ORDER BY pl_name
        """
        return self._run_adql(adql)

    def get_popular_exoplanets(self, limit: int = 50) -> pd.DataFrame:
        """Get a list of popular exoplanets for suggestions"""
        adql = f"""
        SELECT 
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
        LIMIT {limit}
        """
        return self._run_adql(adql)

def _format_tic(val):
    """Format TIC ID properly"""
    if pd.isna(val):
        return None
    try:
        s = str(val).strip()
        if s.upper().startswith('TIC'):
            return s
        # Try to extract numeric part
        digits = ''.join(filter(str.isdigit, s))
        if digits:
            return f"TIC {digits}"
        return None
    except Exception:
        return None

def _f(val):
    """Safe float conversion"""
    try:
        if pd.isna(val) or val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None

def _format_value(val, unit=""):
    """Format numeric values with appropriate precision"""
    if val is None:
        return "-"
    try:
        num_val = float(val) if not isinstance(val, (int, float)) else val
        if num_val < 0.01:
            return f"{num_val:.2e} {unit}".strip()
        elif num_val < 1:
            return f"{num_val:.3f} {unit}".strip()
        elif num_val < 1000:
            return f"{num_val:.2f} {unit}".strip()
        else:
            return f"{num_val:.1f} {unit}".strip()
    except (TypeError, ValueError):
        return str(val) if val else "-"

# Mock data for fallback when API is unavailable
MOCK_EXOPLANET_DATA = {
    "kepler-186f": {
        "name": "Kepler-186f",
        "hostname": "Kepler-186",
        "tic_id": "TIC 12419272",
        "period_days": 129.9,
        "radius_earth": 1.17,
        "radius_jupiter": 0.104,
        "stellar_radius": 0.52,
        "stellar_teff": 3755,
        "distance_pc": 492,
        "description": "Kepler-186f is an exoplanet orbiting the red dwarf Kepler-186, about 492 light-years from Earth. It was the first rocky planet to be found within the habitable zone of another star."
    },
    "trappist-1e": {
        "name": "TRAPPIST-1e",
        "hostname": "TRAPPIST-1",
        "tic_id": "TIC 188547431",
        "period_days": 6.10,
        "radius_earth": 0.92,
        "radius_jupiter": 0.082,
        "stellar_radius": 0.117,
        "stellar_teff": 2559,
        "distance_pc": 39.5,
        "description": "TRAPPIST-1e is an exoplanet orbiting within the habitable zone of the ultracool dwarf star TRAPPIST-1, located about 39 light-years away from Earth."
    },
    "proxima centauri b": {
        "name": "Proxima Centauri b",
        "hostname": "Proxima Centauri",
        "tic_id": "TIC 234712277",
        "period_days": 11.19,
        "radius_earth": 1.07,
        "radius_jupiter": 0.095,
        "stellar_radius": 0.141,
        "stellar_teff": 3050,
        "distance_pc": 1.30,
        "description": "Proxima Centauri b is an exoplanet orbiting within the habitable zone of the red dwarf star Proxima Centauri, the closest star to the Sun."
    }
}

# Cache for popular exoplanets
popular_exoplanets_cache = None
cache_timestamp = None
CACHE_DURATION = 3600  # 1 hour

def get_popular_exoplanets():
    """Get cached popular exoplanets with fallback to mock data"""
    global popular_exoplanets_cache, cache_timestamp
    
    current_time = time.time()
    if popular_exoplanets_cache is None or (current_time - cache_timestamp > CACHE_DURATION):
        print("[CACHE] Attempting to refresh exoplanet cache from API...")
        api = NASAExoplanetAPI()
        df = api.get_popular_exoplanets(limit=50)
        popular_exoplanets_cache = []
        
        if not df.empty:
            for _, row in df.iterrows():
                planet_name = row.get("pl_name", "").strip()
                if planet_name and planet_name not in [p["name"] for p in popular_exoplanets_cache]:
                    popular_exoplanets_cache.append({
                        "name": planet_name,
                        "hostname": row.get("hostname", ""),
                        "radius_earth": _f(row.get("pl_rade"))
                    })
            print(f"[CACHE] Successfully cached {len(popular_exoplanets_cache)} exoplanets from API")
        else:
            # Fallback to mock data
            print("[CACHE] Using mock data as fallback")
            popular_exoplanets_cache = [
                {"name": "Kepler-186f", "hostname": "Kepler-186", "radius_earth": 1.17},
                {"name": "TRAPPIST-1e", "hostname": "TRAPPIST-1", "radius_earth": 0.92},
                {"name": "Proxima Centauri b", "hostname": "Proxima Centauri", "radius_earth": 1.07},
                {"name": "Kepler-452b", "hostname": "Kepler-452", "radius_earth": 1.63},
                {"name": "GJ 667 Cc", "hostname": "GJ 667 C", "radius_earth": 1.54},
                {"name": "HD 209458 b", "hostname": "HD 209458", "radius_earth": 1.35},
                {"name": "WASP-12b", "hostname": "WASP-12", "radius_earth": 1.90},
                {"name": "GJ 1214b", "hostname": "GJ 1214", "radius_earth": 2.68},
                {"name": "HD 189733 b", "hostname": "HD 189733", "radius_earth": 1.14},
                {"name": "Kepler-22b", "hostname": "Kepler-22", "radius_earth": 2.10}
            ]
        
        cache_timestamp = current_time
    
    return popular_exoplanets_cache

@app.route('/search', methods=['POST'])
def search_planet():
    """Search for exoplanet data"""
    try:
        planet_name = request.form.get('planet_name', '').strip()
        if not planet_name:
            return jsonify({'error': 'Planet name is required'}), 400

        print(f"[SEARCH] Searching for: {planet_name}")
        
        # First check mock data for common planets
        planet_key = planet_name.lower().replace(' ', '-')
        if planet_key in MOCK_EXOPLANET_DATA:
            print(f"[SEARCH] Found in mock data: {planet_key}")
            target = MOCK_EXOPLANET_DATA[planet_key]
            target['is_planet'] = True
            return jsonify(target)
        
        # Try API search
        api = NASAExoplanetAPI()
        df = api.search_planet_by_name(planet_name)
        
        if not df.empty:
            # Get the first matching result
            row = df.iloc[0]
            print(f"[SEARCH] API found: {row.get('pl_name', 'Unknown')}")
            
            # Format the response
            target = {
                'is_planet': True,
                'name': row.get("pl_name", planet_name),
                'hostname': row.get("hostname", "Unknown"),
                'tic_id': _format_tic(row.get("tic_id")),
                'period_days': _format_value(_f(row.get("pl_orbper")), "days"),
                'radius_earth': _format_value(_f(row.get("pl_rade")), "R⊕"),
                'radius_jupiter': _format_value(_f(row.get("pl_radj")), "R♃"),
                'transit_depth_ppm': _format_value(_f(row.get("pl_trandep")), "ppm"),
                'stellar_radius': _format_value(_f(row.get("st_rad")), "R☉"),
                'stellar_teff': _format_value(_f(row.get("st_teff")), "K"),
                'distance_pc': _format_value(_f(row.get("sy_dist")), "pc"),
                'tess_mag': _format_value(_f(row.get("sy_tmag"))),
            }
            
            # Generate description
            description_parts = [f"{target['name']} is an exoplanet"]
            if target['hostname'] != "Unknown":
                description_parts.append(f"orbiting the star {target['hostname']}")
            if target['period_days'] != "-":
                description_parts.append(f"with an orbital period of {target['period_days']}")
            if target['radius_earth'] != "-":
                description_parts.append(f"and a radius of {target['radius_earth']}")
            if target['distance_pc'] != "-":
                description_parts.append(f"located approximately {target['distance_pc']} from Earth")
            
            target['description'] = '. '.join(description_parts) + '.'
            
            return jsonify(target)
        else:
            print(f"[SEARCH] No API results for: {planet_name}")
            return jsonify({'is_planet': False})
        
    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        # Fallback to mock data if available
        planet_key = planet_name.lower().replace(' ', '-')
        if planet_key in MOCK_EXOPLANET_DATA:
            target = MOCK_EXOPLANET_DATA[planet_key]
            target['is_planet'] = True
            return jsonify(target)
        return jsonify({'error': 'Internal server error', 'is_planet': False}), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get exoplanet name suggestions for autocomplete"""
    try:
        query = request.args.get('q', '').lower().strip()
        if len(query) < 2:
            return jsonify([])
        
        popular_planets = get_popular_exoplanets()
        
        # Filter by query
        suggestions = [
            planet['name'] for planet in popular_planets 
            if query in planet['name'].lower()
        ]
        
        # Return top 10 suggestions
        return jsonify(suggestions[:10])
        
    except Exception as e:
        print(f"[ERROR] Suggestions failed: {str(e)}")
        # Fallback to basic suggestions
        basic_suggestions = ["Kepler-186f", "TRAPPIST-1e", "Proxima Centauri b", "Kepler-452b"]
        return jsonify([s for s in basic_suggestions if query in s.lower()][:10])

@app.route('/popular', methods=['GET'])
def get_popular():
    """Get list of popular exoplanets"""
    try:
        popular_planets = get_popular_exoplanets()
        return jsonify([planet['name'] for planet in popular_planets[:20]])
    except Exception as e:
        print(f"[ERROR] Popular planets failed: {str(e)}")
        return jsonify(["Kepler-186f", "TRAPPIST-1e", "Proxima Centauri b", "Kepler-452b"])

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': time.time(),
        'cache_size': len(popular_exoplanets_cache) if popular_exoplanets_cache else 0
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Pre-populate cache on startup
    print("[STARTUP] Pre-populating exoplanet cache...")
    get_popular_exoplanets()
    
    print("[STARTUP] Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=3000)