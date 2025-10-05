import requests
import pandas as pd
import numpy as np
import lightkurve as lk
import time
from typing import List, Dict, Optional

def query_tess_false_positives(limit: int = 50) -> pd.DataFrame:
    """Query NASA Exoplanet Archive for TESS false positives"""
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    query = f"""
    SELECT TOP {limit}
        toi, full_toi_id, tic_id, tfopwg_disp, disposition,
        ra, dec, tmag, pl_tranmid, pl_orbper, pl_trandur, 
        pl_trandep, pl_rade, pl_insol, pl_eqt, st_dist, 
        st_teff, st_logg, st_rad
    FROM toi
    WHERE tfopwg_disp = 'FP' OR disposition = 'FALSE POSITIVE'
    ORDER BY tic_id
    """
    
    params = {
        "query": query,
        "format": "json"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error querying NASA archive: {e}")
        return pd.DataFrame()

def check_lightcurve_availability(tic_id: int) -> bool:
    """Check if TESS light curves are available for a TIC ID"""
    try:
        print(f"  Checking TIC {tic_id}...")
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
        return len(search_result) > 0
    except Exception as e:
        print(f"    Error searching TIC {tic_id}: {e}")
        return False

def get_lightcurve_info(tic_id: int) -> Dict:
    """Get detailed light curve information for a TIC ID"""
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
        if len(search_result) > 0:
            # Get the first light curve to check details
            lc = search_result[0].download()
            return {
                'available': True,
                'sectors': list(search_result.table['mission']),
                'cadence': search_result.table['exptime'][0],
                'data_points': len(lc.flux),
                'duration_days': lc.time.value[-1] - lc.time.value[0] if len(lc.time) > 1 else 0
            }
        return {'available': False}
    except Exception as e:
        print(f"    Error getting light curve for TIC {tic_id}: {e}")
        return {'available': False, 'error': str(e)}

def find_false_positives_with_lightcurves(max_tests: int = 20) -> List[Dict]:
    """Find false positives that have TESS light curve data available"""
    print("Querying NASA Exoplanet Archive for TESS false positives...")
    df_fp = query_tess_false_positives(limit=100)
    
    if df_fp.empty:
        print("No false positives found in NASA archive!")
        return []
    
    print(f"Found {len(df_fp)} false positives in catalog")
    
    # Filter for promising candidates (bright stars, known parameters)
    potential_fps = []
    for _, row in df_fp.iterrows():
        tic_id = row.get('tic_id')
        if tic_id and not pd.isna(tic_id):
            tmag = row.get('tmag')
            # Prioritize brighter stars and those with known periods
            if tmag and tmag < 13:  # Brighter than 13th magnitude
                potential_fps.append({
                    'tic_id': int(tic_id),
                    'toi': row.get('toi'),
                    'full_toi_id': row.get('full_toi_id'),
                    'tmag': tmag,
                    'period_days': row.get('pl_orbper'),
                    'radius_earth': row.get('pl_rade'),
                    'transit_depth_ppm': row.get('pl_trandep'),
                    'disposition': row.get('tfopwg_disp') or row.get('disposition')
                })
    
    # If no bright targets, take first few
    if not potential_fps:
        for _, row in df_fp.head(20).iterrows():
            tic_id = row.get('tic_id')
            if tic_id and not pd.isna(tic_id):
                potential_fps.append({
                    'tic_id': int(tic_id),
                    'toi': row.get('toi'),
                    'full_toi_id': row.get('full_toi_id'),
                    'tmag': row.get('tmag'),
                    'period_days': row.get('pl_orbper'),
                    'radius_earth': row.get('pl_rade'),
                    'transit_depth_ppm': row.get('pl_trandep'),
                    'disposition': row.get('tfopwg_disp') or row.get('disposition')
                })
    
    print(f"\nTesting {min(len(potential_fps), max_tests)} potential false positives for light curve availability...")
    
    fps_with_data = []
    tested_count = 0
    
    for fp in potential_fps:
        if tested_count >= max_tests:
            break
            
        tic_id = fp['tic_id']
        tested_count += 1
        
        if check_lightcurve_availability(tic_id):
            print(f"  ✓ TIC {tic_id} has light curve data!")
            lc_info = get_lightcurve_info(tic_id)
            
            fp_with_data = fp.copy()
            fp_with_data['light_curve_info'] = lc_info
            fps_with_data.append(fp_with_data)
            
            # Small delay to be nice to the servers
            time.sleep(0.5)
        else:
            print(f"  ✗ TIC {tic_id} has no light curve data")
    
    return fps_with_data

def print_fp_details(fp: Dict):
    """Print detailed information about a false positive"""
    print("\n" + "="*80)
    print("FALSE POSITIVE WITH TESS LIGHT CURVE")
    print("="*80)
    print(f"TOI: {fp.get('full_toi_id', fp.get('toi', 'Unknown'))}")
    print(f"TIC ID: {fp['tic_id']}")
    print(f"TESS Magnitude: {fp.get('tmag', 'Unknown')}")
    print(f"Orbital Period: {fp.get('period_days', 'Unknown')} days")
    print(f"Planet Radius: {fp.get('radius_earth', 'Unknown')} R⊕")
    print(f"Transit Depth: {fp.get('transit_depth_ppm', 'Unknown')} ppm")
    print(f"Disposition: {fp.get('disposition', 'Unknown')}")
    
    lc_info = fp.get('light_curve_info', {})
    if lc_info.get('available'):
        print(f"\nLight Curve Information:")
        print(f"  Sectors: {len(lc_info.get('sectors', []))}")
        print(f"  Cadence: {lc_info.get('cadence', 'Unknown')} seconds")
        print(f"  Data Points: {lc_info.get('data_points', 'Unknown')}")
        print(f"  Duration: {lc_info.get('duration_days', 'Unknown'):.1f} days")
    else:
        print(f"\nLight Curve: Not available")
    
    print("="*80)

def download_sample_lightcurve(tic_id: int, save_path: Optional[str] = None):
    """Download and display a sample light curve for a false positive"""
    try:
        print(f"\nDownloading light curve for TIC {tic_id}...")
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
        
        if len(search_result) > 0:
            print(f"Found {len(search_result)} light curve(s)")
            
            # Download the first one
            lc = search_result[0].download()
            
            print(f"Light curve details:")
            print(f"  Time range: {lc.time.value[0]:.2f} to {lc.time.value[-1]:.2f} BTJD")
            print(f"  Flux points: {len(lc.flux)}")
            print(f"  Median flux: {np.nanmedian(lc.flux.value):.2f}")
            print(f"  Flux std: {np.nanstd(lc.flux.value):.4f}")
            
            # Save if requested
            if save_path:
                lc.to_fits(save_path)
                print(f"  Saved to: {save_path}")
            
            return lc
        else:
            print("No light curves found")
            return None
            
    except Exception as e:
        print(f"Error downloading light curve: {e}")
        return None

def main():
    """Main function to find and display false positives with light curves"""
    print("TESS False Positive Finder")
    print("=" * 50)
    
    # Find false positives with light curves
    fps_with_data = find_false_positives_with_lightcurves(max_tests=15)
    
    if not fps_with_data:
        print("\n❌ No false positives with light curve data found!")
        print("This could be due to:")
        print("  - Network issues")
        print("  - TESS data server problems") 
        print("  - All tested TIC IDs being too faint or not observed")
        return
    
    print(f"\n🎉 Found {len(fps_with_data)} false positive(s) with TESS light curve data!")
    
    # Display details for each found FP
    for i, fp in enumerate(fps_with_data, 1):
        print(f"\n{'='*50}")
        print(f"FALSE POSITIVE {i}/{len(fps_with_data)}")
        print(f"{'='*50}")
        print_fp_details(fp)
        
        # Ask if user wants to download a sample
        response = input(f"\nDownload sample light curve for TIC {fp['tic_id']}? (y/n): ")
        if response.lower() in ['y', 'yes']:
            filename = f"TIC_{fp['tic_id']}_lightcurve.fits"
            lc = download_sample_lightcurve(fp['tic_id'], filename)
            
            if lc is not None:
                print(f"✓ Successfully downloaded light curve for TIC {fp['tic_id']}")
            else:
                print(f"✗ Failed to download light curve for TIC {fp['tic_id']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total false positives tested: 15")
    print(f"With light curve data: {len(fps_with_data)}")
    print(f"Success rate: {(len(fps_with_data)/15)*100:.1f}%")
    
    if fps_with_data:
        print(f"\nRecommended TIC IDs for your analysis:")
        for fp in fps_with_data:
            print(f"  - TIC {fp['tic_id']} (TOI {fp.get('full_toi_id', fp.get('toi', 'Unknown'))})")

if __name__ == "__main__":
    main()