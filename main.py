import random
import numpy as np
import torch
import pandas as pd
import time
import os
from tqdm import tqdm

from config.settings import *
from data.loader import CACHE_DIR, get_disposition_label, get_folded_light_curve
from utils.features import prepare_training_data, featurize
from models.mlp import MLP
from models.trainer import train_model
from analysis.transit import analyze_transit_dip, fetch_stellar_radius
from analysis.derivation import compare_theoretical_nn_area, analyze_multiple_targets_comparison
from analysis.boundaries import find_tight_transit_boundaries
from targets.target_list import get_exoplanet_targets

def analyze_single_target(target, target_index, total_targets):
    """Analyze a single exoplanet target"""
    print(f"\n[{target_index+1}/{total_targets}] Analyzing {target['name']}...")
    
    try:
        # Get disposition label (confirmed exoplanet vs false positive)
        disposition = get_disposition_label(target['name'], target.get('tic_id'))
        
        # Set seeds for reproducibility
        random.seed(SEED + target_index)
        np.random.seed(SEED + target_index)
        torch.manual_seed(SEED + target_index)

        # Load and process data WITH CACHING
        phase, flux, flux_err, from_cache = get_folded_light_curve(
            target['name'], 
            target.get('tic_id'), 
            target['period_days']
        )
        
        if from_cache:
            print(f"[CACHE] Using cached data for {target['name']}")
        else:
            print(f"[DOWNLOAD] Downloaded and cached data for {target['name']}")
        
        # Prepare training data
        X_np, y_np, err_np, phase_data, flux_data, flux_err_data = prepare_training_data(
            phase, flux, flux_err, USE_BINNING
        )
        
        # Skip if not enough data points
        if len(X_np) < 10:
            print(f"[WARN] Insufficient data for {target['name']}, skipping...")
            return None

        # Convert to tensors
        X_train = torch.from_numpy(X_np)
        y_train = torch.from_numpy(y_np).unsqueeze(1)
        err_train = torch.from_numpy(err_np).unsqueeze(1)

        # Initialize and train model
        model = MLP(in_dim=X_train.shape[1], h1=H1, h2=H2, out_dim=1)
        best_loss = train_model(model, X_train, y_train, err_train, EPOCHS, PATIENCE, LR, WEIGHT_DECAY, ALPHA_TRANSIT)

        # Analyze transit dip
        dip_center = float(phase_data[np.nanargmin(flux_data)])
        p_dense = np.linspace(dip_center - HALF_WINDOW, dip_center + HALF_WINDOW,
                              DENSE_SAMPLES, dtype=np.float32)
        
        with torch.no_grad():
            y_nn_dense = model(torch.from_numpy(featurize(p_dense))).squeeze(1).cpu().numpy()

        # Find tight transit boundaries
        transit_start, transit_end = find_tight_transit_boundaries(phase_data, flux_data)
        # Handle phase wrapping if needed
        if transit_start < transit_end:
            p_dense = np.linspace(transit_start, transit_end, DENSE_SAMPLES, dtype=np.float32)
        else:
            # Phase wraps around, so concatenate two ranges
            p_dense = np.concatenate([
                np.linspace(transit_start, 1.0, DENSE_SAMPLES//2, dtype=np.float32),
                np.linspace(0.0, transit_end, DENSE_SAMPLES//2, dtype=np.float32)
            ])

        with torch.no_grad():
            y_nn_dense = model(torch.from_numpy(featurize(p_dense))).squeeze(1).cpu().numpy()

        # Perform transit analysis
        results = analyze_transit_dip(model, phase_data, flux_data, p_dense, y_nn_dense, target['period_days'])
        
        # Fetch stellar parameters
        R_s_m, R_s_rsun = fetch_stellar_radius(target['name'])
        R_p_m = R_s_m * results['rp_over_rs_est'] if results['transit_detected'] and not np.isnan(R_s_m) else float("nan")

        # Initialize theoretical comparison fields with NaN
        theoretical_area_seconds = float('nan')
        absolute_difference = float('nan')
        relative_difference_percent = float('nan')
        agreement_score = float('nan')
        final_label = "NO_COMPARISON"
        assessment = "NO_DATA"

        # Calculate theoretical area and comparison if we have all required parameters
        theoretical_comparison = None
        if (results['transit_detected'] and 
            not np.isnan(R_s_m) and 
            not np.isnan(R_p_m) and
            'width_time_seconds' in results and
            results['width_time_seconds'] > 0):
            
            try:
                theoretical_comparison = compare_theoretical_nn_area(
                    R_s=R_s_m,
                    R_p=R_p_m,
                    tau=results['width_time_seconds'],
                    t0=results['t0_seconds'],
                    nn_area_seconds=results['area_time'],
                    target_name=target['name'],
                    disposition=disposition
                )
                
                # Update fields if comparison was successful
                if theoretical_comparison:
                    theoretical_area_seconds = theoretical_comparison['theoretical_area_seconds']
                    absolute_difference = theoretical_comparison['absolute_difference']
                    relative_difference_percent = theoretical_comparison['relative_difference_percent']
                    agreement_score = theoretical_comparison['agreement_score']
                    final_label = theoretical_comparison['final_label']
                    assessment = "COMPARISON_SUCCESS"
                    
            except Exception as e:
                print(f"[WARN] Theoretical comparison failed for {target['name']}: {e}")
                theoretical_comparison = None
                assessment = f"COMPARISON_FAILED: {str(e)}"

        # Compile results - ALWAYS include theoretical fields
        target_results = {
            'target_name': target['name'],
            'tic_id': target.get('tic_id', ''),
            'period_days': target['period_days'],
            'catalog_period': target.get('period_days'),
            'catalog_depth_ppm': target.get('transit_depth_ppm'),
            'catalog_radius_earth': target.get('radius_earth'),
            'stellar_radius_catalog': target.get('stellar_radius'),
            'model_loss': best_loss,
            'transit_detected': results['transit_detected'],
            'area_phase': results['area_phase'],
            'area_time_seconds': results['area_time'],
            'width_phase': results['width_phase'],
            'width_time_seconds': results.get('width_time_seconds', results.get('width_time', 0)),
            'equivalent_depth': results['depth_eq'],
            'rp_over_rs': results['rp_over_rs_est'],
            't0_phase': results['t0_phase'],
            't0_seconds': results['t0_seconds'],
            'stellar_radius_m': R_s_m,
            'stellar_radius_rsun': R_s_rsun,
            'planet_radius_m': R_p_m,
            'dip_center_phase': results['dip_center'],
            'data_points': len(phase_data),
            'from_cache': from_cache,
            'disposition': disposition,
            
            # ALWAYS include theoretical fields (will be NaN if comparison failed)
            'theoretical_area_seconds': theoretical_area_seconds,
            'absolute_difference': absolute_difference,
            'relative_difference_percent': relative_difference_percent,
            'agreement_score': agreement_score,
            'final_label': final_label,
            'comparison_status': assessment,
            'status': 'success'
        }
        
        # Only print basic status, detailed results will be shown later
        cache_status = "[CACHED]" if from_cache else "[DOWNLOADED]"
        disposition_status = f"[{disposition}]"
        if results['transit_detected']:
            print(f"✅ TRANSIT {cache_status} {disposition_status} for {target['name']}")
        else:
            print(f"❌ NO TRANSIT {cache_status} {disposition_status} for {target['name']}")
        
        return target_results

    except Exception as e:
        print(f"[ERROR] Failed to analyze {target['name']}: {str(e)}")
        if not SKIP_FAILED:
            raise e
        return {
            'target_name': target['name'],
            'tic_id': target.get('tic_id', ''),
            'period_days': target['period_days'],
            'status': f'failed: {str(e)}',
            'transit_detected': False,
            'area_phase': float('nan'),
            'area_time_seconds': float('nan'),
            'width_phase': float('nan'),
            'width_time_seconds': float('nan'),
            'equivalent_depth': float('nan'),
            'rp_over_rs': float('nan'),
            'from_cache': False,
            'disposition': 'Unknown',
            'theoretical_area_seconds': float('nan'),
            'absolute_difference': float('nan'),
            'relative_difference_percent': float('nan'),
            'agreement_score': float('nan'),
            'final_label': 'FAILED_ANALYSIS',
            'comparison_status': f'ANALYSIS_FAILED: {str(e)}'
        }

def print_comprehensive_results(results_df):
    """Print all results together in a comprehensive table after training"""
    
    successful_results = results_df[results_df['status'] == 'success']
    successful_with_transit = successful_results[successful_results['transit_detected'] == True]
    
    # SAFELY check for theoretical comparisons - use .get() to avoid KeyError
    theoretical_area_col = 'theoretical_area_seconds'
    if theoretical_area_col in successful_with_transit.columns:
        successful_with_comparison = successful_with_transit[successful_with_transit[theoretical_area_col].notna()]
    else:
        successful_with_comparison = pd.DataFrame()  # Empty DataFrame
    
    print("\n" + "="*140)
    print("COMPREHENSIVE RESULTS - ALL TARGETS")
    print("="*140)
    
    # Overall statistics
    total_targets = len(results_df)
    successful_analyses = len(successful_results)
    transit_detections = len(successful_with_transit)
    comparisons_available = len(successful_with_comparison)
    
    # Disposition statistics
    confirmed_exoplanets = len(results_df[results_df['disposition'] == 'Confirmed Exoplanet'])
    false_positives = len(results_df[results_df['disposition'] == 'False Positive'])
    unknown_disposition = len(results_df[results_df['disposition'] == 'Unknown'])
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total targets analyzed: {total_targets}")
    print(f"Successful analyses: {successful_analyses} ({successful_analyses/total_targets*100:.1f}%)")
    print(f"Targets with detected transits: {transit_detections} ({transit_detections/total_targets*100:.1f}%)")
    print(f"Targets with theoretical comparison: {comparisons_available} ({comparisons_available/total_targets*100:.1f}%)")
    print(f"Confirmed Exoplanets: {confirmed_exoplanets} ({confirmed_exoplanets/total_targets*100:.1f}%)")
    print(f"False Positives: {false_positives} ({false_positives/total_targets*100:.1f}%)")
    print(f"Unknown Disposition: {unknown_disposition} ({unknown_disposition/total_targets*100:.1f}%)")
    
    if len(successful_with_transit) > 0:
        # Print detailed table for all targets with transits
        print(f"\n" + "="*140)
        print(f"{'Target Name':<20} {'Area (s)':<10} {'Width (s)':<10} {'Depth':<8} {'Rp/Rs':<8} {'Agreement':<12} {'Final Label':<25} {'Disposition':<20}")
        print("="*140)
        
        # Sort by agreement score (best first), then by area
        if len(successful_with_comparison) > 0:
            successful_with_comparison_sorted = successful_with_comparison.sort_values(
                ['agreement_score', 'area_time_seconds'], ascending=[False, False]
            )
            
            # First print targets with theoretical comparison
            for _, result in successful_with_comparison_sorted.iterrows():
                area_seconds = result['area_time_seconds']
                width_seconds = result.get('width_time_seconds', 0)
                agreement_display = f"{result['agreement_score']:.1f}%" if not pd.isna(result['agreement_score']) else "N/A"
                final_label_display = result.get('final_label', 'NO_LABEL')[:24]
                disposition_display = result['disposition'][:19]
                
                print(f"{result['target_name']:<20} {area_seconds:9.2f} {width_seconds:9.2f} "
                      f"{result['equivalent_depth']:7.4f} {result['rp_over_rs']:7.4f} "
                      f"{agreement_display:>11} {final_label_display:<25} {disposition_display:<20}")
            
            # Then print targets without theoretical comparison but with transit
            targets_without_comparison = successful_with_transit[~successful_with_transit.index.isin(successful_with_comparison.index)]
            for _, result in targets_without_comparison.iterrows():
                area_seconds = result['area_time_seconds']
                width_seconds = result.get('width_time_seconds', 0)
                disposition_display = result['disposition'][:19]
                
                print(f"{result['target_name']:<20} {area_seconds:9.2f} {width_seconds:9.2f} "
                      f"{result['equivalent_depth']:7.4f} {result['rp_over_rs']:7.4f} "
                      f"{'N/A':>11} {'NO_COMPARISON':<25} {disposition_display:<20}")
        
        else:
            # If no theoretical comparisons, just sort by area
            successful_with_transit_sorted = successful_with_transit.sort_values('area_time_seconds', ascending=False)
            for _, result in successful_with_transit_sorted.iterrows():
                area_seconds = result['area_time_seconds']
                width_seconds = result.get('width_time_seconds', 0)
                disposition_display = result['disposition'][:19]
                
                print(f"{result['target_name']:<20} {area_seconds:9.2f} {width_seconds:9.2f} "
                      f"{result['equivalent_depth']:7.4f} {result['rp_over_rs']:7.4f} "
                      f"{'N/A':>11} {'NO_COMPARISON':<25} {disposition_display:<20}")
        
        # Print summary statistics for comparisons
        if len(successful_with_comparison) > 0:
            print(f"\nFINAL CLASSIFICATION SUMMARY ({len(successful_with_comparison)} targets):")
            
            # Count final labels safely
            final_labels = successful_with_comparison.get('final_label', pd.Series(['NO_LABEL'] * len(successful_with_comparison)))
            confirmed_exoplanets_final = len(final_labels[final_labels == 'CONFIRMED EXOPLANET'])
            confirmed_fp_final = len(final_labels[final_labels == 'CONFIRMED FALSE POSITIVE'])
            potential_exoplanets_final = len(final_labels[final_labels == 'POTENTIAL EXOPLANET'])
            potential_fp_final = len(final_labels[final_labels == 'POTENTIAL FALSE POSITIVE'])
            no_comparison_final = len(final_labels[final_labels == 'NO_COMPARISON'])
            
            print(f"Confirmed Exoplanets:        {confirmed_exoplanets_final}/{len(successful_with_comparison)}")
            print(f"Confirmed False Positives:   {confirmed_fp_final}/{len(successful_with_comparison)}")
            print(f"Potential Exoplanets:        {potential_exoplanets_final}/{len(successful_with_comparison)}")
            print(f"Potential False Positives:   {potential_fp_final}/{len(successful_with_comparison)}")
            print(f"No Comparison Available:     {no_comparison_final}/{len(successful_with_comparison)}")
            
            # Agreement statistics
            agreements = successful_with_comparison['agreement_score'].dropna()
            if len(agreements) > 0:
                print(f"\nAgreement Statistics:")
                print(f"Mean Agreement Score: {agreements.mean():.1f}% ± {agreements.std():.1f}%")
                print(f"Median Agreement: {agreements.median():.1f}%")
                
                # Count agreement levels
                excellent = len(agreements[agreements >= 95])
                very_good = len(agreements[(agreements >= 90) & (agreements < 95)])
                good = len(agreements[(agreements >= 85) & (agreements < 90)])
                fair = len(agreements[(agreements >= 80) & (agreements < 85)])
                poor = len(agreements[agreements < 80])
                
                print(f"\nAgreement Levels:")
                print(f"  Excellent (≥95%): {excellent}/{len(agreements)} ({excellent/len(agreements)*100:.1f}%)")
                print(f"  Very Good (90-94%): {very_good}/{len(agreements)} ({very_good/len(agreements)*100:.1f}%)")
                print(f"  Good (85-89%): {good}/{len(agreements)} ({good/len(agreements)*100:.1f}%)")
                print(f"  Fair (80-84%): {fair}/{len(agreements)} ({fair/len(agreements)*100:.1f}%)")
                print(f"  Poor (<80%): {poor}/{len(agreements)} ({poor/len(agreements)*100:.1f}%)")
    
    else:
        print("No transits detected in any of the analyzed systems.")

def main():
    print("=== Multi-Target Exoplanet Transit Analysis ===")
    print(f"[CACHE] Light curve cache directory: {CACHE_DIR}")
    
    # Get targets from API
    print("Fetching exoplanet data from NASA Exoplanet Archive...")
    targets = get_exoplanet_targets(limit=MAX_TARGETS, use_api=True, cache=True)
    
    print(f"Analyzing {len(targets)} exoplanet systems...")
    
    all_results = []
    successful_analyses = 0
    cached_analyses = 0
    start_time = time.time()

    # Analyze each target (minimal printing during analysis)
    print("\nAnalyzing targets...")
    for i, target in enumerate(tqdm(targets)):
        result = analyze_single_target(target, i, len(targets))
        if result:
            all_results.append(result)
            if result['status'] == 'success':
                successful_analyses += 1
                if result.get('from_cache', False):
                    cached_analyses += 1

    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[SAVE] Detailed results saved to {RESULTS_CSV}")

    # Print comprehensive results table AFTER all training is complete
    print_comprehensive_results(results_df)
    
    # Performance statistics
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"Total analysis time: {total_time/60:.2f} minutes")
    print(f"Average time per target: {total_time/len(targets):.2f} seconds")
    print(f"Cached analyses: {cached_analyses}/{successful_analyses} ({cached_analyses/successful_analyses*100:.1f}%)")

    # Compare theoretical vs NN areas for all targets
    print("\n" + "="*100)
    print("THEORETICAL GEOMETRIC vs NN-FITTED AREA COMPARISON")
    print("="*100)
    
    # Safely call comparison function
    try:
        comparisons = analyze_multiple_targets_comparison(results_df)
        
        # Add comparison results to main results if available
        if comparisons:
            comparison_df = pd.DataFrame(comparisons)
            comparison_df.to_csv("theoretical_nn_comparison_results.csv", index=False)
            print(f"[SAVE] Comparison results saved to theoretical_nn_comparison_results.csv")
    except Exception as e:
        print(f"[WARN] Could not generate comparison analysis: {e}")

if __name__ == "__main__":
    main()