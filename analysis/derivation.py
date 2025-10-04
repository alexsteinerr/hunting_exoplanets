import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_theoretical_transit_area(R_s, R_p, tau, t0, plot=False):
    k = R_p / R_s

    chord = 2.0 * math.sqrt(max(0.0, (1.0 + k)**2))
    vR = chord / tau   # speed across disk [R_s / s]

    # time grid
    t = np.linspace(t0 - tau, t0 + tau, 4001)
    x = vR * (t - t0)
    z = np.sqrt(x**2)

    def overlap_normalized(zval, k):
        out = np.zeros_like(zval)
        mask0 = (zval >= 1.0 + k)   # no overlap
        mask1 = (zval <= 1.0 - k)   # full overlap
        mask2 = (~mask0) & (~mask1)

        out[mask0] = 0.0
        out[mask1] = k**2

        zz = zval[mask2]
        def clamp(u): return np.clip(u, -1.0, 1.0)
        term1 = np.arccos(clamp((zz**2 + 1 - k**2) / (2*zz)))
        term2 = np.arccos(clamp((zz**2 + k**2 - 1) / (2*zz*k)))
        term3 = 0.5 * np.sqrt(np.clip((-zz + 1 + k)*(zz + 1 - k)*(zz - 1 + k)*(zz + 1 + k), 0, None))
        A = term1 + k**2 * term2 - term3
        out[mask2] = A / math.pi
        return out

    lam = overlap_normalized(z, k)

    dt = np.diff(t)
    mid = (lam[:-1] + lam[1:]) / 2
    theoretical_area = np.sum(mid * dt)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(t, lam, 'b-', linewidth=2, label='Theoretical Flux Deficit')
        plt.fill_between(t, lam, 0, alpha=0.3, label="Theoretical Area")
        plt.axvline(t0 - tau/2, linestyle="--", color='red', alpha=0.7, label='Transit Boundaries')
        plt.axvline(t0 + tau/2, linestyle="--", color='red', alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Flux Deficit ΔF(t)")
        plt.title(f"Theoretical Transit Drop: Area = {theoretical_area:.2f} seconds")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return theoretical_area, t, lam

def compare_theoretical_nn_area(R_s, R_p, tau, t0, nn_area_seconds, target_name="", disposition=""):
    """
    Compare theoretical area with NN-fitted area and calculate agreement
    Returns disposition-based assessment
    """
    # Calculate theoretical area
    theoretical_area, t, lam = calculate_theoretical_transit_area(R_s, R_p, tau, t0)
    
    # Calculate differences and ratios
    absolute_diff = nn_area_seconds - theoretical_area
    relative_diff = (absolute_diff / theoretical_area) * 100  #
    agreement_score = 100 - abs(relative_diff)
    
    if agreement_score >= 80:  
        if "False Positive" in disposition:
            final_label = "CONFIRMED FALSE POSITIVE"
        else:
            final_label = "CONFIRMED EXOPLANET"
    else:
        if "False Positive" in disposition:
            final_label = "POTENTIAL EXOPLANET"  
        else:
            final_label = "POTENTIAL FALSE POSITIVE"  
    
    return {
        'theoretical_area_seconds': theoretical_area,
        'nn_area_seconds': nn_area_seconds,
        'absolute_difference': absolute_diff,
        'relative_difference_percent': relative_diff,
        'agreement_score': agreement_score,
        'final_label': final_label,
        'disposition': disposition
    }

def analyze_multiple_targets_comparison(results_df):
    """
    Analyze comparison between theoretical and NN areas for multiple targets
    Focus on final classification labels
    """
    successful_with_transit = results_df[
        (results_df['status'] == 'success') & 
        (results_df['transit_detected'] == True) &
        (results_df['stellar_radius_m'].notna()) &
        (results_df['planet_radius_m'].notna())
    ].copy()
    
    if len(successful_with_transit) == 0:
        print("No targets with complete data for theoretical comparison")
        return None
    
    comparisons = []
    
    print(f"\n{'='*120}")
    print(f"FINAL CLASSIFICATION BASED ON THEORETICAL-NN AGREEMENT")
    print(f"{'='*120}")
    print(f"{'Target':<20} {'Disposition':<20} {'Agreement':<10} {'Final Label':<25} {'Area Diff %':<12}")
    print(f"{'-'*120}")
    
    for _, target in successful_with_transit.iterrows():
        try:
            comparison = compare_theoretical_nn_area(
                R_s=target['stellar_radius_m'],
                R_p=target['planet_radius_m'],
                tau=target['width_time_seconds'],
                t0=target['t0_seconds'],
                nn_area_seconds=target['area_time_seconds'],
                target_name=target['target_name'],
                disposition=target.get('disposition', 'Unknown')
            )
            
            # Add target info to comparison
            comparison['target_name'] = target['target_name']
            comparison['period_days'] = target['period_days']
            comparisons.append(comparison)
            
            # Print classification line
            print(f"{target['target_name']:<20} {comparison['disposition'][:19]:<20} "
                  f"{comparison['agreement_score']:9.1f}% {comparison['final_label']:<25} "
                  f"{comparison['relative_difference_percent']:11.1f}%")
                  
        except Exception as e:
            print(f"Error analyzing {target['target_name']}: {e}")
            continue
    
    if comparisons:
        # Calculate overall statistics
        agreements = [c['agreement_score'] for c in comparisons]
        final_labels = [c['final_label'] for c in comparisons]
        dispositions = [c['disposition'] for c in comparisons]
        
        print(f"{'-'*120}")
        print(f"CLASSIFICATION SUMMARY ({len(comparisons)} targets):")
        
        # Count final labels
        confirmed_exoplanets = final_labels.count("CONFIRMED EXOPLANET")
        confirmed_fp = final_labels.count("CONFIRMED FALSE POSITIVE")
        potential_exoplanets = final_labels.count("POTENTIAL EXOPLANET")
        potential_fp = final_labels.count("POTENTIAL FALSE POSITIVE")
        
        print(f"Confirmed Exoplanets:     {confirmed_exoplanets}/{len(comparisons)}")
        print(f"Confirmed False Positives: {confirmed_fp}/{len(comparisons)}")
        print(f"Potential Exoplanets:     {potential_exoplanets}/{len(comparisons)}")
        print(f"Potential False Positives: {potential_fp}/{len(comparisons)}")
        
        # Agreement statistics
        print(f"\nAgreement Statistics:")
        print(f"Mean Agreement Score: {np.mean(agreements):.1f}% ± {np.std(agreements):.1f}%")
        print(f"Median Agreement: {np.median(agreements):.1f}%")
    
    return comparisons