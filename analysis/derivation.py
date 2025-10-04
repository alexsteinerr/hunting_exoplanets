import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_theoretical_transit_area(R_s, R_p, tau, t0, plot=False):
    k = R_p / R_s
    chord = 2.0 * (1.0 + k)
    vR = chord / tau
    t = np.linspace(t0 - tau/2, t0 + tau/2, 4001)
    x = vR * (t - t0)
    z = np.abs(x)

    def overlap_normalized(zval, k_):
        out = np.zeros_like(zval)
        mask0 = (zval >= 1.0 + k_)
        mask1 = (zval <= 1.0 - k_)
        mask2 = (~mask0) & (~mask1)
        out[mask0] = 0.0
        out[mask1] = k_**2
        if np.any(mask2):
            zz = zval[mask2]
            def clamp(u): return np.clip(u, -1.0, 1.0)
            term1 = np.arccos(clamp((zz**2 + 1 - k_**2) / (2*zz)))
            term2 = np.arccos(clamp((zz**2 + k_**2 - 1) / (2*zz*k_)))
            term3 = 0.5 * np.sqrt(np.clip((-zz + 1 + k_)*(zz + 1 - k_)*(zz - 1 + k_)*(zz + 1 + k_), 0, None))
            A = term1 + k_**2 * term2 - term3
            out[mask2] = A / math.pi
        return out

    lam = overlap_normalized(z, k)
    dt = np.diff(t)
    mid = 0.5 * (lam[:-1] + lam[1:])
    theoretical_area = np.sum(mid * dt)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(t, lam, linewidth=2, label='Theoretical Flux Deficit ΔF(t)')
        plt.fill_between(t, lam, 0, alpha=0.3, label="Area (∫ΔF dt)")
        plt.axvline(t0 - tau/2, linestyle="--", alpha=0.7, label='Transit Boundaries')
        plt.axvline(t0 + tau/2, linestyle="--", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Flux Deficit ΔF(t)")
        plt.title(f"Theoretical Transit Drop: Area = {theoretical_area:.2f} s")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return theoretical_area, t, lam

def compare_theoretical_nn_area(R_s, R_p, tau, t0, nn_area_seconds, target_name="", disposition=""):
    theoretical_area, _, _ = calculate_theoretical_transit_area(R_s, R_p, tau, t0)
    absolute_diff = nn_area_seconds - theoretical_area
    relative_diff = (absolute_diff / theoretical_area) * 100.0 if theoretical_area != 0 else np.inf
    agreement_score = 100.0 - abs(relative_diff)
    if agreement_score >= 50.0:
        if isinstance(disposition, str) and ("false positive" in disposition.lower()):
            final_label = "CONFIRMED FALSE POSITIVE"
        else:
            final_label = "CONFIRMED EXOPLANET"
    else:
        if isinstance(disposition, str) and ("false positive" in disposition.lower()):
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

def _disposition_family(disposition):
    if not isinstance(disposition, str):
        return 'unknown'
    d = disposition.strip().lower()
    if 'false' in d or 'fp' in d:
        return 'fp'
    if 'confirm' in d:
        return 'planet'
    if 'cand' in d or 'toi' in d:
        return 'candidate'
    return 'unknown'

def _label_family(final_label):
    if not isinstance(final_label, str):
        return 'unknown'
    l = final_label.lower()
    if 'false positive' in l:
        return 'fp'
    if 'exoplanet' in l:
        return 'planet'
    return 'unknown'

def analyze_multiple_targets_comparison(results_df):
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
    correct_count = 0
    known_gt_count = 0
    print(f"\n{'='*120}")
    print(f"FINAL CLASSIFICATION BASED ON THEORETICAL-NN AGREEMENT")
    print(f"{'='*120}")
    print(f"{'Target':<20} {'Disposition':<20} {'Agreement':<10} {'Final Label':<28} {'Area Diff %':<12} {'✓?':<3}")
    print(f"{'-'*120}")
    for _, target in successful_with_transit.iterrows():
        try:
            comparison = compare_theoretical_nn_area(
                R_s=target['stellar_radius_m'],
                R_p=target['planet_radius_m'],
                tau=target['width_time_seconds'],
                t0=target['t0_seconds'],
                nn_area_seconds=target['area_time_seconds'],
                target_name=target.get('target_name', ''),
                disposition=target.get('disposition', 'Unknown')
            )
            gt_family = _disposition_family(comparison['disposition'])
            pred_family = _label_family(comparison['final_label'])
            is_correct = None
            if gt_family in ('planet', 'fp') and pred_family in ('planet', 'fp'):
                known_gt_count += 1
                is_correct = (gt_family == pred_family)
                if is_correct:
                    correct_count += 1
            comparison.update({
                'target_name': target.get('target_name', ''),
                'period_days': target.get('period_days', np.nan),
                'gt_family': gt_family,
                'pred_family': pred_family,
                'is_correct': is_correct
            })
            comparisons.append(comparison)
            tick = '✔' if is_correct else ('?' if is_correct is None else '✖')
            print(f"{comparison['target_name']:<20} {comparison['disposition'][:19]:<20} "
                  f"{comparison['agreement_score']:9.1f}% {comparison['final_label']:<28} "
                  f"{comparison['relative_difference_percent']:11.1f}% {tick:<3}")
        except Exception as e:
            print(f"Error analyzing {target.get('target_name', '<unknown>')}: {e}")
            continue
    if comparisons:
        agreements = [c['agreement_score'] for c in comparisons]
        final_labels = [c['final_label'] for c in comparisons]
        print(f"{'-'*120}")
        print(f"CLASSIFICATION SUMMARY ({len(comparisons)} targets):")
        print(f"Confirmed Exoplanets:      {final_labels.count('CONFIRMED EXOPLANET')}/{len(comparisons)}")
        print(f"Confirmed False Positives: {final_labels.count('CONFIRMED FALSE POSITIVE')}/{len(comparisons)}")
        print(f"Potential Exoplanets:      {final_labels.count('POTENTIAL EXOPLANET')}/{len(comparisons)}")
        print(f"Potential False Positives: {final_labels.count('POTENTIAL FALSE POSITIVE')}/{len(comparisons)}")
        print(f"\nAgreement Statistics:")
        print(f"Mean Agreement Score: {np.mean(agreements):.1f}% ± {np.std(agreements):.1f}%")
        print(f"Median Agreement: {np.median(agreements):.1f}%")
        if known_gt_count > 0:
            accuracy = 100.0 * correct_count / known_gt_count
            print(f"\nGround-truth comparable (planet/fp): {correct_count}/{known_gt_count} correct "
                  f"({accuracy:.1f}% accuracy)")
        else:
            print("\nNo targets with scorable ground-truth disposition (CONFIRMED or FALSE POSITIVE).")
    return comparisons