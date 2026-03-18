"""
analysis.py — Aggregation, regime classification, and result serialization.

Provides:
    - aggregate_results(): Per-replicate → per-condition summary statistics
    - classify_regime(): Regime classification at configurable thresholds
    - result_to_dict(): Serialize MCResult for JSON output
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any


# ============================================================
# Regime classification
# ============================================================

def classify_regime(mean_ratio: float, mean_n_tr: float,
                    ratio_threshold: float = 10,
                    total_n_tr: int = None) -> str:
    """Classify dynamical regime from summary statistics.

    Regimes:
        FROZEN:     total_n_tr == 0 across all replicates (if available),
                    otherwise mean_n_tr < 1
        PUNCTUATED: ratio ≥ threshold
        MARGINAL:   ratio ≥ threshold/2 (borderline)
        DIFFUSIVE:  otherwise

    The FROZEN criterion follows the paper's definition: β*_high(d) is
    the lowest β at which ALL replicates yield n_tr = 0 (total_n_tr = 0).
    When total_n_tr is provided, it is used for FROZEN classification;
    otherwise falls back to mean_n_tr < 1.

    Args:
        mean_ratio: Mean stasis/transit ratio across replicates
        mean_n_tr: Mean number of complete transitions
        ratio_threshold: Ratio cutoff for punctuated (default: 10)
        total_n_tr: Total transitions across all replicates (optional)

    Returns:
        One of 'FROZEN', 'PUNCTUATED', 'MARGINAL', 'DIFFUSIVE'
    """
    if total_n_tr is not None:
        is_frozen = (total_n_tr == 0)
    else:
        is_frozen = (mean_n_tr < 1)

    if is_frozen:
        return 'FROZEN'
    elif mean_ratio >= ratio_threshold:
        return 'PUNCTUATED'
    elif mean_ratio >= ratio_threshold * 0.5:
        return 'MARGINAL'
    else:
        return 'DIFFUSIVE'


# ============================================================
# Result serialization
# ============================================================

def result_to_dict(result) -> Dict[str, Any]:
    """Convert MCResult to a JSON-serializable dict.

    Large arrays (stasis/transit durations) are capped at 10,000 entries.
    """
    d = {
        'd': result.d, 'K': result.K, 'beta': result.beta,
        'scoeff': result.scoeff, 'n_steps': result.n_steps,
        'seed': result.seed, 'h_star': result.h_star,
        'n_transitions': result.n_transitions,
        'mean_stasis': result.mean_stasis,
        'mean_transit': result.mean_transit,
        'ratio': result.ratio,
        'wall_time': result.wall_time,
        'dw_mean': result.dw_mean,
        'dw_std': result.dw_std,
        'dw_abs_mean': result.dw_abs_mean,
        'dw_abs_median': result.dw_abs_median,
        'dw_n_samples': result.dw_n_samples,
    }
    if result.stasis_durations is not None and len(result.stasis_durations) > 0:
        n = min(len(result.stasis_durations), 10000)
        d['stasis_durations'] = result.stasis_durations[:n].tolist()
        d['transit_durations'] = result.transit_durations[:n].tolist()
    return d


def aggregate_variant_results(results: List[Dict], worker_type: str = 'standard') -> List[Dict]:
    """Dispatch to appropriate aggregation function based on worker type."""
    if worker_type == 'poly':
        return _aggregate_poly(results)
    elif worker_type == 'wf':
        return _aggregate_wf(results)
    else:
        return aggregate_results(results)


def _aggregate_poly(results: List[Dict]) -> List[Dict]:
    """Aggregate Exp 8 results by (d, beta, p_shape)."""
    groups = defaultdict(list)
    for r in results:
        key = (r['d'], r['beta'], r['p_shape'])
        groups[key].append(r)

    summary = []
    rng_boot = np.random.RandomState(12345)
    for key, reps in sorted(groups.items()):
        d, beta, p_shape = key
        n_trs = [r['n_transitions'] for r in reps]
        ratios = [r['ratio'] for r in reps if r['ratio'] < 1e9]
        row = {
            'd': d, 'K': reps[0].get('K', 2), 'beta': beta, 'p_shape': p_shape,
            'scoeff': reps[0].get('scoeff', 5.0), 'h_star': reps[0].get('h_star', 0.35),
            'n_reps': len(reps),
            'mean_n_tr': float(np.mean(n_trs)),
            'std_n_tr': float(np.std(n_trs)),
            'total_n_tr': int(np.sum(n_trs)),
            'mean_ratio': float(np.mean(ratios)) if ratios else 1e10,
            'std_ratio': float(np.std(ratios)) if len(ratios) > 1 else 0.0,
            'mean_stasis': float(np.mean([r['mean_stasis'] for r in reps])),
            'mean_transit': float(np.mean([r['mean_transit'] for r in reps
                                           if r['mean_transit'] > 0])) \
                            if any(r['mean_transit'] > 0 for r in reps) else 0.0,
            'mean_wall_time': float(np.mean([r.get('wall_time', 0) for r in reps])),
        }
        for thresh in [5, 10, 20]:
            row[f'regime_r{thresh}'] = classify_regime(
                row['mean_ratio'], row['mean_n_tr'], thresh,
                total_n_tr=row['total_n_tr'])
        summary.append(row)
    return summary


def _aggregate_wf(results: List[Dict]) -> List[Dict]:
    """Aggregate Exp 9 results by (d, beta, N_pop, mu_rate)."""
    groups = defaultdict(list)
    for r in results:
        key = (r['d'], r['beta'], r['N_pop'], r['mu_rate'])
        groups[key].append(r)

    summary = []
    for key, reps in sorted(groups.items()):
        d, beta, N_pop, mu_rate = key
        n_trs = [r['n_transitions'] for r in reps]
        ratios = [r['ratio'] for r in reps if r['ratio'] < 1e9]
        row = {
            'd': d, 'K': reps[0].get('K', 2), 'beta': beta,
            'scoeff': reps[0].get('scoeff', 5.0), 'h_star': reps[0].get('h_star', 0.35),
            'N_pop': N_pop, 'mu_rate': mu_rate, 'Nmu': N_pop * mu_rate,
            'n_reps': len(reps),
            'mean_n_tr': float(np.mean(n_trs)),
            'std_n_tr': float(np.std(n_trs)),
            'total_n_tr': int(np.sum(n_trs)),
            'mean_ratio': float(np.mean(ratios)) if ratios else 1e10,
            'std_ratio': float(np.std(ratios)) if len(ratios) > 1 else 0.0,
            'mean_stasis': float(np.mean([r['mean_stasis'] for r in reps])),
            'mean_transit': float(np.mean([r['mean_transit'] for r in reps
                                           if r['mean_transit'] > 0])) \
                            if any(r['mean_transit'] > 0 for r in reps) else 0.0,
            'mean_wall_time': float(np.mean([r.get('wall_time', 0) for r in reps])),
        }
        for thresh in [5, 10, 20]:
            row[f'regime_r{thresh}'] = classify_regime(
                row['mean_ratio'], row['mean_n_tr'], thresh,
                total_n_tr=row['total_n_tr'])
        summary.append(row)
    return summary


# ============================================================
# Aggregation (standard experiments 1–7)
# ============================================================

def aggregate_results(results: List[Dict]) -> List[Dict]:
    """Aggregate per-replicate results into per-condition summaries.

    Groups results by (d, K, β, scoeff, h*) and computes:
        - Mean, std, CI for transition counts
        - Mean stasis/transit ratio
        - Regime classification at thresholds 5, 10, 20
        - Bootstrap 95% CI for mean n_transitions

    Args:
        results: List of per-replicate dicts (from result_to_dict)

    Returns:
        List of summary dicts, one per unique (d, K, β, scoeff, h*) combination
    """
    groups = defaultdict(list)
    for r in results:
        key = (r['d'], r['K'], r['beta'], r['scoeff'], r['h_star'])
        groups[key].append(r)

    summary = []
    for key, reps in sorted(groups.items()):
        d, K, beta, scoeff, h_star = key
        n_trs = [r['n_transitions'] for r in reps]
        ratios = [r['ratio'] for r in reps if r['ratio'] < 1e9]

        row = {
            'd': d, 'K': K, 'beta': beta, 'scoeff': scoeff, 'h_star': h_star,
            'n_reps': len(reps),
            # Transition count statistics
            'mean_n_tr': float(np.mean(n_trs)),
            'std_n_tr': float(np.std(n_trs)),
            'median_n_tr': float(np.median(n_trs)),
            'min_n_tr': int(np.min(n_trs)),
            'max_n_tr': int(np.max(n_trs)),
            'total_n_tr': int(np.sum(n_trs)),
            # Ratio statistics
            'mean_ratio': float(np.mean(ratios)) if ratios else 1e10,
            'std_ratio': float(np.std(ratios)) if len(ratios) > 1 else 0.0,
            # Duration statistics
            'mean_stasis': float(np.mean([r['mean_stasis'] for r in reps])),
            'mean_transit': float(np.mean(
                [r['mean_transit'] for r in reps if r['mean_transit'] > 0]
            )) if any(r['mean_transit'] > 0 for r in reps) else 0.0,
            # Timing
            'mean_wall_time': float(np.mean([r['wall_time'] for r in reps])),
        }

        # Δw statistics (if available)
        dw_means = [r['dw_abs_mean'] for r in reps if r.get('dw_abs_mean', 0) > 0]
        if dw_means:
            row['dw_abs_mean'] = float(np.mean(dw_means))
            row['dw_std'] = float(np.mean(
                [r['dw_std'] for r in reps if r.get('dw_std', 0) > 0]
            ))

        # Bootstrap 95% CI for mean n_transitions (fixed seed for reproducibility)
        if len(n_trs) >= 5:
            rng_boot = np.random.RandomState(12345)
            boot = [float(np.mean(rng_boot.choice(n_trs, size=len(n_trs),
                                                   replace=True)))
                    for _ in range(2000)]
            row['n_tr_ci_low'] = float(np.percentile(boot, 2.5))
            row['n_tr_ci_high'] = float(np.percentile(boot, 97.5))

        # Regime classification at multiple thresholds
        for thresh in [5, 10, 20]:
            row[f'regime_r{thresh}'] = classify_regime(
                row['mean_ratio'], row['mean_n_tr'], thresh,
                total_n_tr=row['total_n_tr']
            )

        summary.append(row)

    return summary
