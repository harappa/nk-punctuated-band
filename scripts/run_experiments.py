#!/usr/bin/env python3
"""
run_experiments.py — Parallel experiment runner for all simulations (Exp 1–9).

Distributes MC chains across CPU cores via multiprocessing.Pool.
Each job is fully independent (no shared state, thread-safe).

Usage:
    python scripts/run_experiments.py                  # Run all experiments
    python scripts/run_experiments.py --exp 1 2 3      # Run specific experiments
    python scripts/run_experiments.py --workers 32     # Override worker count
    python scripts/run_experiments.py --quick           # Quick validation mode
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from nk_core import MCParams, MCResult, run_single_mc
from experiments import EXPERIMENTS
from analysis import aggregate_results, aggregate_variant_results, result_to_dict


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path('results')
N_WORKERS_DEFAULT = max(1, mp.cpu_count() - 4)


# ============================================================
# Worker functions (top-level for pickling)
# ============================================================

def _worker(params: MCParams):
    """Execute a single MC run (standard Gaussian basin)."""
    result = run_single_mc(params)
    return result_to_dict(result)


def _worker_poly(params):
    """Execute a single MC run with polynomial basin shape."""
    from nk_variants import run_poly
    return run_poly(params)


def _worker_wf(params):
    """Execute a single Wright-Fisher population run."""
    from nk_variants import run_wf
    return run_wf(params)


# ============================================================
# Runner
# ============================================================

def run_experiment(exp_id: str, quick: bool = False,
                   n_workers: int = N_WORKERS_DEFAULT):
    """Run all jobs for one experiment in parallel.

    Args:
        exp_id: Experiment identifier ('1' through '9')
        quick: If True, use short runs for validation
        n_workers: Number of parallel worker processes

    Returns:
        List of per-replicate result dicts
    """
    exp_fn = EXPERIMENTS[exp_id]
    exp = exp_fn(quick=quick)
    jobs = exp['jobs']
    worker_type = exp.get('worker', 'standard')

    if len(jobs) == 0:
        print(f'  Experiment {exp_id} ({exp["name"]}): '
              f'No simulation jobs (analysis-only).')
        return []

    # Select worker function
    if worker_type == 'poly':
        worker_fn = _worker_poly
    elif worker_type == 'wf':
        worker_fn = _worker_wf
    else:
        worker_fn = _worker

    total_steps = sum(j.n_steps for j in jobs)
    print(f'\n{"="*70}')
    print(f'  Experiment {exp_id}: {exp["name"]}')
    print(f'  Jobs: {len(jobs)}, Workers: {n_workers}')
    print(f'  Steps/job: {jobs[0].n_steps:,}, '
          f'Total MC steps: {total_steps:,.0f}')
    print(f'{"="*70}')

    t0 = time.perf_counter()

    with mp.Pool(processes=n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(worker_fn, jobs,
                                                        chunksize=4)):
            results.append(result)
            if (i + 1) % max(1, len(jobs) // 20) == 0 or i + 1 == len(jobs):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(jobs) - i - 1) / rate if rate > 0 else 0
                print(f'    [{i+1:6d}/{len(jobs):6d}] '
                      f'{elapsed:.0f}s elapsed, ETA {eta:.0f}s, '
                      f'{rate:.1f} jobs/s')

    elapsed = time.perf_counter() - t0
    print(f'  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)')

    # Save results
    output_dir = OUTPUT_DIR / f'exp_{exp_id}'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'  Saved {len(results)} results to {results_path}')

    summary = aggregate_variant_results(results, worker_type)
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'  Saved {len(summary)} summary rows to {summary_path}')

    return results


def merge_summaries(exp_ids):
    """Merge per-experiment summaries into a single all_summaries.json."""
    all_data = {}
    for exp_id in exp_ids:
        path = OUTPUT_DIR / f'exp_{exp_id}' / 'summary.json'
        if path.exists():
            with open(path) as f:
                all_data[exp_id] = json.load(f)
            print(f'  Loaded Exp {exp_id}: {len(all_data[exp_id])} rows')

    out_path = OUTPUT_DIR / 'all_summaries.json'
    with open(out_path, 'w') as f:
        json.dump(all_data, f, indent=1)
    print(f'  Merged data saved to {out_path}')
    return all_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run NK fitness landscape Monte Carlo experiments'
    )
    parser.add_argument(
        '--exp', nargs='*', default=None,
        help='Experiment IDs to run (1–9). Default: all.'
    )
    parser.add_argument(
        '--workers', type=int, default=N_WORKERS_DEFAULT,
        help=f'Number of worker processes (default: {N_WORKERS_DEFAULT})'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick validation mode (short runs, 3 replicates)'
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    exp_ids = args.exp if args.exp else sorted(EXPERIMENTS.keys())

    print('NK Fitness Landscape — Monte Carlo Experiments')
    print(f'Workers: {args.workers}, Quick: {args.quick}')
    print(f'Experiments: {", ".join(exp_ids)}')
    print(f'Output directory: {OUTPUT_DIR.absolute()}')

    # JIT warmup
    print('\nJIT compilation warmup...')
    t_warmup = time.perf_counter()
    _ = run_single_mc(MCParams(d=10, K=2, beta=1.0, scoeff=5.0,
                               n_steps=1000, seed=0))
    print(f'  Core engine: OK ({time.perf_counter() - t_warmup:.1f}s)')

    # Warmup variant engines if needed
    if any(e in exp_ids for e in ['8', '9']):
        t_v = time.perf_counter()
        from nk_variants import PolyParams, WFParams, run_poly, run_wf
        if any(e in exp_ids for e in ['8']):
            _ = run_poly(PolyParams(d=10, K=2, beta=1.0, scoeff=5.0,
                                     n_steps=1000, seed=0, h_star=0.35,
                                     p_shape=1.0))
        if any(e in exp_ids for e in ['9']):
            _ = run_wf(WFParams(d=10, K=2, beta=1.0, scoeff=5.0,
                                 n_steps=100, seed=0, h_star=0.35,
                                 N_pop=10, mu_rate=0.1))
        print(f'  Variants: OK ({time.perf_counter() - t_v:.1f}s)')

    t_total = time.perf_counter()
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f'  Unknown experiment: {exp_id}, skipping.')
            continue
        run_experiment(exp_id, quick=args.quick, n_workers=args.workers)

    total = time.perf_counter() - t_total
    print(f'\n{"="*70}')
    print(f'ALL DONE. Total: {total:.0f}s ({total/60:.1f} min)')

    # Merge all summaries
    print('\nMerging summaries...')
    merge_summaries(exp_ids)
    print(f'{"="*70}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
