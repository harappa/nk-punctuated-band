#!/usr/bin/env python3
"""
self_test.py — Validation and benchmark suite.

Runs three test phases:
    1. Compilation: Verify Numba JIT compiles successfully
    2. Regime verification: Check known (d, β) points produce expected regimes
    3. Benchmark: Measure MC steps/s across dimensions

Usage:
    python scripts/self_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from nk_core import MCParams, run_single_mc


def test_compilation():
    """Phase 1: JIT compilation smoke test."""
    print('=== Phase 1: Compilation ===')
    t0 = time.perf_counter()
    p = MCParams(d=10, K=2, beta=0.5, scoeff=5.0, n_steps=10000, seed=42,
                 record_dw=True, record_trajectory=True, traj_interval=100)
    r = run_single_mc(p)
    elapsed = time.perf_counter() - t0
    print(f'  Compile + run: {elapsed:.2f}s')
    print(f'  n_transitions={r.n_transitions}, ratio={r.ratio:.1f}')
    if r.trajectory_gbar is not None:
        print(f'  gbar range: [{r.trajectory_gbar.min():.3f}, '
              f'{r.trajectory_gbar.max():.3f}]')
    assert r.n_transitions >= 0, 'Negative transitions'
    print('  PASSED\n')


def test_regimes():
    """Phase 2: Verify known regimes at reference (d, β) points."""
    print('=== Phase 2: Regime Verification ===')
    cases = [
        # (d, beta, expected_regime)
        (10, 0.5, 'DIFFUSIVE'),
        (10, 8.0, 'PUNCTUATED'),   # β=4 is borderline (ratio≈10.6); β=8 is clear
        (10, 32.0, 'FROZEN'),
        (40, 2.0, 'DIFFUSIVE'),
        (40, 64.0, 'FROZEN'),
    ]
    all_pass = True
    for d, beta, expected in cases:
        p = MCParams(d=d, K=2, beta=beta, scoeff=5.0,
                     n_steps=500_000, seed=42)
        r = run_single_mc(p)
        if r.n_transitions == 0:
            regime = 'FROZEN'
        elif r.ratio > 10:
            regime = 'PUNCTUATED'
        else:
            regime = 'DIFFUSIVE'

        ok = regime == expected
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            all_pass = False
        print(f'  d={d:3d} β={beta:5.1f}: n_tr={r.n_transitions:6d}, '
              f'ratio={r.ratio:8.1f} → {regime:10s} '
              f'(expected {expected:10s}) [{status}]')

    if all_pass:
        print('  ALL PASSED\n')
    else:
        print('  SOME FAILED — check results above\n')
    return all_pass


def test_benchmark():
    """Phase 3: Performance benchmark across dimensions."""
    print('=== Phase 3: Benchmark ===')
    for d in [10, 40, 80, 150, 300]:
        p = MCParams(d=d, K=2, beta=4.0, scoeff=5.0,
                     n_steps=1_000_000, seed=42)
        t0 = time.perf_counter()
        r = run_single_mc(p)
        elapsed = time.perf_counter() - t0
        rate = p.n_steps / elapsed
        print(f'  d={d:3d}: {elapsed:.3f}s '
              f'({rate/1e6:.1f}M steps/s), '
              f'n_tr={r.n_transitions}')
    print()


def main():
    print('NK Fitness Landscape — Self-Test Suite\n')
    test_compilation()
    passed = test_regimes()
    test_benchmark()
    if passed:
        print('All tests passed.')
    else:
        print('WARNING: Some regime tests failed.')
        sys.exit(1)


if __name__ == '__main__':
    main()
