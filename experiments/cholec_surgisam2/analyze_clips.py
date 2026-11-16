#!/usr/bin/env python3
"""Analyze existing temporal_per_frame.csv without rerunning model inference.

Per-clip metrics are unweighted averages of available pair-level metrics.
Bootstrap resamples clips, not frames. No claims of aggregate confusion-matrix IoU.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = {
    'temporal_foreground_miou_present': 'temporal_miou_pair_mean',
    'temporal_foreground_mdice_present': 'temporal_mdice_pair_mean',
    'label_change_rate': 'label_change_rate',
    'mean_foreground_abs_area_fraction_change': 'foreground_area_change',
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path('results/surgi_temporal_full'))
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.bootstrap < 100: p.error('--bootstrap must be at least 100')
    src = a.results / 'temporal_per_frame.csv'
    df = pd.read_csv(src)
    required = {'clip', 'frame_index', 'frame', 'method', 'valid_pixels', *METRICS}
    absent = required - set(df.columns)
    if absent: raise ValueError(f'Missing columns: {sorted(absent)}')
    if df.duplicated(['clip', 'frame_index', 'frame', 'method']).any():
        raise ValueError('Duplicate clip/frame/method rows')
    if not {'baseline', 'stitch'}.issubset(set(df.method)):
        raise ValueError('Expected baseline and stitch methods')
    methods = [m for m in ('baseline', 'stitch', 'stitch_fixed') if m in set(df.method)]
    keys = ['clip', 'frame_index', 'frame']
    baseline_keys = set(map(tuple, df.loc[df.method == 'baseline', keys].itertuples(index=False, name=None)))
    for method in methods[1:]:
        current = set(map(tuple, df.loc[df.method == method, keys].itertuples(index=False, name=None)))
        if current != baseline_keys:
            raise ValueError(f'Frame-pair mismatch for {method}: baseline={len(baseline_keys)} {method}={len(current)}')
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    groups = df.groupby(['clip', 'method'], as_index=False)
    per_clip = groups.agg(frame_pairs=('frame_index', 'size'), valid_pixels=('valid_pixels', 'sum'),
                          **{out: (col, 'mean') for col, out in METRICS.items()})
    out = a.results / 'per_clip_temporal.csv'
    per_clip.sort_values(['clip', 'method']).to_csv(out, index=False)

    pivot = per_clip.set_index(['clip', 'method']).unstack('method')
    rows = []
    for method in methods[1:]:
        for clip in sorted(df['clip'].unique()):
            row = {'clip': clip, 'comparison': f'{method}-baseline',
                   'frame_pairs': int(pivot.loc[clip, ('frame_pairs', 'baseline')])}
            for metric in METRICS.values():
                base = pivot.loc[clip, (metric, 'baseline')]
                other = pivot.loc[clip, (metric, method)]
                row[f'{metric}_baseline'] = base
                row[f'{metric}_{method}'] = other
                row[f'delta_{metric}'] = other - base
            rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(a.results / 'paired_comparison.csv', index=False)

    rng = np.random.default_rng(a.seed)
    ci_rows = []
    for comparison, group in paired.groupby('comparison', sort=True):
        for metric in METRICS.values():
            values = group[f'delta_{metric}'].dropna().to_numpy(float)
            n = len(values)
            if n < 2:
                ci_rows.append({'comparison': comparison, 'metric': metric, 'clips': n,
                                'mean_delta': float(np.mean(values)) if n else np.nan,
                                'ci95_low': np.nan, 'ci95_high': np.nan})
                continue
            sample_ids = rng.integers(0, n, size=(a.bootstrap, n))
            estimates = values[sample_ids].mean(axis=1)
            low, high = np.quantile(estimates, [0.025, 0.975])
            ci_rows.append({'comparison': comparison, 'metric': metric, 'clips': n,
                            'mean_delta': values.mean(), 'ci95_low': low, 'ci95_high': high})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(a.results / 'bootstrap_ci.csv', index=False)
    failures = paired.sort_values('delta_temporal_miou_pair_mean').copy()
    failures.to_csv(a.results / 'failure_sequences.csv', index=False)
    print(f'Clips: {df['clip'].nunique()}, temporal pairs/method: {len(baseline_keys)}')
    print('\nPaired clip-level bootstrap (equal weight per clip):')
    print(ci.to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print('\nWorst five clips, stitch vs baseline (pair-mean temporal mIoU):')
    print(failures[failures.comparison == 'stitch-baseline'][['clip', 'frame_pairs', 'delta_temporal_miou_pair_mean']].head(5).to_string(index=False))
    print(f'\nSaved 4 CSVs to {a.results}')

if __name__ == '__main__':
    main()
