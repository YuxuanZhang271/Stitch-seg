"""Summarize YOLO prompt differences and frame-level performance."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, required=True)
a = p.parse_args()
rows = [json.loads(line) for line in (a.results/'prompt_diagnostics.jsonl').read_text().splitlines() if line.strip()]
by_method = defaultdict(list)
for row in rows:
    by_method[row['method']].append(row)
for method in sorted(by_method, key=lambda x: (x != 'baseline', x)):
    entries = by_method[method]
    if not entries:
        print(method, ': no records')
        continue
    dets = [d for r in entries for d in r['detections']]
    print('\n', method, 'frames:', len(entries), 'detections:', len(dets))
    print('mean detections/frame:', len(dets)/len(entries))
    if dets:
        for field in ('confidence', 'box_area_fraction', 'sam_mask_area_fraction'):
            vals = np.array([d[field] for d in dets])
            print(field, 'mean:', round(float(vals.mean()), 4), 'median:', round(float(np.median(vals)), 4))
        print('fraction rejected by background score:', round(sum(not d['passes_background_score'] for d in dets)/len(dets), 4))
        class_counts = defaultdict(int)
        for d in dets:
            class_counts[d['class_index']] += 1
        print('detections per class:', dict(sorted(class_counts.items())))

with (a.results/'frame_diagnostics.csv').open(newline='') as f:
    frames = list(csv.DictReader(f))
for method in ('stitch', 'stitch_fixed'):
    if not frames or f'{method}_foreground_miou_present' not in frames[0]:
        continue
    valid = []
    for r in frames:
        try:
            delta = float(r[f'{method}_foreground_miou_present']) - float(r['baseline_foreground_miou_present'])
            if np.isfinite(delta): valid.append((r, delta))
        except (TypeError, ValueError):
            pass
    print(f'\nComparable {method} frame-level mIoU rows:', len(valid))
    for label, group in [('worst', sorted(valid, key=lambda item:item[1])[:10]),
                         ('best', sorted(valid, key=lambda item:item[1], reverse=True)[:10])]:
        print(f'\n{label} {method} frames')
        for r, delta in group:
            print(r['frame_index'], Path(r['frame']).name, f'delta={delta:+.4f}')
print('\nNOTE: per-frame present-class mIoU is diagnostic, not the dataset-level mIoU.')
print('NOTE: box coordinates differ across original and stitched crops; direct box IoU is invalid without reprojection.')
