"""UNet++ baseline vs StitchInferencer on consecutive CholecSeg8k PNG clips.

Run from exp_seg/cholecseg8k_benchmark with the original infer_unetpp.py,
model.py and metrics.py present. No Cholec80 videos are required.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import importlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from infer_unetpp import (CFG, CanvasSegModelUnetPP, load_unetpp_model,
                         logits_to_pred_idx, mask_raw_to_index)
from stitch_seg import StitchInferencerDev


class _EmptyMaskModel(nn.Module):
    """No learned tool/port detector: predict zero invalid pixels."""
    def forward(self, x):
        return x.new_zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]))


class StitchInferencerNoLearnedMasks(StitchInferencerDev):
    """Use original stitcher geometry, matching and blending without learned masks.

    The original constructor unconditionally loads two private checkpoints.
    Temporarily substitute zero-output modules at construction, then restore
    the module's loader functions. Endoscope boundary masking remains active.
    """
    def __init__(self, *args, **kwargs):
        mod = importlib.import_module('stitch_seg.inferencer_dev')
        original_tool = mod.load_masking_model
        original_port = mod.load_masking_model2
        mod.load_masking_model = lambda cfg: _EmptyMaskModel().to(cfg.device).eval()
        mod.load_masking_model2 = lambda cfg: _EmptyMaskModel().to(cfg.device).eval()
        try:
            super().__init__(*args, **kwargs)
        finally:
            mod.load_masking_model = original_tool
            mod.load_masking_model2 = original_port

    def _predict_masks(self, frame_u):
        # Preserve original ellipse/border exclusion, but no tool/port mask.
        if self.apply_ellipse_mask and self.ellipse_mask is not None:
            return (torch.nn.functional.interpolate(
                self.ellipse_mask.float(), size=frame_u.shape[-2:],
                mode='nearest') > 0.5).to(torch.uint8) * 255
        return torch.zeros((1, 1, *frame_u.shape[-2:]),
                           dtype=torch.uint8, device=frame_u.device)

EXCLUDE = {0, 10, 11}
FRAME_RE = re.compile(r'^frame_(\d+)_endo\.png$')


def resolve_mask_path(value):
    p = Path(str(value))
    if p.is_absolute() and p.exists():
        return p
    for base in (HERE, ROOT, Path.cwd()):
        q = (base / p).resolve()
        if q.exists():
            return q
    raise FileNotFoundError(f'Cannot resolve CSV mask path: {value}')


def clip_index(mask_path):
    folder = mask_path.parent
    images = []
    for p in folder.iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            images.append((int(m.group(1)), p))
    images.sort(key=lambda x: x[0])
    return images


def confusion_update(cm, pred, gt, ignore_tool_gt=True):
    valid = (gt >= 0) & (gt < cm.shape[0]) & (pred >= 0) & (pred < cm.shape[0])
    if ignore_tool_gt:
        valid &= (gt != 10) & (gt != 11)
    inds = gt[valid].astype(np.int64) * cm.shape[0] + pred[valid].astype(np.int64)
    cm += np.bincount(inds, minlength=cm.size).reshape(cm.shape)


def scores(cm, included=None):
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    union = gt + pred - tp
    dice_denom = gt + pred
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    dice = np.divide(2 * tp, dice_denom, out=np.full_like(tp, np.nan), where=dice_denom > 0)
    if included is None:
        included = [c for c in range(len(gt)) if c not in EXCLUDE and gt[c] > 0]
    return {'miou': float(np.nanmean(iou[included])) if included else float('nan'),
            'mdice': float(np.nanmean(dice[included])) if included else float('nan'),
            'pixel_accuracy': float(tp.sum() / cm.sum()) if cm.sum() else float('nan'),
            'iou': iou, 'dice': dice, 'gt_pixels': gt, 'included': included}


def create_config(args, device):
    cfg = CFG()
    cfg.device = device
    cfg.bbox_mode = args.bbox_mode
    cfg.use_static_roi = args.use_static_roi
    cfg.apply_ellipse_mask = args.apply_ellipse_mask
    cfg.equalize_hist_rgb = False
    cfg.output_mode = 'logits'
    # Prevent forced class predictions in evaluation. Masks still protect the canvas.
    cfg.tool_class_ch = None
    cfg.enable_depth_mask = False
    for name in ('aliked_weights', 'lightglue_weights', 'tool_detector_weights', 'port_detector_weights'):
        value = Path(getattr(cfg, name))
        setattr(cfg, name, str((HERE / value).resolve()) if not value.is_absolute() else str(value))
    return cfg


def check_stitch_weights(cfg):
    needed = ('aliked_weights', 'lightglue_weights')
    missing = [f'{name}: {getattr(cfg, name)}' for name in needed if not Path(getattr(cfg, name)).is_file()]
    if missing:
        raise FileNotFoundError('StitchInferencer requires these missing checkpoints:\n' + '\n'.join(missing) +
                                '\nSupply the ALIKED and LightGlue weights before running stitched inference.')


def save_preview(out_dir, key, frame, gt, base, stitch):
    palette = np.random.default_rng(42).integers(0, 256, (13, 3), dtype=np.uint8)
    palette[0] = 0
    def overlay(labels):
        return cv2.addWeighted(frame, 0.55, palette[labels], 0.45, 0)
    panel = np.concatenate([frame, overlay(gt), overlay(base), overlay(stitch)], axis=1)
    folder = out_dir / 'examples'
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f'{key}.jpg'), panel)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fold', type=int, default=4)
    ap.add_argument('--csv-path', default=None)
    ap.add_argument('--weights-path', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--bbox-mode', choices=['internal', 'external', 'entire'], default='internal')
    ap.add_argument('--limit', type=int, default=0, help='Number of annotated target frames; 0 = all')
    ap.add_argument('--max-clips', type=int, default=0, help='Limit clips; 0 = all')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--preview-every', type=int, default=20)
    ap.add_argument('--use-static-roi', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--apply-ellipse-mask', action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else HERE / 'splits' / f'cholecseg8k_test_{args.fold}.csv'
    weights = Path(args.weights_path) if args.weights_path else HERE / 'models' / 'unetpp' / f'fold{args.fold}.pth'
    out = Path(args.out_dir) if args.out_dir else HERE / 'infer_outputs' / f'images_unetpp_{args.bbox_mode}_fold{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if args.limit > 0:
        df = df.head(args.limit)
    grouped = defaultdict(list)
    for path in df['file']:
        mask = resolve_mask_path(path)
        grouped[str(mask.parent)].append(mask)
    groups = sorted(grouped.items())
    if args.max_clips > 0:
        groups = groups[:args.max_clips]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = create_config(args, device)
    check_stitch_weights(cfg)
    model = load_unetpp_model(str(weights), device)
    # Same wrapper and preprocessing for baseline and Stitch.
    seg = CanvasSegModelUnetPP(model, device, input_size=CFG.image_size, dynamic_shape=False).to(device).eval()
    cm_base = np.zeros((13, 13), dtype=np.int64)
    cm_stitch = np.zeros((13, 13), dtype=np.int64)
    cm_peri_base = np.zeros((13, 13), dtype=np.int64)
    cm_peri_stitch = np.zeros((13, 13), dtype=np.int64)
    n_eval = 0
    n_processed = 0
    report_rows = []
    print(f'[INFO] device={device}, clips={len(groups)}, annotated_targets={len(df)}, bbox={args.bbox_mode}', flush=True)
    print('[INFO] Clip-based image evaluation: stitching resets for every clip.', flush=True)
    print('[INFO] Learned tool/port masks DISABLED; ellipse/border mask retained.', flush=True)
    with torch.inference_mode():
        for clip_num, (clip, masks) in enumerate(groups, 1):
            targets = {int(re.search(r'frame_(\d+)_endo_watershed_mask', p.name).group(1)): p for p in masks}
            images = clip_index(masks[0])
            if not images:
                raise RuntimeError(f'No input images in {clip}')
            first_target = min(targets)
            last_target = max(targets)
            images = [(fid, p) for fid, p in images if fid <= last_target]
            # Include all available earlier frames within this clip for stitching history.
            inferencer = StitchInferencerNoLearnedMasks(seg, cfg=cfg)
            inferencer.eval()
            clip_count = 0
            last_fid = None
            for fid, image_path in images:
                if last_fid is not None and fid != last_fid + 1:
                    # Avoid incorrectly stitching across missing frames.
                    inferencer.reset_state(clear_preprocess_state=True)
                last_fid = fid
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise FileNotFoundError(image_path)
                frame_u = inferencer.preprocess_frame(frame)
                inferencer.step_canvas(frame_u)
                n_processed += 1
                if fid not in targets:
                    continue
                stitched_logits = inferencer.model_inference(tuple(frame_u.shape[-2:]))
                if stitched_logits is None:
                    raise RuntimeError(f'No stitched output at {image_path}')
                direct_logits = seg(frame_u)
                stitched = logits_to_pred_idx(stitched_logits).cpu().numpy().astype(np.uint8)
                baseline = logits_to_pred_idx(direct_logits).cpu().numpy().astype(np.uint8)
                gt_raw = cv2.imread(str(targets[fid]), cv2.IMREAD_GRAYSCALE)
                if gt_raw is None:
                    raise FileNotFoundError(targets[fid])
                if inferencer.use_roi and inferencer.roi is not None:
                    x, y, w, h = inferencer.roi
                    gt_raw = gt_raw[y:y+h, x:x+w]
                    frame_view = frame[y:y+h, x:x+w]
                else:
                    frame_view = frame
                gt = mask_raw_to_index(gt_raw)
                if gt.shape != baseline.shape or gt.shape != stitched.shape:
                    raise ValueError(f'Shape mismatch at {image_path}: gt={gt.shape} baseline={baseline.shape} stitch={stitched.shape}')
                confusion_update(cm_base, baseline, gt)
                confusion_update(cm_stitch, stitched, gt)
                h, w = gt.shape
                yy, xx = np.ogrid[:h, :w]
                peripheral = ((xx-(w-1)/2)/max(w/2, 1))**2 + ((yy-(h-1)/2)/max(h/2, 1))**2 > (2/3)**2
                # Mask out central pixels by passing only peripheral pixels to confusion accumulation.
                confusion_update(cm_peri_base, baseline[peripheral], gt[peripheral])
                confusion_update(cm_peri_stitch, stitched[peripheral], gt[peripheral])
                if args.debug and (n_eval % max(args.preview_every, 1) == 0):
                    key = f'{Path(clip).parent.name}_{Path(clip).name}_{fid:06d}'
                    save_preview(out, key, frame_view, gt, baseline, stitched)
                report_rows.append({'clip': str(Path(clip).name), 'frame_id': fid})
                n_eval += 1
                clip_count += 1
            print(f'[{clip_num}/{len(groups)}] {Path(clip).name}: evaluated={clip_count}, processed={len(images)}', flush=True)
            del inferencer
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Common class set based on GT support in either prediction's confusion matrix.
    included = [c for c in range(13) if c not in EXCLUDE and cm_base[c].sum() > 0]
    b = scores(cm_base, included)
    s = scores(cm_stitch, included)
    pb = scores(cm_peri_base, included)
    ps = scores(cm_peri_stitch, included)
    np.save(out / 'cm_baseline.npy', cm_base)
    np.save(out / 'cm_stitch.npy', cm_stitch)
    np.save(out / 'cm_peripheral_baseline.npy', cm_peri_base)
    np.save(out / 'cm_peripheral_stitch.npy', cm_peri_stitch)
    with (out / 'per_class.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'gt_pixels', 'baseline_dice', 'stitch_dice', 'baseline_iou', 'stitch_iou', 'included'])
        for c in range(13):
            writer.writerow([c, int(b['gt_pixels'][c]), b['dice'][c], s['dice'][c], b['iou'][c], s['iou'][c], c in included])
    metrics = {'fold': args.fold, 'bbox_mode': args.bbox_mode, 'evaluated_frames': n_eval,
               'processed_frames': n_processed, 'clips': len(groups), 'included_classes': included,
               'baseline_mdice': b['mdice'], 'stitch_mdice': s['mdice'],
               'baseline_miou': b['miou'], 'stitch_miou': s['miou'],
               'baseline_pixel_accuracy': b['pixel_accuracy'], 'stitch_pixel_accuracy': s['pixel_accuracy'],
               'baseline_peripheral_mdice': pb['mdice'], 'stitch_peripheral_mdice': ps['mdice'],
               'note': 'Short-clip PNG evaluation WITHOUT learned tool/port masking; not full Cholec80 continuous-video reproduction.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(out / 'metrics.csv', index=False)
    print('\n=== FINAL RESULTS ===')
    for k, v in metrics.items():
        print(f'{k}: {v}')
    print(f'[SAVED] {out.resolve()}')


if __name__ == '__main__':
    main()
