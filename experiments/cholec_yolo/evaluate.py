"""Compare YOLO semantic baseline with no-aux-mask StitchInferencer.

Run from Stitch-seg repository root. Ground truth is read from original masks,
not from rasterized YOLO polygons. No test-frame skipping is permitted.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from yolo_adapter import DenseYOLO
from stitch_no_masks import StitchNoAuxMasks
from stitch_seg.config import build_default_cfg


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=Path('data/raw/CholecSeg8k'))
    p.add_argument('--split-dir', type=Path, default=Path('data/yolo_cholec/images/val'))
    p.add_argument('--dataset-yaml', type=Path, default=Path('data/yolo_cholec/dataset.yaml'))
    p.add_argument('--weights', type=Path, default=Path('weights/cholec_yolo_seg.pt'))
    p.add_argument('--out', type=Path, default=Path('results/yolo_stitch'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--conf', type=float, default=0.10)
    p.add_argument('--background-score', type=float, default=0.5)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--save-every', type=int, default=0, help='Save example label PNGs every N frames')
    return p.parse_args()


def raw_code_mapping(dataset_yaml, model):
    config = yaml.safe_load(dataset_yaml.read_text())
    names = config['names']
    names = [names[i] if i in names else names[str(i)] for i in range(model.n_foreground)] if isinstance(names, dict) else names
    if len(names) != model.n_foreground:
        raise ValueError('Dataset class count differs from checkpoint')
    for i, name in enumerate(names):
        if str(name) != str(model.yolo.names[i]):
            raise ValueError(f'Checkpoint/YAML class {i} mismatch: {model.yolo.names[i]} vs {name}')
    codes = []
    for name in names:
        match = re.fullmatch(r'raw_(\d+)', str(name))
        if not match:
            raise ValueError('Expected raw_N class names; inspect raw_class_mapping.yaml and adapt mapping before evaluation')
        codes.append(int(match.group(1)))
    if len(set(codes)) != len(codes):
        raise ValueError('Duplicate raw class values')
    return codes, names


def collect_frames(raw, split_dir):
    # Converted names from prepare.py: videoXX_videoXX_NNNNN_frame_NNN_endo.png
    # Resolve each original image by its video, clip and frame suffix.
    originals = sorted(raw.rglob('*_endo.png'))
    by_key = {}
    for image in originals:
        video = next((p for p in image.parts if re.fullmatch(r'video\d+', p)), None)
        if video is None:
            continue
        key = (video, image.parent.name, image.name)
        if key in by_key:
            raise ValueError(f'Duplicate raw image key: {key}')
        by_key[key] = image
    selected = []
    for converted in sorted(split_dir.glob('*.png')):
        match = re.fullmatch(r'(video\d+)_(video\d+_\d+)_(.+_endo)\.png', converted.name)
        if match is None:
            raise ValueError(f'Unexpected converted filename: {converted.name}')
        key = (match.group(1), match.group(2), match.group(3) + '.png')
        if key not in by_key:
            raise FileNotFoundError(f'Cannot match validation image {converted} to raw key {key}')
        selected.append(by_key[key])
    if not selected:
        raise ValueError(f'No validation images in {split_dir}')
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate validation frames')
    return sorted(selected, key=lambda p: (p.parent.as_posix(), p.name))


def find_gt(image):
    # Require the same convention used in the original prepare.py.
    mask = image.with_name(image.stem + '_watershed_mask.png')
    if not mask.exists():
        raise FileNotFoundError(f'Expected ground truth mask: {mask}')
    return mask


def read_gt(mask_path, codes):
    mask = cv2.imread(
        str(mask_path), cv2.IMREAD_UNCHANGED
    )
    if mask is None:
        raise FileNotFoundError(mask_path)

    if mask.ndim == 3:
        if not (
            np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 1], mask[..., 2])
        ):
            raise ValueError(
                f"Unexpected multichannel mask: {mask_path}"
            )
        mask = mask[..., 0]

    allowed = set(codes) | {0, 255}
    unknown = set(np.unique(mask).tolist()) - allowed

    if unknown:
        raise ValueError(
            f"Unknown GT values: {sorted(unknown)}"
        )

    gt = np.full(mask.shape, 255, dtype=np.uint8)
    gt[mask == 0] = 0

    for cls, code in enumerate(codes, start=1):
        gt[mask == code] = cls

    return gt


def update_cm(cm, gt, pred):
    if gt.shape != pred.shape:
        raise ValueError("GT/prediction shape mismatch")

    n = cm.shape[0]
    valid = gt != 255

    if not np.all((pred[valid] >= 0) & (pred[valid] < n)):
        raise ValueError("Invalid prediction class")

    indices = (
        gt[valid].astype(np.int64) * n
        + pred[valid].astype(np.int64)
    )

    cm += np.bincount(
        indices,
        minlength=n * n
    ).reshape(n, n)


def scores(cm):
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    den_dice = cm.sum(0) + cm.sum(1)
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
    dice = np.divide(2*tp, den_dice, out=np.full(len(tp), np.nan), where=den_dice > 0)
    return iou, dice, float(tp.sum() / max(int(cm.sum()), 1))


def main():
    a = arguments()
    for path in (a.raw, a.split_dir, a.dataset_yaml, a.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    model = DenseYOLO(a.weights, a.device, a.conf, a.imgsz, a.background_score).eval()
    codes, names = raw_code_mapping(a.dataset_yaml, model)
    frames = collect_frames(a.raw, a.split_dir)
    if a.limit:
        frames = frames[:a.limit]
    print(f'Frames: {len(frames)}; classes: {model.n_foreground}; raw codes: {codes}', flush=True)
    cm = {key: np.zeros((len(codes)+1, len(codes)+1), np.int64) for key in ('baseline', 'stitch')}
    cfg = build_default_cfg()
    cfg.device = device
    cfg.num_classes = len(codes) + 1
    cfg.output_mode = 'softmax'  # output is comparable class scores, not calibrated probabilities
    cfg.use_static_roi = False
    cfg.apply_ellipse_mask = False
    cfg.equalize_hist_rgb = False
    cfg.tool_class_ch = None
    cfg.inject_tool_mask = False
    stitch = StitchNoAuxMasks(model=model, cfg=cfg).eval()
    previous_clip = None
    timings = {'baseline': [], 'stitch': []}
    for idx, image in enumerate(frames):
        clip = str(image.parent)
        if clip != previous_clip:
            stitch.reset_state(clear_preprocess_state=True)
            previous_clip = clip
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image)
        gt = read_gt(find_gt(image), codes)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            if device.type == 'cuda': torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            baseline = model(x).argmax(1)[0]
            if device.type == 'cuda': torch.cuda.synchronize(device)
            timings['baseline'].append(time.perf_counter()-t0)
            frame = stitch.preprocess_frame(bgr)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            stitched = stitch(frame)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            timings['stitch'].append(time.perf_counter()-t0)
            if stitched is None:
                raise RuntimeError(f'Stitch returned None for {image}')
            if stitched.ndim != 3 or stitched.shape[0] != len(codes)+1:
                raise ValueError(f'Unexpected stitch output shape: {tuple(stitched.shape)}')
            stitched = stitched.argmax(0)
        baseline = baseline.cpu().numpy().astype(np.uint8)
        stitched = stitched.cpu().numpy().astype(np.uint8)
        update_cm(cm['baseline'], gt, baseline)
        update_cm(cm['stitch'], gt, stitched)
        if a.save_every and (idx % a.save_every == 0):
            example_dir = a.out / 'examples'
            example_dir.mkdir(exist_ok=True)
            prefix = f'{idx:05d}_{image.stem}'
            cv2.imwrite(str(example_dir / f'{prefix}_gt.png'), gt)
            cv2.imwrite(str(example_dir / f'{prefix}_baseline.png'), baseline)
            cv2.imwrite(str(example_dir / f'{prefix}_stitch.png'), stitched)
        if idx == 0 or (idx+1) % 25 == 0:
            print(f'{idx+1}/{len(frames)} {image.name}', flush=True)
    with (a.out/'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method','foreground_mIoU','foreground_mDice','pixel_accuracy','mean_seconds_per_frame'])
        for method in cm:
            iou, dice, accuracy = scores(cm[method])
            values = [method, float(np.nanmean(iou[1:])), float(np.nanmean(dice[1:])), accuracy,
                      float(np.mean(timings[method]))]
            writer.writerow(values)
            print(values, flush=True)
    with (a.out/'per_class.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method','class_index','name','raw_code','iou','dice','gt_pixels','pred_pixels'])
        for method in cm:
            iou, dice, _ = scores(cm[method])
            for i in range(len(iou)):
                writer.writerow([method, i, 'background' if i == 0 else names[i-1],
                                 0 if i == 0 else codes[i-1], iou[i], dice[i],
                                 cm[method].sum(1)[i], cm[method].sum(0)[i]])
        np.save(a.out/'cm_baseline.npy', cm['baseline'])
        np.save(a.out/'cm_stitch.npy', cm['stitch'])
    (a.out/'settings.json').write_text(json.dumps(vars(a), default=str, indent=2))
    print('Saved results to', a.out)


if __name__ == '__main__':
    main()
