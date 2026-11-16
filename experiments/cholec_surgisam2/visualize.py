
#!/usr/bin/env python3
"""Visualize GT, baseline YOLO, and stitched semantic predictions."""

import argparse
from pathlib import Path

import cv2
import numpy as np


# Class 0 = background; classes 1-13 = YOLO foreground classes.
COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [255, 128, 0],
], dtype=np.uint8)


def colorize(mask):
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    if mask.max() >= len(COLORS):
        raise ValueError(f"Unexpected class value: {mask.max()}")

    return COLORS[mask.astype(np.int64)]


def add_title(image, title):
    image = image.copy()
    cv2.rectangle(
        image, (0, 0), (image.shape[1], 38),
        (30, 30, 30), -1
    )
    cv2.putText(
        image, title, (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/yolo_pilot100")
    )
    args = parser.parse_args()

    examples = args.results / "examples"
    output = args.results / "comparisons"
    output.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(examples.glob("*_gt.png"))

    if not gt_files:
        raise FileNotFoundError(
            f"No ground-truth masks found in {examples}"
        )

    for gt_path in gt_files:
        prefix = gt_path.name[:-7]

        panels = []

        names = ("gt", "baseline", "stitch", "stitch_fixed") if (examples / f"{prefix}_stitch_fixed.png").exists() else ("gt", "baseline", "stitch")
        for name in names:
            path = examples / f"{prefix}_{name}.png"

            mask = cv2.imread(
                str(path), cv2.IMREAD_UNCHANGED
            )

            if mask is None:
                raise FileNotFoundError(path)

            # 255 is the ignored label in the GT.
            # Display it in gray, not as a semantic class.
            ignored = mask == 255
            safe_mask = mask.copy()
            safe_mask[ignored] = 0

            colored = colorize(safe_mask)
            colored[ignored] = (80, 80, 80)

            title = {
                "gt": "GROUND TRUTH",
                "baseline": "SURGISAM2 BASELINE",
                "stitch": "SURGISAM2 + STITCH",
                "stitch_fixed": "STITCH + FIXED PROMPTS",
            }[name]

            panels.append(add_title(colored, title))

        comparison = np.concatenate(panels, axis=1)

        cv2.imwrite(
            str(output / f"{prefix}_comparison.png"),
            comparison
        )

    print(f"Generated {len(gt_files)} comparisons")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
