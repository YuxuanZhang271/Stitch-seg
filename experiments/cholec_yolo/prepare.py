
#!/usr/bin/env python3
"""
Prepare CholecSeg8k for Ultralytics YOLO segmentation.

Input:
    data/raw/CholecSeg8k/videoXX/videoXX_YYYYY/
        frame_N_endo.png
        frame_N_endo_watershed_mask.png

Output:
    data/yolo_cholec/
        dataset.yaml
        raw_class_mapping.yaml
        images/train/
        images/val/
        labels/train/
        labels/val/

Run:
    python experiments/cholec_yolo/prepare.py \
        --raw data/raw/CholecSeg8k \
        --scan-only

    python experiments/cholec_yolo/prepare.py \
        --raw data/raw/CholecSeg8k \
        --out data/yolo_cholec
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


# Raw values from:
# exp_seg/cholecseg8k_benchmark/split.py
#
# 0   = background
# 255 = ignored label
#
# Other values become YOLO foreground classes.
#
# Names are deliberately based on raw codes until the
# semantic class-name mapping is independently verified.

RAW_FOREGROUND_CODES = [
    5,
    11,
    12,
    13,
    21,
    22,
    23,
    24,
    25,
    31,
    32,
    33,
    50,
]

BACKGROUND_CODE = 0
IGNORE_CODE = 255

RAW_TO_YOLO = {
    raw_code: class_id
    for class_id, raw_code in enumerate(RAW_FOREGROUND_CODES)
}

CLASS_NAMES = [
    f"raw_{code}" for code in RAW_FOREGROUND_CODES
]

ALLOWED_VALUES = (
    set(RAW_FOREGROUND_CODES)
    | {BACKGROUND_CODE, IGNORE_CODE}
)


def read_mask(path):
    """Read a CholecSeg8k annotation as a 2D integer array."""

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if mask is None:
        raise FileNotFoundError(
            f"Cannot read annotation: {path}"
        )

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]

        if not (
            np.array_equal(mask[:, :, 0], mask[:, :, 1])
            and np.array_equal(mask[:, :, 1], mask[:, :, 2])
        ):
            raise ValueError(
                "Mask channels are not identical. "
                f"Check annotation encoding: {path}"
            )

        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D annotation, got {mask.shape}: {path}"
        )

    return mask


def find_frames(root):
    """Find all original endoscopic frames."""

    frames = sorted(root.rglob("*_endo.png"))

    if not frames:
        raise FileNotFoundError(
            f"No *_endo.png frames found under {root}"
        )

    return frames


def get_video_name(frame, root):
    """Get videoXX from the path relative to dataset root."""

    relative = frame.relative_to(root)

    for part in relative.parts:
        if part.startswith("video") and "_" not in part:
            return part

    raise ValueError(
        f"Cannot determine video name from {frame}"
    )


def mask_path_for_frame(frame):
    """Locate the benchmark-compatible watershed annotation."""

    return frame.with_name(
        frame.stem + "_watershed_mask.png"
    )


def scan_dataset(frames):
    """Validate all masks and collect raw pixel-value counts."""

    counts = Counter()
    missing = []
    unexpected = Counter()

    for index, frame in enumerate(frames, start=1):
        mask_path = mask_path_for_frame(frame)

        if not mask_path.exists():
            missing.append(str(mask_path))
            continue

        mask = read_mask(mask_path)

        values, frequencies = np.unique(
            mask,
            return_counts=True,
        )

        for value, frequency in zip(values, frequencies):
            value = int(value)
            frequency = int(frequency)

            counts[value] += frequency

            if value not in ALLOWED_VALUES:
                unexpected[value] += frequency

        if index % 1000 == 0:
            print(
                f"Scanned {index}/{len(frames)} frames"
            )

    print("\nDataset scan")
    print("-" * 50)
    print("Frames:", len(frames))
    print("Raw pixel counts:")

    for value, count in sorted(counts.items()):
        print(f"  {value:3d}: {count:,}")

    if missing:
        print("\nMissing annotations:")
        for path in missing[:10]:
            print(" ", path)

        raise FileNotFoundError(
            f"{len(missing)} annotation masks are missing."
        )

    if unexpected:
        raise ValueError(
            "Unexpected annotation values: "
            f"{dict(sorted(unexpected.items()))}. "
            "Verify the annotation encoding before conversion."
        )

    print("\nAll mask values are recognized.")

    return counts


def make_video_split(frames, root, val_fraction, seed):
    """Split complete videos, never individual frames."""

    videos = sorted({
        get_video_name(frame, root)
        for frame in frames
    })

    if len(videos) < 2:
        raise ValueError(
            "At least two videos are required for train/val splitting."
        )

    rng = random.Random(seed)
    rng.shuffle(videos)

    n_val = max(
        1,
        min(
            len(videos) - 1,
            round(len(videos) * val_fraction),
        ),
    )

    val_videos = set(videos[:n_val])
    train_videos = set(videos[n_val:])

    print("\nTraining videos:", sorted(train_videos))
    print("Validation videos:", sorted(val_videos))

    return train_videos, val_videos


def mask_to_yolo_polygons(mask, min_area=4.0, epsilon=1.0):
    """Convert semantic foreground regions into YOLO polygons.

    Note:
        RETR_EXTERNAL does not preserve holes within regions.
        This is a limitation of polygon-based YOLO supervision.
    """

    height, width = mask.shape
    lines = []

    for raw_code, class_id in RAW_TO_YOLO.items():
        binary = np.uint8(mask == raw_code)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            polygon = cv2.approxPolyDP(
                contour,
                epsilon=epsilon,
                closed=True,
            ).reshape(-1, 2)

            if len(polygon) < 3:
                continue

            # YOLO expects normalized coordinates.
            coordinates = []

            for x, y in polygon:
                x_norm = np.clip(
                    float(x) / width, 0.0, 1.0
                )
                y_norm = np.clip(
                    float(y) / height, 0.0, 1.0
                )

                coordinates.extend([
                    f"{x_norm:.6f}",
                    f"{y_norm:.6f}",
                ])

            lines.append(
                f"{class_id} " + " ".join(coordinates)
            )

    return lines


def prepare_dataset(
    frames,
    raw_root,
    output_root,
    val_videos,
    min_area,
    epsilon,
):
    """Create YOLO image and polygon-label directories."""

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (output_root / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    split_counts = Counter()

    for index, frame in enumerate(frames, start=1):
        video = get_video_name(frame, raw_root)

        split = (
            "val" if video in val_videos else "train"
        )

        # Keep original video, clip and frame identifiers.
        unique_stem = (
            f"{video}_{frame.parent.name}_{frame.stem}"
        )

        image_destination = (
            output_root
            / "images"
            / split
            / f"{unique_stem}.png"
        )

        label_destination = (
            output_root
            / "labels"
            / split
            / f"{unique_stem}.txt"
        )

        mask_path = mask_path_for_frame(frame)
        mask = read_mask(mask_path)

        # Verify each annotation before writing.
        unknown = set(np.unique(mask).tolist()) - ALLOWED_VALUES

        if unknown:
            raise ValueError(
                f"Unexpected values {sorted(unknown)} "
                f"in {mask_path}"
            )

        polygons = mask_to_yolo_polygons(
            mask,
            min_area=min_area,
            epsilon=epsilon,
        )

        shutil.copy2(
            frame,
            image_destination,
        )

        label_destination.write_text(
            "\n".join(polygons) + ("\n" if polygons else ""),
            encoding="utf-8",
        )

        split_counts[split] += 1

        if index % 1000 == 0:
            print(
                f"Converted {index}/{len(frames)} frames"
            )

    return split_counts


def write_configs(output_root, train_videos, val_videos):
    """Write Ultralytics YAML and raw-label metadata."""

    dataset_config = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {
            i: name
            for i, name in enumerate(CLASS_NAMES)
        },
    }

    mapping_config = {
        "background_raw_value": BACKGROUND_CODE,
        "ignore_raw_value": IGNORE_CODE,
        "raw_to_yolo": RAW_TO_YOLO,
        "class_names": CLASS_NAMES,
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
        "annotation_suffix": "_endo_watershed_mask.png",
    }

    with open(
        output_root / "dataset.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            dataset_config,
            file,
            sort_keys=False,
        )

    with open(
        output_root / "raw_class_mapping.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            mapping_config,
            file,
            sort_keys=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert CholecSeg8k to YOLO-Seg."
    )

    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Original CholecSeg8k root directory.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/yolo_cholec"),
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Validate annotations without converting.",
    )

    args = parser.parse_args()

    if not args.raw.is_dir():
        parser.error(
            f"Dataset directory does not exist: {args.raw}"
        )

    if not 0 < args.val_fraction < 1:
        parser.error(
            "--val-fraction must be between 0 and 1."
        )

    frames = find_frames(args.raw)

    print(f"Found {len(frames)} frames.")

    # Scan the complete dataset before any conversion.
    scan_dataset(frames)

    if args.scan_only:
        print("\nScan complete. No files were generated.")
        return

    train_videos, val_videos = make_video_split(
        frames,
        args.raw,
        args.val_fraction,
        args.seed,
    )

    counts = prepare_dataset(
        frames=frames,
        raw_root=args.raw,
        output_root=args.out,
        val_videos=val_videos,
        min_area=args.min_area,
        epsilon=args.epsilon,
    )

    write_configs(
        args.out,
        train_videos,
        val_videos,
    )

    print("\nConversion complete!")
    print("Training frames:", counts["train"])
    print("Validation frames:", counts["val"])
    print("Total:", sum(counts.values()))
    print("Dataset YAML:", args.out / "dataset.yaml")
    print("Mapping YAML:", args.out / "raw_class_mapping.yaml")


if __name__ == "__main__":
    main()
