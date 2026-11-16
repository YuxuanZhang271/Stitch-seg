"""Stitch variant using original-frame YOLO prompts transformed into canvas-crop coordinates.

No changes to stitch_seg itself. The base class calls _current_canvas_bbox immediately
before invoking the segmentation model, so this hook installs prompts for that call.
"""
import numpy as np
import torch
from stitch_no_masks import StitchNoAuxMasks


class StitchFixedPrompts(StitchNoAuxMasks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_prompts = []
        self.transformed_count = 0

    def _current_canvas_bbox(self, frame_shape):
        bbox = super()._current_canvas_bbox(frame_shape)
        self.model.external_prompts = []  # explicit empty list means do not run YOLO
        self.transformed_count = 0
        if any(v is None for v in bbox):
            return bbox
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            return bbox
        H = self._current_to_canvas_h().detach().float().cpu().numpy()
        transformed = []
        for box, cls, confidence in self.original_prompts:
            xa, ya, xb, yb = map(float, box)
            corners = np.array([[xa, ya, 1], [xb, ya, 1],
                                [xb, yb, 1], [xa, yb, 1]], dtype=np.float64)
            projected = (H @ corners.T).T
            if not np.isfinite(projected).all() or np.any(np.abs(projected[:, 2]) < 1e-8):
                continue
            xy = projected[:, :2] / projected[:, 2:3]
            if not np.isfinite(xy).all():
                continue
            left = float(np.clip(xy[:, 0].min() - x1, 0, crop_w))
            top = float(np.clip(xy[:, 1].min() - y1, 0, crop_h))
            right = float(np.clip(xy[:, 0].max() - x1, 0, crop_w))
            bottom = float(np.clip(xy[:, 1].max() - y1, 0, crop_h))
            if right - left < 2 or bottom - top < 2:
                continue
            transformed.append((np.array([left, top, right, bottom], dtype=np.float32),
                                int(cls), float(confidence)))
        self.model.external_prompts = transformed
        self.transformed_count = len(transformed)
        return bbox
