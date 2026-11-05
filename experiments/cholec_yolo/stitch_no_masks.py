"""Keep the original StitchInferencer geometry, bypass unavailable auxiliary U-Nets.

The patch only replaces the two constructor-time masking model loaders. Its
_predict_masks implementation supplies an all-zero exclusion mask. This is a
no-masking ablation, NOT an equivalent replacement for the authors' U-Nets.
"""
from unittest.mock import patch
import torch
import stitch_seg.inferencer as inferencer_module


class StitchNoAuxMasks(inferencer_module.StitchInferencer):
    def __init__(self, *args, **kwargs):
        with patch.object(inferencer_module, 'load_masking_model', return_value=torch.nn.Identity()), \
             patch.object(inferencer_module, 'load_masking_model2', return_value=torch.nn.Identity()):
            super().__init__(*args, **kwargs)

    def _predict_masks(self, frame_u):
        h, w = frame_u.shape[-2:]
        return torch.zeros((1, 1, h, w), dtype=torch.uint8, device=frame_u.device)
