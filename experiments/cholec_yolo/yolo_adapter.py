"""Ultralytics instance masks -> dense semantic scores for StitchInferencer."""
import numpy as np
import torch
from ultralytics import YOLO


class DenseYOLO(torch.nn.Module):
    def __init__(self, weights, device='cuda:0', conf=0.10, imgsz=640, background_score=0.5):
        super().__init__()
        # self.yolo = YOLO(str(weights))
        self.__dict__["yolo"] = YOLO(str(weights))
        if self.yolo.task != 'segment':
            raise ValueError('Checkpoint must be an Ultralytics segmentation model')
        self.n_foreground = len(self.yolo.names)
        self.device_name = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.background_score = float(background_score)

    @torch.inference_mode()
    def forward(self, x):
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
            raise ValueError(f'Expected RGB [1,3,H,W], got {tuple(x.shape)}')
        h, w = x.shape[-2:]
        rgb = x[0].detach().permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        result = self.yolo.predict(source=rgb, device=self.device_name, conf=self.conf,
                                   imgsz=self.imgsz, retina_masks=True, verbose=False)[0]
        scores = np.zeros((self.n_foreground + 1, h, w), np.float32)
        scores[0] = self.background_score
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.int().cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            import cv2
            for mask, cls, conf in zip(masks, classes, confidences):
                if not (0 <= int(cls) < self.n_foreground):
                    continue
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                scores[int(cls) + 1] = np.maximum(scores[int(cls) + 1], mask * float(conf))
        return torch.from_numpy(scores[None]).to(x.device)
