"""SAM2 with YOLO-box prompting, producing semantic scores for StitchInferencer.

This is a YOLO-assisted pipeline, not a standalone or zero-shot SurgiSAM2 model.
"""
import numpy as np
import torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class DenseSurgiSAM2(torch.nn.Module):
    def __init__(self, weights, sam_config, detector_weights, device='cuda:0', conf=0.10,
                 imgsz=640, background_score=0.5, mask_threshold=0.0):
        super().__init__()
        self.device_name=device
        self.__dict__["detector"] = YOLO(str(detector_weights))
        # Keep Ultralytics outside nn.Module children: its train() starts a training run.
        if self.detector.task != 'segment':
            raise ValueError('Expected YOLO segmentation checkpoint for prompts')
        self.n_foreground=len(self.detector.names)
        self.names=self.detector.names
        self.conf=float(conf);self.imgsz=int(imgsz)
        self.background_score=float(background_score)
        self.mask_threshold=float(mask_threshold)
        sam=build_sam2(sam_config,str(weights),device=device,mode='eval')
        self.predictor=SAM2ImagePredictor(sam)
        self.predictor.model.eval()
        self.diagnostic_context = None
        self.diagnostic_records = []
        self.external_prompts = None  # None: YOLO; list: use fixed class-labeled boxes
        self.last_detections = []

    def train(self, mode=True):
        """Never dispatch nn.Module.train() to the Ultralytics YOLO wrapper."""
        super().train(mode)
        self.predictor.model.eval()
        return self

    @torch.inference_mode()
    def forward(self,x):
        if x.ndim!=4 or x.shape[0]!=1 or x.shape[1]!=3:
            raise ValueError(f'Expected [1,3,H,W], got {tuple(x.shape)}')
        h,w=x.shape[-2:]
        rgb=x[0].detach().permute(1,2,0).cpu().numpy()
        rgb=np.clip(rgb,0,255).astype(np.uint8)
        external = self.external_prompts
        self.external_prompts = None  # consume exactly once, even for empty prompts
        result = None
        if external is None:
            result=self.detector.predict(source=rgb,device=self.device_name,conf=self.conf,
                                         imgsz=self.imgsz,verbose=False)[0]
        scores=np.zeros((self.n_foreground+1,h,w),np.float32)
        scores[0]=self.background_score
        context = self.diagnostic_context
        record = None
        if context is not None:
            record = dict(context)
            record.update(input_height=int(h), input_width=int(w), detections=[])
            self.diagnostic_records.append(record)
        if external is None:
            if result.boxes is None or len(result.boxes)==0:
                self.last_detections = []
                return torch.from_numpy(scores[None]).to(x.device)
            boxes=result.boxes.xyxy.detach().cpu().numpy()
            classes=result.boxes.cls.int().detach().cpu().numpy()
            confidences=result.boxes.conf.detach().cpu().numpy()
            self.last_detections = [
                (box.copy(), int(cls), float(confidence))
                for box, cls, confidence in zip(boxes, classes, confidences)
            ]
        else:
            self.last_detections = []
            if not external:
                return torch.from_numpy(scores[None]).to(x.device)
            boxes = [entry[0] for entry in external]
            classes = [entry[1] for entry in external]
            confidences = [entry[2] for entry in external]
        self.predictor.set_image(rgb)
        for box,cls,confidence in zip(boxes,classes,confidences):
            cls=int(cls)
            if not 0<=cls<self.n_foreground: continue
            full_res_logits,_,low_res_logits=self.predictor.predict(
                box=box[None,:], multimask_output=False, return_logits=True)
            # SAM2 logits represent foreground evidence; threshold to avoid
            # turning all pixels into foreground via sigmoid's positive floor.
            mask=(np.asarray(full_res_logits)[0]>self.mask_threshold).astype(np.float32)
            if mask.shape!=(h,w): raise ValueError(f'Unexpected SAM2 mask shape {mask.shape}')
            scores[cls+1]=np.maximum(scores[cls+1],mask*float(confidence))
            if record is not None:
                record['detections'].append({
                    'class_index': cls + 1,
                    'confidence': float(confidence),
                    'box_xyxy': [float(v) for v in box],
                    'box_area_fraction': float(max(0,box[2]-box[0])*max(0,box[3]-box[1])/(h*w)),
                    'sam_mask_area_fraction': float(mask.mean()),
                    'passes_background_score': bool(confidence > self.background_score),
                })
        return torch.from_numpy(scores[None]).to(x.device)
