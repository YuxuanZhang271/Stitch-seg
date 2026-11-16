"""Motion-compensated consecutive-frame consistency metrics.

Flow is computed ONLY from original RGB frames, identically for every method.
Use backward Farneback flow (current -> previous) for cv2.remap; optionally
filter occlusions using forward-backward consistency. Pixels with GT ignore
label in either frame are excluded. This is a stability metric, not accuracy.
"""
import cv2
import numpy as np


class TemporalEvaluator:
    def __init__(self, methods, class_names, fb_threshold=2.0, flow_scale=0.5):
        if not 0 < flow_scale <= 1:
            raise ValueError('flow_scale must be in (0, 1]')
        self.methods = tuple(methods)
        self.names = tuple(class_names)
        self.n = len(self.names)
        self.fb_threshold = float(fb_threshold)
        self.flow_scale = float(flow_scale)
        self.previous = None
        self.rows = []
        self.class_rows = []
        self.acc = {m: dict(inter=np.zeros(self.n, np.int64),
                            union=np.zeros(self.n, np.int64),
                            a=np.zeros(self.n, np.int64),
                            b=np.zeros(self.n, np.int64),
                            valid=0, changed=0, pairs=0,
                            macro_iou=[], macro_dice=[], area_delta=[]) for m in self.methods}

    def reset(self):
        self.previous = None

    def _flow(self, current_gray, previous_gray):
        h, w = current_gray.shape
        if self.flow_scale == 1.0:
            curr, prev = current_gray, previous_gray
        else:
            size = (max(16, round(w*self.flow_scale)), max(16, round(h*self.flow_scale)))
            curr = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
            prev = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
        f = cv2.calcOpticalFlowFarneback(curr, prev, None, .5, 3, 21, 3, 5, 1.2, 0)
        if f.shape[:2] != (h, w):
            sh, sw = f.shape[:2]
            f = cv2.resize(f, (w,h), interpolation=cv2.INTER_LINEAR)
            f[...,0] *= w/sw
            f[...,1] *= h/sh
        return f

    def step(self, frame_index, frame_name, clip, bgr, gt, predictions):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.previous is None:
            self.previous = (gray.copy(), gt.copy(), {k:v.copy() for k,v in predictions.items()})
            return
        prev_gray, prev_gt, prev_preds = self.previous
        if gray.shape != prev_gray.shape:
            self.reset()
            self.step(frame_index, frame_name, clip, bgr, gt, predictions)
            return
        h,w = gray.shape
        backward = self._flow(gray, prev_gray) # current -> previous
        forward = self._flow(prev_gray, gray) # previous -> current
        yy,xx = np.mgrid[:h,:w].astype(np.float32)
        px = xx + backward[...,0]
        py = yy + backward[...,1]
        inside = (px >= 0)&(px <= w-1)&(py >= 0)&(py <= h-1)
        sampled_forward = cv2.remap(forward, px, py, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fb_error = np.linalg.norm(backward + sampled_forward, axis=-1)
        valid = inside & np.isfinite(fb_error) & (fb_error <= self.fb_threshold)
        warped_gt = cv2.remap(prev_gt, px, py, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        valid &= (gt != 255) & (warped_gt != 255)
        valid_n = int(valid.sum())
        for method in self.methods:
            prev_pred = prev_preds[method]
            curr_pred = predictions[method]
            warped = cv2.remap(prev_pred, px, py, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            # Global per-pair metrics are micro-averaged across classes (incl background)
            changed = int(np.count_nonzero((warped != curr_pred) & valid))
            ious, dices, area_deltas = [], [], []
            for c in range(self.n):
                a = (warped == c) & valid
                b = (curr_pred == c) & valid
                intersection = int(np.count_nonzero(a & b))
                union = int(np.count_nonzero(a | b))
                na, nb = int(a.sum()), int(b.sum())
                agg = self.acc[method]
                agg['inter'][c] += intersection
                agg['union'][c] += union
                agg['a'][c] += na
                agg['b'][c] += nb
                iou = intersection / union if union else np.nan
                dice = 2*intersection/(na+nb) if na+nb else np.nan
                area_delta = abs(nb-na)/valid_n if valid_n else np.nan
                if c > 0:
                    ious.append(iou)
                    dices.append(dice)
                    area_deltas.append(area_delta)
                self.class_rows.append(dict(frame_index=frame_index, frame=frame_name,
                                            clip=clip, method=method, class_index=c,
                                            name=self.names[c], intersection=intersection,
                                            union=union, previous_pixels=na,
                                            current_pixels=nb, temporal_iou=iou,
                                            temporal_dice=dice, abs_area_fraction_change=area_delta))
            mean_iou = float(np.nanmean(ious)) if np.isfinite(ious).any() else np.nan
            mean_dice = float(np.nanmean(dices)) if np.isfinite(dices).any() else np.nan
            mean_area = float(np.nanmean(area_deltas)) if np.isfinite(area_deltas).any() else np.nan
            self.rows.append(dict(frame_index=frame_index, frame=frame_name, clip=clip,
                                  method=method, valid_pixels=valid_n,
                                  valid_fraction=valid_n/(h*w),
                                  temporal_foreground_miou_present=mean_iou,
                                  temporal_foreground_mdice_present=mean_dice,
                                  label_change_rate=changed/valid_n if valid_n else np.nan,
                                  mean_foreground_abs_area_fraction_change=mean_area))
            agg = self.acc[method]
            agg['valid'] += valid_n
            agg['changed'] += changed
            agg['pairs'] += 1
            if np.isfinite(mean_iou): agg['macro_iou'].append(mean_iou)
            if np.isfinite(mean_dice): agg['macro_dice'].append(mean_dice)
            if np.isfinite(mean_area): agg['area_delta'].append(mean_area)
        self.previous = (gray.copy(), gt.copy(), {k:v.copy() for k,v in predictions.items()})

    def summary(self):
        rows=[]
        for method, agg in self.acc.items():
            union = agg['union'][1:]
            den = agg['a'][1:] + agg['b'][1:]
            iou = np.divide(agg['inter'][1:], union,
                            out=np.full(len(union), np.nan), where=union>0)
            dice = np.divide(2*agg['inter'][1:], den,
                             out=np.full(len(den), np.nan), where=den>0)
            rows.append(dict(method=method, frame_pairs=agg['pairs'],
                             valid_pixels=agg['valid'],
                             foreground_temporal_mIoU=float(np.nanmean(iou)) if np.isfinite(iou).any() else np.nan,
                             foreground_temporal_mDice=float(np.nanmean(dice)) if np.isfinite(dice).any() else np.nan,
                             mean_per_pair_foreground_mIoU=float(np.mean(agg['macro_iou'])) if agg['macro_iou'] else np.nan,
                             label_change_rate=agg['changed']/agg['valid'] if agg['valid'] else np.nan,
                             mean_foreground_abs_area_fraction_change=float(np.mean(agg['area_delta'])) if agg['area_delta'] else np.nan))
        return rows
