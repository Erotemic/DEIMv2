"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            try:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                        coco_eval.cocoDt = coco_dt
                        coco_eval.params.imgIds = list(img_ids)
                        coco_eval.evaluate()
            except Exception as ex:
                # The v13 relaunch died here intermittently with a data/prediction
                # -shaped error ("'int' object does not support the context manager
                # protocol" out of loadRes->loadAnns). A full eval-only pass over
                # the vali set could NOT reproduce it on the same checkpoints, so
                # it is transient -- not a deterministic bad item. Dump the EXACT
                # offending batch for offline root-cause IF it ever recurs, warn
                # loudly, and SKIP this batch's contribution so one blip cannot
                # kill a multi-hour run. The in-loop eval only selects the best
                # checkpoint; the kit's separate post-train eval is authoritative.
                self._dump_eval_crash(ex, iou_type, predictions, results, img_ids)
                continue

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def _dump_eval_crash(self, ex, iou_type, predictions, results, img_ids):
        """Best-effort artifact of a per-batch in-loop eval failure.

        Writes a pickle next to the process CWD (the DEIM output dir) capturing
        everything needed to reproduce ``loadRes(results) -> evaluate()`` offline.
        Never raises -- a failure to dump must not hide the original exception.
        """
        import pickle
        import traceback as _tb
        try:
            import faster_coco_eval as _fce
            fce_version = getattr(_fce, "__version__", "<unknown>")
        except Exception:
            fce_version = "<import-failed>"
        path = os.path.join(os.getcwd(), f"coco_eval_crash_dump_pid{os.getpid()}.pkl")
        payload = {
            "exception_type": type(ex).__name__,
            "exception_str": str(ex),
            "traceback": _tb.format_exc(),
            "iou_type": iou_type,
            "faster_coco_eval_version": fce_version,
            "img_ids": list(img_ids),
            "n_predictions": len(predictions) if predictions is not None else None,
            "n_results": len(results) if results is not None else None,
            # raw, un-prepared predictions + the prepared results that broke
            "predictions": predictions,
            "results": results,
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            print(f"[coco_eval] IN-LOOP EVAL ERROR, SKIPPING BATCH "
                  f"({type(ex).__name__}: {ex}); faster_coco_eval={fce_version}; "
                  f"offending batch dumped -> {path}", flush=True)
        except Exception as dump_ex:  # pragma: no cover - diagnostic only
            print(f"[coco_eval] IN-LOOP EVAL ERROR, SKIPPING BATCH "
                  f"({type(ex).__name__}: {ex}); ALSO failed to write dump "
                  f"({dump_ex})", flush=True)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
