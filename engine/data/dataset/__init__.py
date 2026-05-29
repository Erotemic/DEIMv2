"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator

# kit extension — WebDataset-backed CocoDetection. Import for side-effect
# so the @register() decorator runs and YAMLConfig can resolve
# `type: WebDatasetCocoDetection`. Optional: skip silently when its
# runtime deps (webdataset, kwcoco_dataloader) aren't installed.
try:
    from .wds_coco_dataset import WebDatasetCocoDetection  # noqa: F401
except Exception:
    pass
