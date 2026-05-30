"""
WebDataset-backed CocoDetection.

Reads kwcoco_dataloader detection shards (one tar per ``dominant_raw_class``
bucket, with ``<key>.jpg`` + ``<key>.json`` pairs) and yields samples in
the same ``(PIL.Image, target_dict)`` format that DEIMv2's stock
:class:`CocoDetection` does, so the downstream transforms (Mosaic with
cache, MixUp, CopyBlend, photometric, resize, etc.) and the
:class:`BatchImageCollateFunction` work unchanged.

Designed for **HDD-friendly streaming**: each tar shard is read
sequentially; samples are mixed across buckets by
:class:`WeightedChunkMix`. Random access is intentionally not supported
because that's the property we'd be giving up.

Caveats / constraints:
  * Mosaic must run with ``use_cache=True`` (the only path that doesn't
    need ``dataset[idx]`` / ``len(dataset)``). Our generated configs
    already do this. Non-cache mode will raise informative errors.
  * DDP rank-splitting happens in
    :func:`kwcoco_dataloader.readers.detection.load_bucket_streams`;
    one set of streams is opened per (DDP rank, dataloader worker) pair.
  * Category remapping: the dataset receives ``category_names`` (in
    class-index order) and emits ``target['labels']`` as 0-indexed
    class IDs matching that list. The kit's
    ``relabel_detection_sample`` applies the source -> target collapse
    so each sample's annotations come in already-relabeled.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.utils.data
from PIL import Image

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class WebDatasetCocoDetection(torch.utils.data.IterableDataset, DetDataset):
    """DEIMv2 dataset adapter for kwcoco_dataloader WebDataset shards.

    The kit's _build_train_yml emits ``type: WebDatasetCocoDetection`` for
    this class when ``KCD_USE_WEBDATASET=1``. The config-level fields
    map onto __init__ kwargs through DEIMv2's YAMLConfig machinery.

    Args:
        shards_dpath: directory containing bucketed shard subdirs, e.g.
            ``<root>/dominant_raw_class_EQ_P/``, ``..._EQ_B/`` ... .
            Produced by ``kwcoco_dataloader.cli.build_detection_webdataset``.
        category_names: ordered list of TARGET class names (the same list
            that's passed to the kit at training time). Defines the
            integer class IDs emitted in ``target['labels']``.
        source_to_target: mapping from raw source category name (as
            written into each sample's annotations by the writer) to
            the target class name in ``category_names``. Source classes
            absent from this mapping are dropped from the sample's
            annotations (they're treated as background, like the kit's
            apply_scheme step does).
        transforms: DEIMv2 transform stack (injected via __inject__).
        bucket_weights: optional dict mapping bucket name -> float weight
            for :class:`WeightedChunkMix`. Defaults to equal weights.
        chunk_size: WeightedChunkMix block size (default 1).
        num_workers_hint: heuristic for splitting shards across workers
            in :func:`load_bucket_streams`.
        epoch_length: nominal samples per epoch (used to terminate
            ``__iter__`` if the source is infinite). 0 means "drain the
            streams once and stop".
    """

    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(
        self,
        shards_dpath: str,
        category_names: Sequence[str],
        source_to_target: Optional[dict] = None,
        transforms=None,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
        bucket_weights: Optional[dict] = None,
        chunk_size: int = 1,
        num_workers_hint: int = 4,
        epoch_length: int = 0,
        # Upstream CocoDetection config keys that the YAML merger leaks
        # through from configs/dataset/coco_detection.yml when our
        # __include__ chain inherits from it. Accept-and-ignore so
        # YAMLConfig.create()'s kwarg-passthrough doesn't error on
        # them. The streaming reader resolves images from the shard
        # tar archives, not from a filesystem path.
        img_folder: Optional[str] = None,
        ann_file: Optional[str] = None,
    ):
        super().__init__()
        from kwcoco_dataloader.readers.detection import (
            SchemeMapping, load_bucket_streams, WeightedChunkMix,
        )

        self._transforms = transforms
        self.return_masks = bool(return_masks)
        self.remap_mscoco_category = bool(remap_mscoco_category)
        self.category_names = list(category_names)
        self.shards_dpath = Path(shards_dpath)

        if source_to_target is None:
            # Identity mapping when no scheme collapse needed (e.g.
            # single_sealion where every kept source class is "sealion").
            source_to_target = {n: n for n in self.category_names}
        # SchemeMapping's API uses (target_order, mapping) names.
        # `unmapped_policy='drop'` (the default) means raw source
        # classes not in `mapping` are silently dropped per-sample.
        # That handles the scheme YAML's `drop:` list implicitly:
        # raw labels there are absent from the mapping, so they fall
        # through to the unmapped policy.
        self.scheme = SchemeMapping(
            target_order=list(self.category_names),
            mapping=dict(source_to_target),
        )

        self._bucket_weights = bucket_weights or {}
        self._chunk_size = int(chunk_size)
        self._num_workers_hint = int(num_workers_hint)
        self._epoch_length = int(epoch_length)
        self._epoch = 0

        # Probe at construction so a missing shards path fails loudly,
        # not after the first epoch starts.
        self._buckets = self._discover_buckets()
        if not self._buckets:
            raise FileNotFoundError(
                f"no bucketed shard subdirs under {self.shards_dpath}; "
                "did build_detection_webdataset run?"
            )

    # ----- bookkeeping ------------------------------------------------

    def _discover_buckets(self) -> List[Path]:
        return sorted(
            d for d in self.shards_dpath.iterdir()
            if d.is_dir() and any(d.glob("*.tar"))
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    # ----- DetDataset interface that some transforms touch -----------
    #
    # IterableDataset doesn't have len/getitem. Mosaic with use_cache=True
    # never calls these; use_cache=False would. Raise informative errors
    # so misconfigs fail fast rather than producing junk batches.

    def __len__(self):
        """Total sample count for one logical epoch.

        Resolution order:

        1. If ``epoch_length > 0`` was passed at construction, return
           it. Lets the caller pin a definite number of samples per
           epoch regardless of underlying corpus size.
        2. Otherwise, sum the per-shard counts read from each
           ``*.tar.index.json`` in the bucket subdirs. These are
           written by the kwcoco_dataloader writer alongside every
           closed shard, so the count is exact and cheap to compute.
        3. As a last resort (no index files yet, e.g. mid-build),
           raise informatively.

        DEIMv2's solver needs ``len()`` for:
          * ``FlatCosineLRScheduler``'s warmup/flat/cosine schedule
          * ``DataLoader.__len__`` (the IterableDataset path returns
            ``len(dataset)`` directly)
          * Per-epoch progress bars.

        Mosaic-with-cache + transforms work without ``len()``; we
        define one anyway so DEIMv2's scheduler can plan its
        warmup/flat boundaries.
        """
        if self._epoch_length > 0:
            return self._epoch_length

        import json
        total = 0
        index_files = sorted(self.shards_dpath.glob("*/*.tar.index.json"))
        if not index_files:
            # Fall back to .tar count × maxcount estimate so the
            # scheduler can still compute roughly-correct boundaries.
            tars = sorted(self.shards_dpath.glob("*/*.tar"))
            if not tars:
                raise NotImplementedError(
                    f"WebDatasetCocoDetection __len__: no .tar.index.json "
                    f"or .tar files under {self.shards_dpath}; pass "
                    "epoch_length explicitly or finish the shard build."
                )
            return len(tars) * 5000   # writer maxcount default

        for idx_fpath in index_files:
            try:
                data = json.loads(idx_fpath.read_text())
            except Exception:
                continue
            fnames = data.get("fnames") or []
            total += len(fnames)
        return total

    def load_item(self, idx):
        raise NotImplementedError(
            "WebDatasetCocoDetection is stream-only. Mosaic must be "
            "configured with use_cache=True so it doesn't reach into "
            "the dataset by index."
        )

    @property
    def categories(self):
        # Used by the eval pipeline / config consumers; emit the kit's
        # ordered list verbatim.
        return [{"id": i + 1, "name": n} for i, n in enumerate(self.category_names)]

    # ----- iteration --------------------------------------------------

    def _build_stream(self):
        from kwcoco_dataloader.readers.detection import (
            load_bucket_streams, WeightedChunkMix,
        )
        # load_bucket_streams returns a BucketStreamSet with aligned
        # streams + weights. We override per-bucket weights only when
        # the caller passed bucket_weights (rare); otherwise use the
        # footer-derived defaults.
        bucket_set = load_bucket_streams(shards_dpath=self.shards_dpath)
        if not bucket_set.streams:
            return iter(())
        if self._bucket_weights:
            weights = [
                float(self._bucket_weights.get(d.name, w))
                for d, w in zip(bucket_set.bucket_dirs, bucket_set.weights)
            ]
        else:
            weights = bucket_set.weights
        return iter(WeightedChunkMix(
            bucket_set.streams, weights,
            chunk_size=self._chunk_size,
            seed=self._epoch,
        ))

    def __iter__(self):
        from kwcoco_dataloader.readers.detection import (
            relabel_detection_sample,
        )
        n = 0
        for sample in self._build_stream():
            sample = relabel_detection_sample(sample, self.scheme)
            if not sample.target.get("annotations"):
                # All annotations were dropped by the scheme collapse —
                # this sample is effectively background. Skip so the
                # matcher doesn't see empty-target items it can't use.
                continue

            img = Image.fromarray(sample.image)
            target = self._make_target(sample, img.size)

            if self._transforms is not None:
                img, target, _ = self._transforms(img, target, self)

            yield img, target
            n += 1
            if self._epoch_length and n >= self._epoch_length:
                return

    def _make_target(self, sample, image_size):
        """Convert a kwcoco_dataloader Sample's annotations into
        DEIMv2's expected target dict shape."""
        w, h = image_size

        # Pull (x, y, w, h) boxes + already-relabeled category_id from
        # the sample. We trust relabel_detection_sample to have done
        # the source -> target mapping; what remains is xywh -> xyxy
        # + box-clamp, identical to ConvertCocoPolysToMask.
        anns = [
            a for a in sample.target.get("annotations", [])
            if int(a.get("iscrowd", 0)) == 0 and a.get("bbox")
            and a.get("category_id") is not None
        ]

        if not anns:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            xywh = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32)
            boxes_xyxy = xywh.clone()
            boxes_xyxy[:, 2:] = xywh[:, :2] + xywh[:, 2:]
            boxes_xyxy[:, 0::2].clamp_(min=0, max=w)
            boxes_xyxy[:, 1::2].clamp_(min=0, max=h)
            labels = torch.as_tensor(
                [int(a["category_id"]) for a in anns], dtype=torch.int64
            )
            area_vals = []
            for a, (x0, y0, x1, y1) in zip(anns, boxes_xyxy.tolist()):
                area_vals.append(float(a.get("area", (x1 - x0) * (y1 - y0))))
            area = torch.as_tensor(area_vals, dtype=torch.float32)
            iscrowd = torch.as_tensor(
                [int(a.get("iscrowd", 0)) for a in anns], dtype=torch.int64
            )

            keep = (boxes_xyxy[:, 3] > boxes_xyxy[:, 1]) & \
                   (boxes_xyxy[:, 2] > boxes_xyxy[:, 0])
            boxes_xyxy = boxes_xyxy[keep]
            labels = labels[keep]
            area = area[keep]
            iscrowd = iscrowd[keep]

        # image_id: prefer the writer-stamped value; fall back to a hash
        # of the WebDataset key so it's stable per sample.
        img_id = sample.target.get("image_id")
        if img_id is None:
            img_id = abs(hash(sample.key)) & 0x7FFFFFFF
        idx = sample.target.get("idx", img_id)

        target = {
            "boxes": convert_to_tv_tensor(boxes_xyxy, key="boxes",
                                          spatial_size=(h, w)),
            "labels": labels,
            "image_id": torch.tensor([int(img_id)]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(w), int(h)]),
            "idx": torch.tensor([int(idx)]),
        }
        return target

    def extra_repr(self) -> str:
        return (
            f" shards_dpath: {self.shards_dpath}\n"
            f" buckets: {[b.name for b in self._buckets]}\n"
            f" category_names: {self.category_names}\n"
            f" epoch_length: {self._epoch_length}\n"
        )
