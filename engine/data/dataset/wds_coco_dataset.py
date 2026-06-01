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
import os
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
        # WebDatasetStream knobs. Defaults bound the per-worker memory
        # footprint: shuffle_buffer=128 decoded samples × 4 workers ×
        # ~3 MB/tile = ~1.5 GB host RAM total, vs the upstream
        # shuffle_buffer=1024 default which can balloon to 12 GB and
        # silently OOM-kill workers under tight cgroups (gen002 2552
        # 2026-05-30: 3 of 4 workers died around iter 3000-3500 with
        # no in-process log; surviving worker can't keep the main
        # consumer fed, training appears to "hang"). Override via
        # the YAML config's stream_kwargs field if more shuffle
        # randomness is needed.
        stream_kwargs: Optional[dict] = None,
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

        # Bucket weights are merged from two sources, env-var first so
        # YAML configs win when both are set. JSON in the env var maps
        # bucket NAME (e.g. "dominant_raw_class_EQ_P", but partial
        # substring matches also work) to a float weight; missing
        # buckets get the default 1.0.
        import json as _json
        env_weights = os.environ.get("KCD_WDS_BUCKET_WEIGHTS_JSON", "").strip()
        if env_weights:
            try:
                env_dict = _json.loads(env_weights)
                if not isinstance(env_dict, dict):
                    raise TypeError("must be a JSON object")
                bucket_weights = {**env_dict, **(bucket_weights or {})}
            except Exception as e:
                import sys as _sys
                print(
                    f"[wds_coco_dataset] WARNING: failed to parse "
                    f"KCD_WDS_BUCKET_WEIGHTS_JSON: {e}. "
                    f"Got: {env_weights!r}",
                    file=_sys.stderr,
                )
        self._bucket_weights = bucket_weights or {}
        self._chunk_size = int(chunk_size)
        self._num_workers_hint = int(num_workers_hint)
        # Keep WebDatasetStream's shuffle buffers small enough that
        # 4 workers × per-worker buffer doesn't OOM the host. Users
        # can pass {"shuffle_buffer": ..., "shardshuffle": ...} to
        # override.
        self._stream_kwargs = {"shuffle_buffer": 128, "shardshuffle": 8}
        if stream_kwargs:
            self._stream_kwargs.update(stream_kwargs)
        self._epoch_length = int(epoch_length)
        self._epoch = 0
        # Memoize __len__. DEIMv2's MetricLogger.log_every() calls
        # len(loader) every iteration (loader.__len__ → dataset.__len__),
        # and ours walks every *.tar.index.json + json-decodes them on
        # each call. With ~hundreds of shards that's seconds per iter
        # spent re-parsing the same data. Cache after the first compute.
        self._len_cached: Optional[int] = None

        # Log the effective semantic config so reproducing a run from
        # the slurm log doesn't require inspecting env vars or env
        # dumps. KCD_WDS_SKIP_EMPTY + KCD_WDS_BUCKET_WEIGHTS_JSON both
        # change training-set composition; the journal entry for a
        # run should record what these were. Each worker prints once
        # at __init__; the kit submit scripts also echo their settings
        # before launch.
        _skip_empty_now = os.environ.get("KCD_WDS_SKIP_EMPTY", "0") == "1"
        try:
            import sys as _sys
            print(
                f"[wds_coco_dataset] pid={os.getpid()} "
                f"skip_empty={_skip_empty_now} "
                f"bucket_weights={self._bucket_weights or '<uniform>'} "
                f"stream_kwargs={self._stream_kwargs} "
                f"epoch_length={self._epoch_length} "
                f"shards_dpath={self.shards_dpath}",
                file=_sys.stderr, flush=True,
            )
        except Exception:
            pass

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

        if self._len_cached is not None:
            return self._len_cached

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
            # Each sample contributes multiple entries to ``fnames``
            # (one per sidecar: ``<key>.jpg`` + ``<key>.json`` today,
            # potentially more later). Count primaries — image files —
            # so the total equals the sample count regardless of how
            # many sidecars the writer emits.
            total += sum(
                1 for f in fnames
                if isinstance(f, str) and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
        self._len_cached = total
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
        bucket_set = load_bucket_streams(
            shards_dpath=self.shards_dpath,
            stream_kwargs=self._stream_kwargs,
        )
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
        import os
        import signal
        import threading
        import time
        from kwcoco_dataloader.readers.detection import (
            relabel_detection_sample,
        )
        # When epoch_length is pinned (the right configuration for any
        # DDP run, see below), rebuild the underlying stream whenever it
        # exhausts so the iterator yields exactly epoch_length samples
        # regardless of how unevenly the bucket shards split across
        # ranks/workers. Without this auto-cycle, ranks with fewer
        # shards run out of data before epoch_length, then later DDP
        # collectives find one rank waiting and one rank gone:
        #   RuntimeError: Detected mismatch between collectives on
        #   ranks. Rank 0 ... BROADCAST ... Rank 1 ... REDUCE
        # (host-gpu2x-wds matrix scenario 2026-05-30 reproduced
        # this on yardrat.) With epoch_length=0 (no pin), we keep
        # the original drain-once semantics — single-rank training
        # and the kit's existing gen001 path rely on it.
        def _cycle():
            while True:
                stream = self._build_stream()
                yielded = False
                for sample in stream:
                    yielded = True
                    yield sample
                if not yielded:
                    return  # empty corpus — don't spin forever
                if self._epoch_length <= 0:
                    return  # caller wants one pass only

        # Watchdog: PIL.Image.load() holds the GIL through the entire
        # decode, so signal-based or thread-based timeouts can't
        # interrupt it. We've observed PIL.load deadlock on rare valid
        # JPEGs (kit zombie job 2553, 2026-05-30 → 2026-06-01: 5-day
        # silent hang on a clean shard, root cause buried in webdataset
        # autodecode + PIL interaction). The only reliable recovery is
        # to kill the worker; PyTorch's DataLoader respawns it
        # automatically. Set timeout to 0 to disable.
        sample_timeout_s = float(os.environ.get(
            "KCD_WDS_SAMPLE_TIMEOUT_S", "120"))
        if sample_timeout_s > 0:
            last_progress = [time.monotonic()]
            stop_watchdog = threading.Event()

            def _watchdog():
                while not stop_watchdog.wait(5.0):
                    elapsed = time.monotonic() - last_progress[0]
                    if elapsed > sample_timeout_s:
                        # Last-ditch logging before suicide; DataLoader
                        # will respawn this worker and training continues.
                        try:
                            import sys
                            print(
                                f"[wds_coco_dataset] worker pid={os.getpid()} "
                                f"stalled {elapsed:.0f}s in sample decode "
                                f"(KCD_WDS_SAMPLE_TIMEOUT_S={sample_timeout_s}); "
                                f"SIGKILL-ing self so DataLoader respawns. "
                                f"See journal 2026-06-01_*.md.",
                                file=sys.stderr, flush=True,
                            )
                        except Exception:
                            pass
                        os.kill(os.getpid(), signal.SIGKILL)

            t = threading.Thread(target=_watchdog, daemon=True)
            t.start()
        else:
            last_progress = None
            stop_watchdog = None

        # Whether to drop samples whose annotations are entirely
        # empty (or got dropped by the scheme collapse). Old default
        # was True — silently filter empties so the matcher never
        # sees "no target" items. That turned out to be wrong: for
        # detection-AP training, NEGATIVE TILES are valuable signal,
        # and silently filtering them caused gen002 single_sealion
        # to over-fit to a small positive pool (each positive tile
        # seen ~38× per epoch vs v5's ~1×) → kit AP 0.024 vs v5's
        # 0.177 = 7.4× regression (journal 2026-06-01). Default is
        # now False (keep empties); set KCD_WDS_SKIP_EMPTY=1 to opt
        # back into the old behavior.
        skip_empty = os.environ.get("KCD_WDS_SKIP_EMPTY", "0") == "1"

        n = 0
        for sample in _cycle():
            if last_progress is not None:
                last_progress[0] = time.monotonic()
            sample = relabel_detection_sample(sample, self.scheme)
            if skip_empty and not sample.target.get("annotations"):
                # Legacy behavior — skip background-only tiles. Kept
                # behind KCD_WDS_SKIP_EMPTY=1 in case a downstream
                # config relies on the old contract; new runs should
                # leave it off and let empty tiles flow through as
                # negative samples.
                continue
            # Ensure annotations key exists even when empty (the
            # collate / matcher path expects a list, not missing).
            if "annotations" not in sample.target:
                sample.target["annotations"] = []

            img = Image.fromarray(sample.image)
            target = self._make_target(sample, img.size)

            if self._transforms is not None:
                img, target, _ = self._transforms(img, target, self)

            yield img, target
            n += 1
            if self._epoch_length and n >= self._epoch_length:
                if stop_watchdog is not None:
                    stop_watchdog.set()
                return
        if stop_watchdog is not None:
            stop_watchdog.set()

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
