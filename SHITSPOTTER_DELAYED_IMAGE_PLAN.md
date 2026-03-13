# ShitSpotter Delayed-Image Plan

This note captures a possible future optimization for the local ShitSpotter
integration of DEIMv2.

## Context

The current DEIMv2 training path loads full-resolution images with PIL in
`engine/data/dataset/coco_dataset.py` and then applies geometric transforms
later in the injected transform stack. For ShitSpotter phone images, that means
we often decode large originals only to resize them down to a much smaller
working resolution.

This is functional, but it is likely leaving throughput on the table.

## Why this is not urgent right now

Observed utilization on the current training box was roughly:

- GPU: about 58%
- CPU: about 48%

That suggests the loader / augmentation path is not ideal, but it is not so
bad that it obviously blocks end-to-end progress. The immediate priority is
still getting a reliable training / inference / evaluation path working.

In other words: this optimization looks worthwhile, but not critical enough to
interrupt the current integration work.

## Current bottleneck shape

The main issue is that image loading and resize choice are decoupled:

- DEIMv2 loads the original image first
- augmentation happens afterward through dependency-injected transforms
- the final resize target is only known inside the transform stack

That makes it hard to exploit overviews or delayed-image style pre-scaling at
load time.

## Desired direction

Use delayed-image / kwcoco-style loading logic so the loader can request an
overview near the target working resolution instead of always decoding the full
image.

Representative idea:

```python
import delayed_image

delayed = delayed_image.DelayedLoad(image_fpath)
image = delayed.prepare().resize((640, 460)).finalize()
```

In practice the exact target size should come from loader-side policy rather
than being hard-coded in the example.

## Scope recommendation

Do not try to preserve the full current COCO-style geometric augmentation stack
while introducing delayed-image loading.

Instead, the pragmatic ShitSpotter path is:

1. Ignore Mosaic for this effort.
2. Move the deterministic resize decision into the loader path.
3. Keep lightweight non-geometric augmentation after load.
4. Handle any remaining geometric box/polygon transforms explicitly in the
   loader or in a simplified transform stage.

## Suggested first implementation

Target the DEIMv2 dataset loader in:

- `engine/data/dataset/coco_dataset.py`

Likely patch shape:

1. Add an optional loader-side configuration for ShitSpotter, for example:
   - `preload_mode: none | val | train_fast`
   - `preload_size`
   - `preload_max_dim`
2. Replace the PIL-only image load path with delayed-image based loading when
   enabled.
3. Apply the loader-side scale to annotations before the torchvision-v2
   transforms see them.
4. Return a normal image plus transformed targets so the rest of the DEIMv2
   stack still works.

## Annotation handling

If resize moves into the loader, annotation geometry must move with it.

At minimum:

- scale bounding boxes
- scale polygons / segmentations if present
- preserve `orig_size` metadata separately from the resized working size

ShitSpotter uses polygons heavily, so this part must be done carefully.

## Recommended rollout order

1. Validation / inference loader first
   - easiest path because augmentation is simple
   - good place to verify overview-based loading behavior
2. Add a `train_fast` recipe for ShitSpotter
   - no Mosaic
   - simplified augmentations
   - delayed-image resize in the loader
3. Compare throughput and convergence against the current path
4. Only then consider deeper changes to the remaining geometric augmentations

## Success criteria for revisiting this

This is worth reviving if one or more of the following become true:

- GPU utilization remains low enough that the loader is clearly the bottleneck
- training wall-clock time becomes the dominant pain point
- a simplified ShitSpotter-specific augmentation recipe is preferred anyway
- we want a cleaner kwcoco-native training story across model backends

## Non-goal for now

This note is not a commitment to rewrite DEIMv2 into a full kwcoco-native
trainer. The narrower goal is just to exploit delayed-image style loading to
avoid wasteful full-resolution decodes where possible.
