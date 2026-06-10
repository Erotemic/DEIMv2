"""
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import time
import random
import numpy as np
import atexit

import torch
import torch.nn as nn
import torch.distributed
import torch.backends.cudnn

from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data import DistributedSampler
# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader


def setup_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, ):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        # On CPU-only hosts (no CUDA available — CI, the wds_e2e_demo
        # on a GPU-less box) NCCL isn't usable. Fall back to gloo so
        # dist init succeeds; the loss is intra-node bandwidth but
        # that doesn't matter for a single-process smoke run.
        if torch.cuda.is_available():
            torch.distributed.init_process_group(init_method='env://')
        else:
            torch.distributed.init_process_group(backend='gloo',
                                                 init_method='env://')
        torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            torch.cuda.empty_cache()
        enabled_dist = True
        if get_rank() == print_rank:
            print('Initialized distributed mode...')

    except Exception:
        enabled_dist = False
        print('Not init distributed mode.')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)

    return enabled_dist


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process,
    and prefixes every emitted line with an ISO-8601 timestamp so a
    long-running training run is easy to correlate against slurm
    timestamps, NCCL traces, and external events.

    Timestamping can be disabled per-call by passing ``timestamp=False``
    (e.g. when emitting a multi-line block where the prefix would be
    cosmetically wrong). The pre-existing ``force=True`` kwarg is
    preserved.
    """
    import builtins as __builtin__
    import datetime as _dt

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        timestamp = kwargs.pop('timestamp', True)
        if is_main or force:
            if timestamp:
                ts = _dt.datetime.now().strftime('[%Y-%m-%dT%H:%M:%S]')
                builtin_print(ts, *args, **kwargs)
            else:
                builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save a checkpoint from rank 0, then synchronize all ranks so
    that file IO is durable before any rank proceeds.

    Without the trailing barrier, downstream code that immediately
    reads the just-written file (e.g. det_solver.fit()'s
    `load_resume_state(best_stg1.pth)` at the start of every epoch
    when `stop_epoch` is set) races: rank 0 is still writing the
    checkpoint while rank 1+ already calls torch.load on a partial
    file → silent exception → rank exits → the next collective on
    the surviving rank fingerprints-mismatches the absent rank and
    DDP dies with:

      RuntimeError: Detected mismatch between collectives on ranks.
      Rank 0 ... BROADCAST ... Rank 1 ... GATHER seq=0

    (yardrat 2026-05-30: host-gpu2x-jpeg + host-gpu2x-wds both
    reproduced this between epoch 0 and epoch 1.)
    """
    if is_main_process():
        torch.save(*args, **kwargs)
    if is_dist_available_and_initialized():
        torch.distributed.barrier()



def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    if is_dist_available_and_initialized():
        rank = get_rank()
        # SyncBatchNorm is CUDA-only — DDP rejects it on CPU modules.
        # Silently skip the conversion in CPU-only runs (the
        # wds_e2e_demo on a GPU-less host) since plain BN is fine for
        # a single-process smoke run.
        effective_sync_bn = sync_bn and torch.cuda.is_available()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if effective_sync_bn else model
        # DDP's device_ids/output_device only accept CUDA module
        # parameters. When CUDA is unavailable (CPU-only environments
        # — CI, the wds_e2e_demo on a GPU-less host), the model stays
        # on CPU and `DDP(..., device_ids=[rank])` raises
        # "module parameters {device(type='cpu')}". For DDP on CPU,
        # passing `device_ids=None, output_device=None` is the
        # supported form (the gloo backend wraps CPU tensors). For
        # local single-process runs without dist init we already
        # skip this branch.
        cuda_ready = torch.cuda.is_available()
        ddp_kwargs = (
            dict(device_ids=[rank], output_device=rank)
            if cuda_ready
            else dict(device_ids=None, output_device=None)
        )
        if dist_mode == 'dp':
            if not cuda_ready:
                # nn.DataParallel requires CUDA. Skip wrap entirely.
                pass
            else:
                model = DP(model, device_ids=[rank], output_device=rank)
        elif dist_mode == 'ddp':
            model = DDP(model, find_unused_parameters=find_unused_parameters,
                        **ddp_kwargs)
        else:
            raise AttributeError('')

    if compile:
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        # IterableDataset doesn't define len() and can't be wrapped
        # in a DistributedSampler (which calls len(dataset) in its
        # __init__). The dataset is responsible for its own per-rank
        # + per-worker splitting — WebDataset handles this via
        # split_by_node / split_by_worker inside __iter__. Skip the
        # wrap; the DataLoader as-is is correct.
        from torch.utils.data import IterableDataset
        if isinstance(loader.dataset, IterableDataset):
            return loader
        sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory,
                            num_workers=loader.num_workers)
    return loader


def rebuild_loader_with_sampler(loader, sampler):
    """Rebuild a DataLoader around an explicit, already rank-aware sampler.

    Unlike warp_loader (dist-only, hardcodes DistributedSampler) this is
    used in BOTH dist and non-dist runs — the given sampler owns the
    per-rank split. Added for kwcoco_detector_kit's optional
    dataloader-level balanced sampling (kcd_sample_weights_fpath).
    """
    return DataLoader(loader.dataset,
                      loader.batch_size,
                      sampler=sampler,
                      drop_last=loader.drop_last,
                      collate_fn=loader.collate_fn,
                      pin_memory=loader.pin_memory,
                      num_workers=loader.num_workers)



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            values.append(data[k])

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()



def setup_seed(seed: int, deterministic=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # memory will be large when setting deterministic to True
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


# for torch.compile
def check_compile():
    import torch
    import warnings
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model
