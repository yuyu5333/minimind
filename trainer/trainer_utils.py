"""
训练工具函数集合
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM
from typing import Optional

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
        MixedPrecision,
        CPUOffload,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as _size_based_auto_wrap_policy
    HAS_FSDP = True
except Exception:
    HAS_FSDP = False


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        state_dict = get_state_dict_for_saving(model)
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    try:
                        import torch.nn as nn
                        if isinstance(value, nn.Module):
                            resume_data[key] = get_state_dict_for_saving(value)
                        else:
                            resume_data[key] = value.state_dict()
                    except Exception:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def wrap_model_for_distributed(model: torch.nn.Module, dist_type: str = 'ddp', local_rank: int = 0,
                               dtype: str = 'bfloat16', auto_wrap_threshold: int = 10000000, optimizer: any = None, lr: float = 1e-4):
    if not dist.is_initialized():
        return model
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    if dist_type == 'fsdp' and HAS_FSDP:
        mp_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
        mixed_precision = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        def auto_wrap_policy(module, recurse, nonwrapped_numel):
            return _size_based_auto_wrap_policy(module, recurse, nonwrapped_numel, min_num_params=int(auto_wrap_threshold))
        fsdp_model = FSDP(
            model,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else None,
            use_orig_params=True,
        )
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=lr)
        return fsdp_model, optimizer
    else:
        from torch.nn.parallel import DistributedDataParallel
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
        return ddp_model, optimizer


def is_fsdp_model(model: torch.nn.Module) -> bool:
    return HAS_FSDP and isinstance(model, FSDP)


def get_state_dict_for_saving(model: torch.nn.Module) -> dict:
    if is_fsdp_model(model):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_sd = model.state_dict()
        return full_sd
    from torch.nn.parallel import DistributedDataParallel
    return model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()


def safe_load_state_dict(model: torch.nn.Module, state_dict: dict, strict: bool = False):
    if is_fsdp_model(model):
        return model.module.load_state_dict(state_dict, strict=strict)
    from torch.nn.parallel import DistributedDataParallel
    if isinstance(model, DistributedDataParallel):
        return model.module.load_state_dict(state_dict, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)

