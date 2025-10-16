import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yacs.config import CfgNode as CN

from burn_in import run_bi_step
from domain_adapt import run_da_step
from utils import setup


def init_distributed_mode():
    """Initialize distributed training mode if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # For SLURM cluster
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    
    print(f'| distributed init (rank {rank}): local_rank={local_rank}, world_size={world_size}', flush=True)
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    
    # Initialize distributed training
    distributed, rank, world_size, local_rank = init_distributed_mode()
    
    # Only print on rank 0
    if rank == 0:
        print("Experiment name:", cfg.exp_tags)
    
    exp_save_dir = setup(cfg) if rank == 0 else None
    
    # Broadcast exp_save_dir to all ranks
    if distributed:
        exp_save_dir_list = [exp_save_dir] if rank == 0 else [None]
        dist.broadcast_object_list(exp_save_dir_list, src=0)
        exp_save_dir = exp_save_dir_list[0]
    
    if rank == 0:
        print("Experiment save directory:", exp_save_dir)
    
    # Run burn-in step
    best_ckpt = run_bi_step(cfg, exp_save_dir=exp_save_dir)
    
    # Broadcast checkpoint path to all ranks
    if distributed:
        ckpt_list = [best_ckpt] if rank == 0 else [None]
        dist.broadcast_object_list(ckpt_list, src=0)
        best_ckpt = ckpt_list[0]
    
    if rank == 0:
        print("Loading best checkpoint from burn-in step:", best_ckpt)
    
    # Run domain adaptation step
    run_da_step(cfg, exp_save_dir=exp_save_dir, best_bi_ckpt=best_ckpt)
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
