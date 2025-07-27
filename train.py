# /train.py
import csv
import datetime
import itertools
import os
import sys
import warnings
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import json
import argparse

# --- DDP Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from source.utils import (
    set_seed,
)
from source.training import (
    train_epoch,
    evaluate_epoch,
    get_dataloaders,
    prepare_optimizer,
)
from source.models import ENIGMA

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True


def train(
    model_path,
    config_name,
    cache_path,
    output_path,
    model_name,
    subj_ids,
    batch_size,
    seed,
    num_epochs,
    ckpt_saving,
    ckpt_interval,
    max_lr,
    final_div_factor,
    pct_start,
    mse_loss_scale,
    retrieval_img_loss_scale,
    retrieval_txt_loss_scale,
    retrieval_only,
    rank,
    world_size,
):

    # Set random seeds for reproducibility, with a different seed per process
    if seed is not None:
        set_seed(seed + rank)
        
    device = torch.device(f"cuda:{rank}")
    # Refine input parameters and setup paths
    data_type = torch.float32
    subjects = [f"sub-{subj:02d}" for subj in subj_ids]

    output_path = os.path.join(output_path, model_name)
    model_path = os.path.join(model_path, model_name)

    if rank == 0:
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

    # Prepare dataloaders and multi-gpu samplers
    train_dataloader, test_dataloader, train_sampler = get_dataloaders(
        config_name=config_name,
        subjects=subjects,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )
    num_channels, num_timepoints = (
        train_dataloader.dataset.eeg_data.shape[-2],
        train_dataloader.dataset.eeg_data.shape[-1],
    )

    model = ENIGMA(
        num_channels,
        num_timepoints,
        subjects=subjects,
        embed_dim=1024,
        retrieval_only=retrieval_only,
    )
    model = model.to(device)
    # Wrap model for distributed training
    model = DDP(model, device_ids=[rank])

    optimizer, lr_scheduler = prepare_optimizer(
        model=model,
        max_lr=max_lr,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        pct_start=pct_start,
        final_div_factor=final_div_factor,
    )

    best_loss = float("inf")

    # Training / Fine Tuning Loop
    for epoch in tqdm(
        range(num_epochs), desc=f"Training loop", file=sys.stdout, disable=(rank != 0)
    ):
        # Set the epoch for the sampler to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        
        with torch.amp.autocast("cuda", dtype=data_type):
            model.train()
            train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                mse_loss_scale=mse_loss_scale,
                retrieval_img_loss_scale=retrieval_img_loss_scale,
                retrieval_txt_loss_scale=retrieval_txt_loss_scale,
                retrieval_only=retrieval_only,
                rank=rank,
                world_size=world_size,
            )
            model.eval()
            with torch.no_grad():
                # Test loop to evaluate model
                evaluate_epoch(
                    model=model,
                    dataloader=test_dataloader,
                    device=device,
                    mse_loss_scale=mse_loss_scale,
                    retrieval_img_loss_scale=retrieval_img_loss_scale,
                    retrieval_txt_loss_scale=retrieval_txt_loss_scale,
                    retrieval_only=retrieval_only,
                    rank=rank,
                    world_size=world_size,
                )

                # checkpoint saving (only on rank 0)
                if rank == 0:
                    if (
                        ckpt_saving and (epoch + 1) % ckpt_interval == 0
                    ) or epoch + 1 == num_epochs:
                        torch.save(
                            model.module.state_dict(),
                            os.path.join(model_path, "last.pth"),
                        )
        torch.cuda.empty_cache()



def main():

    parser = argparse.ArgumentParser(
        description="Train the Model with Specified Configurations."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="train_logs",
        help="Path to where model files and weights are stored.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="cache",
        help=(
            "Path to where misc. files downloaded from HuggingFace are stored."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Path to where features and reconstructions will be stored.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="Alljoined-1.6M",
        help=(
            "Name of the config to load for the dataset (looks in configs"
            " directory)."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ENIGMA",
        help="Name of model, used for checkpoint saving",
    )
    parser.add_argument(
        "--subj_ids",
        nargs="+",
        type=int,
        default=[1],
        help="Space-separated list of subject IDs to train on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Per-GPU batch size for model training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generators. Default value is random",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=150,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=3e-4 * 3.8, # changed from 3e-4 to 5e-4
        help="Maximum learning rate used in the schedule for model training.",
    )
    parser.add_argument(
        "--final_div_factor",
        type=float,
        default=10000,
        help="Final division factor for the OneCycleLR scheduler.",
    ),
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.22,
        help="Percentage of total steps to reach the maximum learning rate.",
    ),
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=5,
        help="How often to save backup checkpoints.",
    )
    parser.add_argument(
        "--ckpt_saving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save checkpoints.",
    )
    parser.add_argument(
        "--mse_loss_scale", type=float, default=1.0, help="Scale for MSE loss."
    )
    parser.add_argument(
        "--retrieval_img_loss_scale",
        type=float,
        default=0.5,
        help="Scale for retrieval image loss.",
    )
    parser.add_argument(
        "--retrieval_txt_loss_scale",
        type=float,
        default=0.05,
        help="Scale for retrieval text loss.",
    )
    parser.add_argument(
        "--retrieval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only perform retrieval grid creation, no reconstructions.",
    )
    args = parser.parse_args()
    
    # --- DDP Setup ---
    # To run, use `torchrun --nproc_per_node=8 train.py [ARGS...]`
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Print arguments to the log for reference (only on the main process)
    if rank == 0:
        print(f"Starting DDP with {world_size} GPUs.")
        print("train.py ARGUMENTS:\n-----------------------")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print("-----------------------")

    # Partial function to explicitly pass in arguments to the train function
    train(
        model_path=args.model_path,
        config_name=args.config_name,
        cache_path=args.cache_path,
        output_path=args.output_path,
        model_name=args.model_name,
        subj_ids=args.subj_ids,
        batch_size=args.batch_size,
        seed=args.seed,
        num_epochs=args.num_epochs,
        ckpt_saving=args.ckpt_saving,
        ckpt_interval=args.ckpt_interval,
        max_lr=args.max_lr,
        final_div_factor=args.final_div_factor,
        pct_start=args.pct_start,
        mse_loss_scale=args.mse_loss_scale,
        retrieval_img_loss_scale=args.retrieval_img_loss_scale,
        retrieval_txt_loss_scale=args.retrieval_txt_loss_scale,
        retrieval_only=args.retrieval_only,
        rank=rank,
        world_size=world_size
    )
    
    # Clean up DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    main()