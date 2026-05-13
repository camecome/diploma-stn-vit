# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import time
import math
import random
import numpy as np

from datetime import timedelta

import torch

from tqdm import tqdm

from torch.amp import autocast
from torch.cuda.amp import GradScaler

from torch.nn.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from pathlib import Path

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_last_layers(args, model):
    logger.info(f"Checkpoint path:             {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)

    trainable_state_dict = checkpoint["model_state_dict"]
    model.transformer.encoder.layer[-1].load_state_dict(trainable_state_dict["last_transformer_block"])
    model.head.load_state_dict(trainable_state_dict["classifier_head"])

    # freeze everything but the last transformer layer
    # and classification head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer.encoder.layer[-1].parameters():
        param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True


def load_additional_info(args, optimizer, scheduler):
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)

    start_epoch = checkpoint.get("epoch", 0)
    if start_epoch == args.epoch_num:
        raise ValueError("Checkpoint already reached the target number of epochs. Nothing to resume.")

    best_acc = checkpoint.get("accuracy", 0)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Start epoch:                 {start_epoch}")
    logger.info(f"Best accuracy:               {best_acc:.5f}")

    return start_epoch, best_acc


def save_model(args, model, epoch, optimizer, scheduler, accuracy):
    logger.info("***** Start saving checkpoint *****")
    model_to_save = model.module if hasattr(model, "module") else model

    trainable_state_dict = {
        "last_transformer_block": model_to_save.transformer.encoder.layer[-1].state_dict(),
        "classifier_head": model_to_save.head.state_dict(),
    }

    checkpoint = {
        "model_state_dict": trainable_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
    }

    lr_str = f"{args.learning_rate:.3f}".replace(".", "_")
    checkpoint_path = Path(args.target_dir) / f"model_lr_{lr_str}_epoch_{epoch}.pth"

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint:            {checkpoint_path}")

    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024**2)
    logger.info(f"Checkpoint size:             {checkpoint_size_mb:.2f} MB")
    logger.info("***** Checkpoint saved successfully *****")


def save_val_data(args, epoch, logits, labels):
    logger.info("***** Start saving val data *****")
    lr_str = f"{args.learning_rate:.3f}".replace(".", "_")
    val_data_path = Path(args.target_dir) / f"val_data_lr_{lr_str}_epoch_{epoch}.npz"

    np.savez_compressed(
        val_data_path,
        logits=logits.astype(np.float16),
        labels=labels.astype(np.int16),
    )

    logger.info(f"Saved validation data:       {val_data_path}")
    metrics_size_mb = val_data_path.stat().st_size / (1024**2)
    logger.info(f"Val data size:               {metrics_size_mb:.2f} MB")
    logger.info("***** Val data saved successfully *****")


def reinitialize_last_block_and_head(model):
    last_block = model.transformer.encoder.layer[-1]

    for module in last_block.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=1e-6)

        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)

    # freeze everything but the last transformer layer
    # and classification head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer.encoder.layer[-1].parameters():
        param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True


def setup(args):
    model = VisionTransformer(CONFIGS[args.model_type], args.img_size, zero_head=True, num_classes=1000)

    if not args.pretrained_path:
        raise ValueError("'pretrained_path' must be provided.")

    state_dict = torch.load(args.pretrained_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    if args.checkpoint_path:
        load_last_layers(args, model)
    else:
        reinitialize_last_block_and_head(model)

    model.to(args.device)
    logger.info(f"Total parameters:            {sum(p.numel() for p in model.parameters()) / 1_000_000:.1f}M")
    logger.info(
        f"Total trainable parameters:  {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.1f}M"
    )
    logger.info(f"Out features:                {model.head.out_features}")

    return args, model


def valid(args, model, val_loader, opt_step, scheduler):
    eval_losses = AverageMeter()

    logger.info(f"***** Running validation after optimization step {opt_step} *****")
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"Current LR:                  {current_lr:.5f}")

    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    epoch_iterator = tqdm(
        val_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0],
    )

    loss_fct = torch.nn.CrossEntropyLoss()
    for _, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            with autocast("cuda", enabled=args.fp16):
                logits = model(x)[0]
                eval_loss = loss_fct(logits, y)

            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        all_logits.append(logits.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = simple_accuracy(all_preds, all_labels)

    logger.info("\n")
    logger.info("***** Validation Results *****")
    logger.info(f"Valid Loss:                  {eval_losses.avg:.5f}")
    logger.info(f"Valid Accuracy:              {accuracy:.5f}")

    return accuracy, all_logits, all_labels


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.target_dir, exist_ok=True)

    args.effective_train_batch_size = args.physical_train_batch_size * args.gradient_accumulation_steps

    train_loader, val_loader = get_loader(args)
    logger.info(f"Train images:                {len(train_loader.dataset)}") # last batch is dropped
    logger.info(f"Validation images:           {len(val_loader.dataset)}")
    total_opt_step = (len(train_loader) // args.gradient_accumulation_steps) * args.epoch_num
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
        trainable_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
    )

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_opt_step)
    else:
        raise ValueError("'decay_type' must be cosine")

    best_acc = 0
    opt_step = 0
    start_epoch = 0

    logger.info(f"Total optimization steps:    {total_opt_step}")
    if args.checkpoint_path:
        start_epoch, best_acc = load_additional_info(args, optimizer, scheduler)
        opt_step = start_epoch * (len(train_loader) // args.gradient_accumulation_steps)
        remaining_opt_steps = (args.epoch_num - start_epoch) * (len(train_loader) // args.gradient_accumulation_steps)
        logger.info(f"Remaining opt steps:         {remaining_opt_steps}")

    logger.info(f"Num of validation steps:     {len(val_loader)}")
    logger.info("=" * 80)
    logger.info("\n")

    scaler = GradScaler(enabled=args.fp16)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info(f"***** Running training *****")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epoch_num):
        logger.info(f"***** Epoch [{epoch + 1} / {args.epoch_num}] started *****")
        epoch_start_time = time.time()
        model.train()

        epoch_iterator = tqdm(
            train_loader,
            disable=args.local_rank not in [-1, 0],
        )

        for batch_step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            with autocast("cuda", enabled=args.fp16):
                loss = model(x, y)
                loss /= args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accumulation_step = bool((batch_step + 1) % args.gradient_accumulation_steps == 0)
            is_last_batch = bool(batch_step == (len(train_loader) - 1))
            if is_accumulation_step:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                epoch_iterator.set_description(
                    f"Training ({opt_step} / {total_opt_step} steps) (loss={losses.val:.5f})"
                )

            elif is_last_batch:
                optimizer.zero_grad(set_to_none=True)

        if args.local_rank in [-1, 0]:
            accuracy, logits, labels = valid(args, model, val_loader, opt_step, scheduler)
            if accuracy > best_acc:
                logger.info(f"New best accuracy:           {best_acc:.5f} -> {accuracy:.5f}")
                best_acc = accuracy
            save_model(args, model, epoch + 1, optimizer, scheduler, accuracy)
            save_val_data(args, epoch + 1, logits, labels)

            model.train()

        logger.info(f"***** Epoch [{epoch + 1} / {args.epoch_num}] finished *****")
        logger.info(f"Epoch time:                  {(time.time() - epoch_start_time):.2f} sec")
        logger.info(f"Best accuracy:               {best_acc:.5f}")
        logger.info("\n")

        losses.reset()

    logger.info(f"Best Accuracy: {best_acc:.5f}")
    logger.info("***** End training! *****")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")

    # input params
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/workspace/shared/ViT-B_16.pth",
        help="Where to search for pretrained ViT models.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/workspace/shared/target_dir",
        help="Directory to store validation data.",
    )
    parser.add_argument("--dataset_path", default="/workspace/dev_imagenet1k", help="Path to dataset folder.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training.",
    )

    # main training params
    parser.add_argument(
        "--physical_train_batch_size",
        default=256,
        type=int,
        help="Total batch size for training. Effective batch size = physical_train_batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--eval_batch_size", default=1024, type=int, help="Total batch size for eval.")
    parser.add_argument("--epoch_num", default=4, type=int, help="Total number of epochs to train the model.")
    parser.add_argument("--learning_rate", default=0.06, type=float, help="The initial learning rate for SGD.")

    # less important hyperparameters that are kept fixed
    parser.add_argument(
        "--model_type",
        default="ViT-B_16",
        help="Which variant to use.",
    )
    parser.add_argument("--img_size", default=384, type=int, help="Resolution size.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument(
        "--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate."
    )
    parser.add_argument(
        "--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s"
        % (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    )

    logger.info("\n")
    logger.info("{}".format(CONFIGS[args.model_type]))
    logger.info(f"Training parameters %s", args)
    logger.info(3 * "\n")

    logger.info("=" * 80)
    logger.info("***** Main info *****")
    logger.info(f"Physical train batch size:   {args.physical_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective train batch size:  {args.physical_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Eval batch size:             {args.eval_batch_size}")
    logger.info(f"Number of epochs             {args.epoch_num}")
    logger.info(f"LR:                          {args.learning_rate}")

    set_seed(args)
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()
