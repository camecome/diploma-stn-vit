# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import time
import random
import numpy as np

import torch

from tqdm import tqdm

from torch.amp import autocast, GradScaler


from models.modeling import VisionTransformer, CONFIGS
from models.stn.stn_vit import SpatialTransformerViT
from models.stn.stn_vit_loss import STNViTLoss
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

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


def get_lr_str(learning_rate):
    return f"{learning_rate:.10f}".rstrip("0").rstrip(".").replace(".", "_")


def load_vit(args, base_vit):
    if not args.vit_common_layers_checkpoint:
        raise ValueError("'vit_common_layers_checkpoint' must be provided.")

    if not args.vit_last_layers_checkpoint:
        raise ValueError("'vit_last_layers_checkpoint' must be provided.")

    logger.info("***** Loading common ViT layers *****")
    logger.info(f"Common layers checkpoint path:   {args.vit_common_layers_checkpoint}")
    logger.info(f"Last layers checkpoint path:     {args.vit_last_layers_checkpoint}")

    common_layers = torch.load(args.vit_common_layers_checkpoint, map_location="cpu")
    base_vit.load_state_dict(common_layers, strict=True)

    last_layers_checkpoint = torch.load(args.vit_last_layers_checkpoint, map_location="cpu", weights_only=False)
    last_layers = last_layers_checkpoint["model_state_dict"]
    base_vit.transformer.encoder.layer[-1].load_state_dict(last_layers["last_transformer_block"])
    base_vit.head.load_state_dict(last_layers["classifier_head"])
    logger.info("***** Common ViT layers succesfully downloaded *****")


def load_stn_vit_checkpoint(args, stn_vit):
    logger.info("***** Loading STN checkpoint *****")
    logger.info(f"STN-ViT checkpoint path:         {args.stn_vit_checkpoint}")
    checkpoint = torch.load(args.stn_vit_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    stn_vit.last_layers.load_state_dict(state_dict["last_layers"])
    stn_vit.norms.load_state_dict(state_dict["norms"])
    stn_vit.heads.load_state_dict(state_dict["heads"])
    stn_vit.loc_net.load_state_dict(state_dict["loc_net"])
    logger.info("***** STN checkpoint succesfully downloaded *****")


def freeze_stn_vit_common_layers(stn_vit):
    for param in stn_vit.parameters():
        param.requires_grad = False

    trainable_modules = [
        stn_vit.last_layers,
        stn_vit.norms,
        stn_vit.heads,
        stn_vit.loc_net,
    ]

    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True


def setup(args):
    base_vit = VisionTransformer(CONFIGS[args.model_type], num_classes=1000, img_size=args.img_size, zero_head=True)

    # load common layers
    load_vit(args, base_vit)
    stn_vit = SpatialTransformerViT(base_vit=base_vit, max_rotation_degrees=args.max_rotation_degrees)

    if args.stn_vit_checkpoint:
        load_stn_vit_checkpoint(args, stn_vit)

    freeze_stn_vit_common_layers(stn_vit)
    stn_vit.to(args.device)
    logger.info(f"Total STN-ViT parameters:        {sum(p.numel() for p in stn_vit.parameters()) / 1_000_000:.1f}M")
    logger.info(
        f"Trainable STN-ViT parameters:    {sum(p.numel() for p in stn_vit.parameters() if p.requires_grad) / 1_000_000:.1f}M"
    )
    logger.info(f"Max rotation degrees:            {args.max_rotation_degrees}")

    return args, stn_vit


def save_model(args, model, epoch, best_epoch, optimizer, scheduler, accuracy):
    logger.info("***** Start saving STN checkpoint *****")

    model = model.module if hasattr(model, "module") else model

    stn_vit_trainable_state_dict = {
        "last_layers": model.last_layers.state_dict(),
        "norms": model.norms.state_dict(),
        "heads": model.heads.state_dict(),
        "loc_net": model.loc_net.state_dict(),
    }

    checkpoint = {
        "model_state_dict": stn_vit_trainable_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "best_epoch": best_epoch,
    }

    lr_str = get_lr_str(args.learning_rate)
    checkpoint_path = Path(args.target_dir) / args.target_subdir / f"stn_vit_{lr_str}_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved checkpoint:                {checkpoint_path}")
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024**2)
    logger.info(f"Checkpoint size:                 {checkpoint_size_mb:.2f} MB")
    logger.info("***** STN checkpoint saved successfully *****")


def load_stn_vit_train_state(args, optimizer, scheduler):
    checkpoint = torch.load(args.stn_vit_checkpoint, map_location="cpu", weights_only=False)

    start_epoch = checkpoint.get("epoch", 0) + 1
    if start_epoch > args.epoch_num:
        raise ValueError("Checkpoint already reached the target number of epochs. Nothing to resume.")

    best_acc = checkpoint.get("accuracy", 0)
    best_epoch = checkpoint.get("best_epoch", start_epoch - 1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"Start epoch:                     {start_epoch}")
    logger.info(f"Best accuracy:                   {best_acc:.5f}")
    logger.info(f"Best epoch:                      {best_epoch}")
    logger.info(f"Current LR:                      {current_lr:.5f}")

    return start_epoch, best_acc, best_epoch


def valid(args, model, val_loader, opt_step, scheduler):
    eval_losses = AverageMeter()

    logger.info(f"***** Running validation after optimization step {opt_step} *****")
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"Current LR:                      {current_lr:.5f}")

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
            with autocast("cuda", enabled=args.fp16 and args.device.type == "cuda"):
                logits, _, _ = model(x)
                eval_loss = loss_fct(logits, y)

            eval_losses.update(eval_loss.item(), n=x.shape[0])
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
    logger.info(f"Valid Loss:                      {eval_losses.avg:.5f}")
    logger.info(f"Valid Accuracy:                  {accuracy:.5f}")

    return accuracy, all_logits, all_labels


def save_val_data(args, epoch, logits, labels):
    logger.info("***** Start saving val data *****")
    lr_str = get_lr_str(args.learning_rate)
    val_data_path = Path(args.target_dir) / args.target_subdir / f"val_data_lr_{lr_str}_epoch_{epoch}.npz"

    np.savez_compressed(
        val_data_path,
        logits=logits.astype(np.float16),
        labels=labels.astype(np.int16),
    )

    metrics_size_mb = val_data_path.stat().st_size / (1024**2)

    logger.info(f"Saved validation data:       {val_data_path}")
    logger.info(f"Val data size:               {metrics_size_mb:.2f} MB")
    logger.info("***** Val data saved successfully *****")


def train(args, model):
    pth = Path(args.target_dir) / args.target_subdir
    os.makedirs(pth, exist_ok=True)

    args.effective_train_batch_size = args.physical_train_batch_size * args.gradient_accumulation_steps

    logger.info("=" * 80)
    logger.info("***** Main info *****")
    logger.info(f"Physical train batch size:       {args.physical_train_batch_size}")
    logger.info(f"Gradient accumulation steps:     {args.gradient_accumulation_steps}")
    logger.info(f"Effective train batch size:      {args.effective_train_batch_size}")
    logger.info(f"Eval batch size:                 {args.eval_batch_size}")
    logger.info(f"Number of epochs:                {args.epoch_num}")
    logger.info(f"LR:                              {args.learning_rate}")
    logger.info(f"Beta1:                           {args.beta1}")
    logger.info(f"Beta2:                           {args.beta2}")
    logger.info(f"Number of warmup steps:          {args.warmup_steps}")
    logger.info(f"Weight decay type:               {args.decay_type}")
    logger.info(f"WD:                              {args.weight_decay}")
    logger.info(f"Image size:                      {args.img_size}")

    train_loader, val_loader = get_loader(args)
    logger.info(f"Train images:                    {len(train_loader.dataset)}")  # last batch is dropped
    logger.info(f"Validation images:               {len(val_loader.dataset)}")

    opt_steps_in_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_opt_step = opt_steps_in_epoch * args.epoch_num
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    logger.info(f"Optimization steps in epoch:     {opt_steps_in_epoch}")
    logger.info(f"Total optimization steps:        {total_opt_step}")

    # loss params
    logger.info(f"Loss w_1:                        {args.loss_w_1}")
    logger.info(f"Loss w_2:                        {args.loss_w_2}")
    logger.info(f"Loss w_f:                        {args.loss_w_f}")
    logger.info(f"Loss w_l:                        {args.loss_w_l}")
    logger.info(f"Loss w_affine:                   {args.loss_w_affine}")
    logger.info(f"Loss detach_reference:           {args.loss_detach_reference}")
    logger.info(f"Loss affine_reg_type:            {args.loss_affine_reg_type}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_opt_step)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_opt_step)

    best_acc = 0
    opt_step = 0
    start_epoch = 1
    best_epoch = -1

    if args.stn_vit_checkpoint:
        start_epoch, best_acc, best_epoch = load_stn_vit_train_state(args, optimizer, scheduler)

        opt_step = (start_epoch - 1) * (len(train_loader) // args.gradient_accumulation_steps)
        remaining_opt_steps = (args.epoch_num - start_epoch + 1) * (
            len(train_loader) // args.gradient_accumulation_steps
        )
        logger.info(f"Remaining opt steps:             {remaining_opt_steps}")

    logger.info(f"Num of validation steps:         {len(val_loader)}")
    logger.info(f"Output directory:                {Path(args.target_dir) / args.target_subdir}")
    logger.info("=" * 80)
    logger.info("\n")

    scaler = GradScaler("cuda", enabled=args.fp16 and args.device.type == "cuda")

    # Train!
    logger.info(f"***** Running training *****")

    model.zero_grad()
    set_seed(args)

    # main loss
    loss_fct = STNViTLoss(
        w_1=args.loss_w_1,
        w_2=args.loss_w_2,
        w_f=args.loss_w_f,
        w_l=args.loss_w_l,
        w_affine=args.loss_w_affine,
        detach_reference=args.loss_detach_reference,
        affine_reg_type=args.loss_affine_reg_type,
    ).to(args.device)

    # for logging loss values
    main_losses = AverageMeter()
    ce_1_losses = AverageMeter()
    ce_2_losses = AverageMeter()
    features_l1_losses = AverageMeter()
    logits_l1_losses = AverageMeter()
    affine_l2_losses = AverageMeter()

    for epoch in range(start_epoch, args.epoch_num + 1):
        logger.info(f"***** Epoch [{epoch} / {args.epoch_num}] started *****")
        epoch_start_time = time.time()
        model.train()

        epoch_iterator = tqdm(
            train_loader,
            disable=args.local_rank not in [-1, 0],
        )

        for batch_step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            with autocast("cuda", enabled=args.fp16 and args.device.type == "cuda"):
                logits_per_branch, features_per_branch, theta_per_branch = model(x)
                loss, loss_dict = loss_fct(
                    logits_per_branch=logits_per_branch,
                    features_per_branch=features_per_branch,
                    theta_per_branch=theta_per_branch,
                    targets=y,
                )
                loss /= args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accumulation_step = bool((batch_step + 1) % args.gradient_accumulation_steps == 0)
            is_last_batch = bool(batch_step == (len(train_loader) - 1))
            if is_accumulation_step:
                main_losses.update(loss.item() * args.gradient_accumulation_steps)
                ce_1_losses.update(loss_dict["ce_1"].item())
                ce_2_losses.update(loss_dict["ce_2"].item())
                features_l1_losses.update(loss_dict["features_l1"].item())
                logits_l1_losses.update(loss_dict["logits_l1"].item())
                affine_l2_losses.update(loss_dict["affine_l2"].item())

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
                    f"Training ({opt_step} / {total_opt_step} steps) "
                    f"(loss={main_losses.val:.5f}, "
                    f"ce1={ce_1_losses.val:.5f}, "
                    f"ce2={ce_2_losses.val:.5f}, "
                    f"feat={features_l1_losses.val:.5f}, "
                    f"logits={logits_l1_losses.val:.5f}, "
                    f"aff={affine_l2_losses.val:.5f})"
                )

            elif is_last_batch:
                optimizer.zero_grad(set_to_none=True)

        accuracy, logits, labels = valid(args, model, val_loader, opt_step, scheduler)
        if accuracy > best_acc:
            logger.info(f"New best accuracy:               {best_acc:.5f} -> {accuracy:.5f}")
            best_acc = accuracy
            best_epoch = epoch
        save_model(args, model, epoch, best_epoch, optimizer, scheduler, accuracy)
        save_val_data(args, epoch, logits, labels)

        model.train()

        logger.info(f"***** Epoch [{epoch} / {args.epoch_num}] finished *****")
        logger.info(f"Epoch time:                      {(time.time() - epoch_start_time):.2f} sec")
        logger.info(f"Best accuracy:                   {best_acc:.5f}")
        logger.info(f"Best epoch:                      {best_epoch}")
        logger.info("\n")

        main_losses.reset()
        ce_1_losses.reset()
        ce_2_losses.reset()
        features_l1_losses.reset()
        logits_l1_losses.reset()
        affine_l2_losses.reset()

    logger.info(f"Best Accuracy:                   {best_acc:.5f}")
    logger.info(f"Best epoch:                      {best_epoch}")
    logger.info("***** End training! *****")


def main():
    parser = argparse.ArgumentParser()

    # input params that are kept fixed
    parser.add_argument(
        "--vit_common_layers_checkpoint",
        type=str,
        default="/workspace/shared/ViT-B_16.pth",
        help="Where to search for common ViT layers.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/workspace/shared/target_dir",
        help="Directory to store validation data.",
    )
    parser.add_argument("--dataset_path", default="/workspace/dev_imagenet1k", help="Path to dataset folder.")
    parser.add_argument(
        "--vit_last_layers_checkpoint",
        type=str,
        default="/workspace/shared/target_dir/lr_0_001/model_lr_0_001_epoch_9.pth",
        help="Where to search for reference branch layers",
    )

    # input params that are varied
    parser.add_argument(
        "--target_subdir",
        type=str,
        required=True,
        help="Subdirectory name for checkpoints, logs, validation data, etc.",
    )
    parser.add_argument(
        "--stn_vit_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint with weights for two branches to resume training.",
    )
    parser.add_argument("--max_rotation_degrees", default=None, required=True, type=float)

    # main training params
    parser.add_argument(
        "--physical_train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training. Effective batch size = physical_train_batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--eval_batch_size", default=2048, type=int, help="Total batch size for eval.")
    parser.add_argument("--epoch_num", default=10, type=int, help="Total number of epochs to train the model.")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay for AdamW.")
    parser.add_argument(
        "--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate."
    )
    parser.add_argument(
        "--warmup_steps", default=2000, type=int, help="Step of training to perform learning rate warmup for."
    )

    # loss params
    parser.add_argument("--loss_w_1", default=1.0, type=float)
    parser.add_argument("--loss_w_2", default=1.0, type=float)
    parser.add_argument("--loss_w_f", default=0.1, type=float)
    parser.add_argument("--loss_w_l", default=0.1, type=float)
    parser.add_argument("--loss_w_affine", default=1.0, type=float)
    parser.add_argument("--loss_detach_reference", action="store_true")
    parser.add_argument(
        "--loss_affine_reg_type",
        choices=["orthogonal", "identity"],
        default="orthogonal",
    )

    # less important hyperparameters that are kept fixed
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for AdamW.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for AdamW.")
    parser.add_argument(
        "--model_type",
        default="ViT-B_16",
        help="Which variant to use.",
    )
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("\n")
    logger.info("{}".format(CONFIGS[args.model_type]))
    logger.info(f"Training parameters %s", args)
    logger.info(3 * "\n")

    set_seed(args)
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()
