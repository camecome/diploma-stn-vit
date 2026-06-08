# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from torch.amp import autocast

from models.modeling import VisionTransformer, CONFIGS
from models.rot_vit.rot_vit import ROTVisionTransformer
from utils.augment_data_utils import get_loader

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
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


def save_test_data(args, logits, labels):
    logger.info("***** Start saving test data *****")
    test_data_path = Path(args.target_dir) / args.target_subdir / f"rot_vit_test_data_angle_{int(args.max_rotation_degrees)}.npz"
    np.savez_compressed(test_data_path, logits=logits.astype(np.float16), labels=labels.astype(np.int16))
    logger.info(f"Saved test data:                          {test_data_path}")
    metrics_size_mb = test_data_path.stat().st_size / (1024**2)
    logger.info(f"Test data size:                           {metrics_size_mb:.2f} MB")
    logger.info("***** Test data saved successfully *****")


def load_vit(args, base_vit):
    if not args.vit_common_layers_checkpoint:
        raise ValueError("'vit_common_layers_checkpoint' must be provided.")

    logger.info("***** Loading common ViT layers *****")
    logger.info(f"Common layers checkpoint path:            {args.vit_common_layers_checkpoint}")

    common_layers = torch.load(args.vit_common_layers_checkpoint, map_location="cpu")
    base_vit.load_state_dict(common_layers, strict=True)

    logger.info("***** Common ViT layers succesfully downloaded *****")


def load_rot_vit_checkpoint(args, rot_vit):
    logger.info("***** Loading ROT-ViT checkpoint *****")
    lr_str = get_lr_str(args.learning_rate)
    checkpoint_path = Path(args.target_dir) / args.target_subdir / f"rot_vit_lr_{lr_str}_epoch_{args.epoch_num}.pth"
    logger.info(f"Checkpoint path:                          {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint["model_state_dict"]

    rot_vit.last_layers.load_state_dict(model_state_dict["last_layers"])
    rot_vit.norms.load_state_dict(model_state_dict["norms"])
    rot_vit.heads.load_state_dict(model_state_dict["heads"])

    logger.info("***** ROT-ViT checkpoint successfully loaded *****")


def valid(args, model, val_loader):
    eval_losses = AverageMeter()

    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    epoch_iterator = tqdm(
        val_loader,
        desc="Testing... (loss=X.X)",
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
                outputs = model(x)
                logits = outputs["logits_2"]
                eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        all_logits.append(logits.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        epoch_iterator.set_description(f"Testing... (loss={eval_losses.val:.5f})")

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = simple_accuracy(all_preds, all_labels)

    logger.info("")
    logger.info("***** Test Results *****")
    logger.info(f"Test Loss:                                {eval_losses.avg:.5f}")
    logger.info(f"Test Accuracy:                            {accuracy:.5f}")

    return accuracy, all_logits, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vit_common_layers_checkpoint",
        type=str,
        default="/workspace/shared/ViT-B_16.pth",
        help="Path to pretrained ViT weights.",
    )
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/workspace/shared/target_dir",
        help="Directory to store checkpoints and validation data.",
    )
    parser.add_argument("--dataset_path", default="/workspace/imagenet1k", help="Path to dataset folder.")
    parser.add_argument(
        "--target_subdir",
        type=str,
        required=True,
        help="Subdirectory name for checkpoints, logs, validation data, etc.",
    )
    parser.add_argument(
        "--physical_train_batch_size",
        default=512,
        type=int,
        help="Physical batch size for training. Effective batch size = physical_train_batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument("--eval_batch_size", default=2048, type=int, help="Batch size for evaluation.")
    parser.add_argument("--epoch_num", default=None, required=True, type=int, help="Total number of epochs to train.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate for AdamW.")

    parser.add_argument(
        "--max_rotation_degrees",
        default=None,
        type=float,
        help="Max absolute rotation angle for ROT-ViT latent feature rotation.",
    )

    parser.add_argument("--model_type", default="ViT-B_16", help="Which ViT variant to use.")
    parser.add_argument("--img_size", default=224, type=int, help="Input image resolution.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training on GPUs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use mixed precision training.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    logger.info("***** Testing setup *****")
    logger.info(f"Dataset path:                             {args.dataset_path}")
    logger.info(f"Eval batch size:                          {args.eval_batch_size}")
    logger.info(f"FP16:                                     {args.fp16}")

    set_seed(args)

    base_vit = VisionTransformer(CONFIGS[args.model_type], num_classes=1000, img_size=args.img_size, zero_head=True)
    load_vit(args, base_vit)
    rot_vit = ROTVisionTransformer(base_vit=base_vit, max_rotation_degrees=args.max_rotation_degrees)

    load_rot_vit_checkpoint(args, rot_vit)
    rot_vit.to(args.device)

    logger.info(
        f"Total parameters:                         {sum(p.numel() for p in rot_vit.parameters()) / 1_000_000:.1f}M"
    )
    logger.info(f"Out features:                             {rot_vit.heads[0].out_features}")
    logger.info(f"Output directory:                         {Path(args.target_dir) / args.target_subdir}")
    logger.info(f"max_rotation_degrees:                     {args.max_rotation_degrees}")

    _, val_loader = get_loader(args)

    _, logits, labels = valid(args, rot_vit, val_loader)

    save_test_data(args, logits, labels)


if __name__ == "__main__":
    main()
