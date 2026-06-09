# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import random
import numpy as np

import torch

from tqdm import tqdm

from torch.amp import autocast


from models.modeling import VisionTransformer, CONFIGS
from models.stn.stn_vit import SpatialTransformerViT
from utils.augment_data_utils import get_loader

from pathlib import Path

logger = logging.getLogger(__name__)


class AverageMeter(object):
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

    logger.info("***** Loading common ViT layers *****")
    logger.info(f"Common layers checkpoint path:   {args.vit_common_layers_checkpoint}")

    common_layers = torch.load(args.vit_common_layers_checkpoint, map_location="cpu")
    base_vit.load_state_dict(common_layers, strict=True)
    logger.info("***** Common ViT layers succesfully downloaded *****")


def load_stn_vit_checkpoint(args, stn_vit):
    lr_str = get_lr_str(args.learning_rate)
    stn_vit_checkpoint = (
        Path(args.target_dir)
        / args.target_subdir
        / f"stn_vit_{lr_str}_epoch_{args.epoch_num}.pth"
    )
    logger.info("***** Loading STN checkpoint *****")
    logger.info(f"STN-ViT checkpoint path:         {stn_vit_checkpoint}")
    checkpoint = torch.load(stn_vit_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    stn_vit.last_layers.load_state_dict(state_dict["last_layers"])
    stn_vit.norms.load_state_dict(state_dict["norms"])
    stn_vit.heads.load_state_dict(state_dict["heads"])
    stn_vit.loc_net.load_state_dict(state_dict["loc_net"])
    logger.info("***** STN checkpoint succesfully downloaded *****")


def valid(args, model, val_loader):
    eval_losses = AverageMeter()

    logger.info(f"***** Running final evaluation *****")

    model.eval()
    all_preds, all_labels, all_logits = [], [], []

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
            with autocast("cuda", enabled=args.fp16 and args.device.type == "cuda"):
                logits, _, _ = model(x)
                eval_loss = loss_fct(logits, y)

            eval_losses.update(eval_loss.item(), n=x.shape[0])
            preds = torch.argmax(logits, dim=-1)

        all_logits.append(logits.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        epoch_iterator.set_description(f"Testing... (loss={eval_losses.val:.5f})")

    all_logits = np.concatenate(all_logits, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = simple_accuracy(all_preds, all_labels)

    logger.info("\n")
    logger.info("***** Testing Results *****")
    logger.info(f"Test loss:                       {eval_losses.avg:.5f}")
    logger.info(f"Test accuracy:                 {accuracy:.5f}")

    return accuracy, all_logits, all_labels


def save_test_data(args, logits, labels):
    logger.info("***** Start saving test data *****")

    lr_str = get_lr_str(args.learning_rate)
    test_data_path = (
        Path(args.target_dir)
        / args.target_subdir
        / f"test_data_lr_{lr_str}_epoch_{args.epoch_num}_angle_{int(args.max_rotation_degrees)}.npz"
    )

    np.savez_compressed(
        test_data_path,
        logits=logits.astype(np.float16),
        labels=labels.astype(np.int16),
    )

    logger.info(f"Saved test data:           {test_data_path}")
    metrics_size_mb = test_data_path.stat().st_size / (1024**2)
    logger.info(f"Test data size:                   {metrics_size_mb:.2f} MB")
    logger.info("***** Test data saved successfully *****")


def main():
    parser = argparse.ArgumentParser()

    # where to find all layers
    parser.add_argument(
        "--vit_common_layers_checkpoint",
        type=str,
        default="/workspace/shared/ViT-B_16.pth",
        help="Where to search for common ViT layers.",
    )
    parser.add_argument(
        "--target_subdir",
        type=str,
        required=True,
        help="Subdirectory name for checkpoints, logs, validation data, etc.",
    )
    parser.add_argument(
        "--epoch_num",
        default=None,
        required=True,
        type=int,
        help="Total number of training epochs used to obtain the best checkpoint.",
    )
    parser.add_argument("--dataset_path", default="/workspace/imagenet1k", help="Path to dataset folder.")

    # main eval params
    parser.add_argument("--max_rotation_degrees", default=None, required=True, type=float)
    parser.add_argument("--eval_batch_size", default=2048, type=int, help="Total batch size for eval.")

    # less important hyperparameters that are kept fixed
    parser.add_argument(
        "--physical_train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training. Effective batch size = physical_train_batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/workspace/shared/target_dir",
        help="Directory to store validation data.",
    )
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        "--model_type",
        default="ViT-B_16",
        help="Which variant to use.",
    )

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("***** Testing setup *****")
    logger.info(f"Dataset path:                    {args.dataset_path}")
    logger.info(f"Eval batch size:                 {args.eval_batch_size}")
    logger.info(f"FP16:                            {args.fp16}")

    set_seed(args)

    base_vit = VisionTransformer(CONFIGS[args.model_type], img_size=args.img_size, zero_head=True, num_classes=1000)

    # load common layers
    load_vit(args, base_vit)
    stn_vit = SpatialTransformerViT(base_vit=base_vit, max_rotation_degrees=args.max_rotation_degrees)

    load_stn_vit_checkpoint(args, stn_vit)
    stn_vit.to(args.device)

    logger.info(f"max_rotation_degrees:            {args.max_rotation_degrees}")
    logger.info(f"Out features:                    {stn_vit.heads[-1].out_features}")
    logger.info(f"Output directory:                {Path(args.target_dir) / args.target_subdir}")

    _, val_loader = get_loader(args)

    logger.info(f"Validation images:               {len(val_loader.dataset)}")

    _, logits, labels = valid(args, stn_vit, val_loader)

    save_test_data(args, logits, labels)


if __name__ == "__main__":
    main()
