# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
import numpy as np
import torch

from tqdm import tqdm
from torch.amp import autocast
from pathlib import Path

from models.modeling import VisionTransformer, CONFIGS
from utils.augment_data_utils import get_loader

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def get_lr_str(learning_rate):
    return f"{learning_rate:.10f}".rstrip("0").rstrip(".").replace(".", "_")


def load_best_checkpoint(args, model):
    lr_str = get_lr_str(args.learning_rate)
    checkpoint_path = Path(args.target_dir) / args.target_subdir / f"model_lr_{lr_str}_epoch_{args.epoch_num}.pth"

    logger.info(f"Pretrained path:                 {args.pretrained_path}")
    logger.info(f"Checkpoint path:                 {checkpoint_path}")

    pretrained_state_dict = torch.load(args.pretrained_path, map_location="cpu")
    model.load_state_dict(pretrained_state_dict, strict=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainable_state_dict = checkpoint["model_state_dict"]

    model.transformer.encoder.layer[-1].load_state_dict(trainable_state_dict["last_transformer_block"])
    model.head.load_state_dict(trainable_state_dict["classifier_head"])

    logger.info(f"Checkpoint epoch:                {checkpoint.get('epoch')}")
    logger.info(f"Checkpoint initial LR:           {args.learning_rate:.3f}")
    logger.info(f"Checkpoint accuracy:             {checkpoint.get('accuracy', 0):.5f}")

    return model


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
            with autocast("cuda", enabled=args.fp16):
                logits = model(x)[0]
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

    logger.info("\n")
    logger.info("***** Testing Results *****")
    logger.info(f"Test loss:                      {eval_losses.avg:.5f}")
    logger.info(f"Test accuracy:                  {accuracy:.5f}")

    return accuracy, all_logits, all_labels


def main():
    parser = argparse.ArgumentParser()

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
        help="Subdirectory inside target_dir used for storing checkpoints and validation data.",
    )
    parser.add_argument("--dataset_path", default="/workspace/imagenet1k", help="Path to dataset folder.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial LR for used by best checkpoint.")
    parser.add_argument(
        "--epoch_num",
        default=None,
        required=True,
        type=int,
        help="Total number of training epochs used to obtain the best checkpoint.",
    )
    parser.add_argument("--max_rotation_degrees", default=None, type=float)

    # output params
    parser.add_argument(
        "--target_subdir",
        type=str,
        required=True,
        help="Subdirectory name for checkpoints, logs, validation data, etc.",
    )

    # these args are expected by get_loader(args)
    parser.add_argument("--physical_train_batch_size", default=128, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=2048, type=int)

    parser.add_argument("--model_type", default="ViT-B_16")
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.info("***** Testing setup *****")
    logger.info(f"Dataset path:                    {args.dataset_path}")
    logger.info(f"Eval batch size:                 {args.eval_batch_size}")
    logger.info(f"FP16:                            {args.fp16}")

    set_seed(args)

    model = VisionTransformer(
        CONFIGS[args.model_type],
        args.img_size,
        zero_head=True,
        num_classes=1000,
    )

    model = load_best_checkpoint(args, model)
    model.to(args.device)

    logger.info(f"Total parameters:                {sum(p.numel() for p in model.parameters()) / 1_000_000:.1f}M")
    logger.info(f"Out features:                    {model.head.out_features}")
    logger.info(f"Output directory:                {Path(args.target_dir) / args.target_subdir}")
    logger.info(f"max_rotation_degrees:            {args.max_rotation_degrees}")

    _, val_loader = get_loader(args)

    _, logits, labels = valid(args, model, val_loader)

    save_test_data(args, logits, labels)


if __name__ == "__main__":
    main()
