import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)

from pathlib import Path

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == "imagenet1k":
        data_dir = Path("imagenet1k")

        trainset = datasets.ImageFolder(root=data_dir / "train", transform=transform_train)

        valset = (
            datasets.ImageFolder(root=data_dir / "val", transform=transform_test)
            if args.local_rank in [-1, 0]
            else None
        )
    else:
        raise ValueError("Unsupported dataset")

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)

    val_sampler = SequentialSampler(valset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            valset,
            sampler=val_sampler,
            batch_size=args.eval_batch_size,
            num_workers=4,
            pin_memory=True,
        )
        if valset is not None
        else None
    )

    return train_loader, val_loader
