import logging

import math
import torch

from torchvision import transforms, datasets
from torchvision.transforms import functional as F


from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from pathlib import Path

logger = logging.getLogger(__name__)


class FixedRotationDataset(Dataset):
    def __init__(self, dataset, angles, img_size, max_rotation_degrees):
        self.dataset = dataset
        self.angles = angles
        self.img_size = img_size

        self.safe_size = get_safe_rotation_size(img_size, max_rotation_degrees)

        self.to_tensor_and_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, target = self.dataset.samples[idx]
        img = self.dataset.loader(path)

        img = F.resize(img, [self.safe_size, self.safe_size])
        img = F.rotate(img, angle=self.angles[idx])
        img = F.center_crop(img, [self.img_size, self.img_size])
        img = self.to_tensor_and_normalize(img)

        return img, target


def get_safe_rotation_size(img_size, max_rotation_degrees):
    angle = math.radians(min(abs(max_rotation_degrees), 45))
    return math.ceil(img_size * (math.cos(angle) + math.sin(angle)))


def get_loader(args):
    if args.max_rotation_degrees is None or args.max_rotation_degrees < 0:
        raise ValueError("max_rotation_degrees must be a non-negative value and cannot be None.")

    # safe_size = get_safe_rotation_size(args.img_size, args.max_rotation_degrees)

    transform_train = transforms.Compose(
        [
            # transforms.Resize((safe_size, safe_size)),
            # transforms.RandomRotation(degrees=(-args.max_rotation_degrees, args.max_rotation_degrees)),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    data_dir = Path(args.dataset_path)

    trainset = datasets.ImageFolder(root=data_dir / "train", transform=transform_train)
    base_testset = datasets.ImageFolder(root=data_dir / "val", transform=None)

    # make deterministic angles
    generator = torch.Generator().manual_seed(args.seed)
    test_angles = (
        torch.empty(len(base_testset))
        .uniform_(-args.max_rotation_degrees, args.max_rotation_degrees, generator=generator)
        .tolist()
    )

    if len(test_angles) != len(base_testset):
        raise ValueError("Number of angles must match dataset size.")

    testset = FixedRotationDataset(
        dataset=base_testset,
        angles=test_angles,
        img_size=args.img_size,
        max_rotation_degrees=args.max_rotation_degrees,
    )

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.physical_train_batch_size,
        num_workers=16,
        pin_memory=True,
    )

    test_loader = DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, test_loader
