import logging

from torchvision import transforms, datasets


from utils.augment_data_utils import get_safe_rotation_size

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from pathlib import Path

logger = logging.getLogger(__name__)


# loader for rotated train and fixed eval/test
def get_loader(args):
    if args.max_rotation_degrees is not None:
        safe_size = get_safe_rotation_size(args.img_size, args.max_rotation_degrees)
        transform_train = transforms.Compose(
            [
                transforms.Resize((safe_size, safe_size)),
                transforms.RandomRotation(degrees=args.max_rotation_degrees),
                transforms.CenterCrop((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((safe_size, safe_size)),
                transforms.RandomRotation(degrees=args.max_rotation_degrees),
                transforms.CenterCrop((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
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

    data_dir = Path(args.dataset_path)

    trainset = datasets.ImageFolder(root=data_dir / "train", transform=transform_train)
    testset = datasets.ImageFolder(root=data_dir / "val", transform=transform_test)

    train_sampler = RandomSampler(trainset)  # shuffle indexes
    test_sampler = SequentialSampler(testset)  # get indexes one by one w/o shuffle

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
