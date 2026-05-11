import argparse
import numpy as np
import torch
import torch.nn as nn

from models.modeling import VisionTransformer, CONFIGS

NUM_CLASSES = 1000
FINE_TUNE_IMG_SIZE = 384


def reset_head_to_zero(model: nn.Module, num_classes: int = NUM_CLASSES):
    hidden_size = model.head.in_features

    model.head = nn.Linear(hidden_size, num_classes)

    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)

    return model


def convert_npz_to_pth(npz_path: str, pth_path: str, model_type: str = "ViT-B_16"):
    config = CONFIGS[model_type]
    config.num_classes = NUM_CLASSES

    model = VisionTransformer(config, img_size=FINE_TUNE_IMG_SIZE, num_classes=NUM_CLASSES, zero_head=True)

    weights = np.load(npz_path)
    model.load_from(weights)

    reset_head_to_zero(model, num_classes=NUM_CLASSES)

    torch.save(model.state_dict(), pth_path)

    print(f"Saved: {pth_path}")
    print(model.head)
    print("head.weight sum:", model.head.weight.sum().item())
    print("head.bias sum:", model.head.bias.sum().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="/workspace/shared/ViT-B_16.npz")
    parser.add_argument("--pth_path", type=str, default="/workspace/shared/ViT-B_16.pth")
    parser.add_argument("--model_type", type=str, default="ViT-B_16")

    args = parser.parse_args()

    convert_npz_to_pth(
        npz_path=args.npz_path,
        pth_path=args.pth_path,
        model_type=args.model_type,
    )
