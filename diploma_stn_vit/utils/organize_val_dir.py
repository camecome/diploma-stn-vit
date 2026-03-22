import os
import shutil
from pathlib import Path

from scipy.io import loadmat

NUM_CLASSES = 1000
IMAGENET_DIR = Path("~/diploma-stn-vit/imagenet1k")

val_dir = IMAGENET_DIR / "val"

ground_truth_file_path = (
    IMAGENET_DIR / "devkit/data/ILSVRC2012_validation_ground_truth.txt"
)
meta_file_path = IMAGENET_DIR / "devkit/data/meta.mat"

with open(ground_truth_file_path) as file:
    labels = [int(label.strip()) for label in file.readlines()]

val_images = sorted(val_dir.iterdir())

meta = loadmat(meta_file_path)
synsets = meta["synsets"]

wnids = [str(synset["WNID"][0][0]).strip() for synset in synsets[:NUM_CLASSES]]

for i, image in enumerate(val_images):
    label = labels[i]
    wnid = wnids[label - 1]

    target_dir = val_dir / wnid
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(image.name, target_dir / image.name)
