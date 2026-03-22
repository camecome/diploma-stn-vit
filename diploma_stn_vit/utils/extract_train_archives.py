import tarfile
from pathlib import Path

IMAGENET_DIR = Path("~/diploma-stn-vit/imagenet1k")
train_dir = IMAGENET_DIR / "train"

for tar_path in train_dir.glob("*.tar"):
    folder_name = tar_path.stem
    folder_path = train_dir / folder_name
    folder_path.mkdir(exist_ok=True)

    print(f"Extracting {tar_path.name} into {folder_path} ...")

    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=folder_path)

        tar_path.unlink()
        print(f"Deleted {tar_path.name}")
    except Exception as e:
        print(f"Error with {tar_path.name}: {e}")

print("Done! All archives have been extracted into folders named after WNID.")
