## Docker

You can either build the image locally using the provided Dockerfile:

```bash
docker build --network=host -t camecome/diploma-stn-vit:v4 .
```

or pull the prebuilt image from Docker Hub:

```bash
docker pull camecome/diploma-stn-vit:v4
```

Docker Hub repository:
https://hub.docker.com/r/camecome/diploma-stn-vit

## Running Training and Evaluation

### Training

Start Docker container:

```bash
docker run -it --gpus 1 \
    --ipc=host \
    -v ~/diploma-stn-vit/diploma_stn_vit:/workspace/diploma_stn_vit \
    -v ~/diploma-stn-vit/dev_imagenet1k:/workspace/dev_imagenet1k \
    -v ~/diploma-stn-vit/shared:/workspace/shared \
    camecome/diploma-stn-vit:v4
```

Here, `dev_imagenet1k` is the training subset of the original ImageNet-1k dataset, split into `train` and `val` directories.

After entering the container, activate the virtual environment:

```bash
source vit-env/bin/activate
```

Then run the training script:

```bash
python3 diploma_stn_vit/train.py --fp16 ...
```

Additional training arguments can be passed after `train.py` depending on the experiment configuration.

### Evaluation

Evaluation is launched in a similar way, but the full ImageNet-1k dataset should be mounted instead of `dev_imagenet1k`:

```bash
docker run -it --gpus 1 \
    --ipc=host \
    -v ~/diploma-stn-vit/diploma_stn_vit:/workspace/diploma_stn_vit \
    -v ~/diploma-stn-vit/imagenet1k:/workspace/imagenet1k \
    -v ~/diploma-stn-vit/shared:/workspace/shared \
    camecome/diploma-stn-vit:v4
```

After entering the container, activate the virtual environment:

```bash
source vit-env/bin/activate
```

Then run the testing command, for example:

```bash
python3 diploma_stn_vit/run_testing.py --fp16 ...
```

The `shared` directory is mounted into the container and can be used to store checkpoints, logs, plots, and other experiment artifacts that should remain available on the host machine.
