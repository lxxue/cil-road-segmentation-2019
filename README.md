# Road segmentation

Team Member: Jingyuan Ma, Yongqi Wang, Zhi Ye,

This repository contains the tools and models for the the course project of [Computational Intelligence Lab](http://da.inf.ethz.ch/teaching/2019/CIL/project.php) (Spring 2019): Road Segmentaion.

Credit:

[TorchSeg](https://github.com/ycszen/TorchSeg/) for the structure of repository

## Prerequisites

In our setting, the models are being run inside a deepo [Docker container](https://hub.docker.com/r/ufoym/deepo/) (using the default tag: `latest`)

A sample workflow using the docker container:

```shell
docker pull ufoym/deepo
# change the volume mount before
# mounting existing dir to container is possible with -v /host-dir:/target-dir
docker run -it --name cu100 ufoym/deepo bash
docker ps -a 
# CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
# ae2f6abc24d5        ufoym/deepo         "bash"              2 hours ago         Up About an hour    6006/tcp            eager_ganguly
docker ps -a -q
# ae2f6abc24d5
# entering the bash with container ID
docker exec -it cu100 bash
# follow instructions below
git clone https://github.com/wyq977/cil-road-segmentation-2019.git
cd cil-road-segmentation-2019
cd model/cil-resnet50
# train with CUDA device 0
python train.py -d 0
# eval using the default last epoh
python eval.py -d 0 -p ./val
# generate predicted groundtruth
python pred.py -d 0 -p ./pred
# generate submission.csv
python ../../cil-road-segmentation-2019/mask_to_submission.py --name submission -p ./pred/
# submit the submission.csv generated
```

- PyTorch >= 1.0
  - `pip3 install torch torchvision`
- Easydict
  - `pip3 install easydict`
- tqdm
  - `pip3 install tqdm`
- Apex

## Detailed Usage

Model dir:

```shell
├── config.py
├── dataloader.py
├── eval.py
├── network.py
├── pred.py
└── train.py
```

### Prepare data

With a tab-separated files specifying the path of images and groundtruth, `train.txt`, `val.txt`, `test.txt`.

`train.txt` or `val.txt`:

```shell
path-of-the-image   path-of-the-groundtruth
```

Noted that in the `test.txt`:

```shell
path-of-the-image   path-of-the-image
```

A handy script (`writefile.py`) using the package [glob](https://docs.python.org/3/library/glob.html) can be found inside the dataset directory.

### Training

To specify which CUDA device used for training, one can parse the index to `train.py`

A simple use case using the first CUDA device:

```shell
python train.py -d 0
```

Training can be restored from saved checkpoints

```shell
python train.py -d 0 -c log/snapshot/epoch-last.pth
```

### Predictive groudtruth labels

Similar to training

```shell
python pred.py -d 0 -p ../../cil-road-segmentation-2019/pred/ -e log/snapshot/epoch-last.pth
```

### Evalaute

```shell
python pred.py -d 0 -p ../../cil-road-segmentation-2019/val_pred/ -e log/snapshot/epoch-last.pth
```

### Create submission.csv

```shell
cd ../../cil-road-segmentation-2019/
python mask_to_submission.py --name submission -p pred/
```

### Distributed Training

All models but u-net support distributed training using `torch.distributed.launch`

For each run:

```shell
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
```

## Structure

```shell
├── README.md
├── cil-road-segmentation-2019 # datasets and submission script
├── docs
├── utils # helper function and utils for model
├── log
└── model
```

Under `model` directory, one can train, predict groundtruth (test images) and evaluate a model, details usage see the usage section above.

Different helpers functions used in constructing models, training, evaluation and IO operations regarding to `pytorch` could be found under `utils` folder. Functions or modules adapted from [TorchSeg](https://github.com/ycszen/TorchSeg/tree/master/model) is clearly marked and referenced in the files.

## Logistics

### Links:

1. [Projects description](http://da.inf.ethz.ch/teaching/2019/CIL/project.php)
2. [Road seg](https://inclass.kaggle.com/c/cil-road-segmentation-2019)
3. [Road seg kaggle sign in](https://www.kaggle.com/t/c83d1c6de17c433ca64b3a9174205c44)
4. [Link for dataset.zip](https://storage.googleapis.com/public-wyq/cil-2019/cil-road-segmentation-2019.zip)
5. [Course](http://da.inf.ethz.ch/teaching/2019/CIL/project.php)
6. [How to write paper](http://da.inf.ethz.ch/teaching/2019/CIL/material/howto-paper.pdf)

### Computational resources

1. https://scicomp.ethz.ch/wiki/Leonhard
2. https://scicomp.ethz.ch/wiki/CUDA_10_on_Leonhard#Available_frameworks
3. https://scicomp.ethz.ch/wiki/Using_the_batch_system#GPU

### Project submission

1. Submit the final report: https://cmt3.research.microsoft.com/ETHZCIL2019
2. Signed form here: http://da.inf.ethz.ch/teaching/2019/CIL/material/Declaration-Originality.pdf
3. Kaggle: https://inclass.kaggle.com/c/cil-road-segmentation-2019
