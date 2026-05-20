# TimeSformer

## 项目简介

TimeSformer 是一个基于 Transformer 的视频理解模型，将时空注意力机制直接应用于视频序列建模任务中。与传统 3D 卷积网络相比，它能够更自然地建模长时序依赖关系，并支持灵活切换不同的注意力形式，例如 divided space-time、space-only 和 joint space-time。

本仓库提供了官方 PyTorch 实现，覆盖了视频分类任务中的训练、测试、微调和分布式训练流程，适合用于复现实验结果、开展动作识别研究，以及在标准视频数据集上进行模型对比。

## 快速开始

如果你只是想尽快把项目跑起来，建议按下面流程操作：

1. 创建并激活环境。
2. 安装依赖。
3. 构建项目。
4. 准备数据集路径。
5. 选择一个配置文件启动训练或测试。

示例：

```bash
conda create -n timesformer python=3.7 -y
source activate timesformer
pip install torchvision simplejson einops timm psutil scikit-learn opencv-python tensorboard
pip install 'git+https://github.com/facebookresearch/fvcore'
conda install av -c conda-forge
python setup.py build develop
```

启动默认训练配置：

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8
```

如果你暂时没有 8 张 GPU，可以参考 [configs/](configs/) 中的配置，适当调小 `NUM_GPUS`、`TRAIN.BATCH_SIZE`、`TEST.BATCH_SIZE` 和 `DATA_LOADER.NUM_WORKERS`。

## 常见问题

### 1. 没有多张 GPU，可以运行吗？

可以，但你需要修改配置文件中的 GPU 数量、batch size 和 worker 数量。仓库中提供了部分多 GPU 之外的示例配置，你也可以基于现有 yaml 自行调整。

### 2. 数据集路径必须每次从命令行传入吗？

不是。你可以在运行命令时通过 `DATA.PATH_TO_DATA_DIR` 传入，也可以直接把它写进对应的 yaml 配置文件中。

### 3. 测试或推理时最关键的参数是什么？

通常是 `TEST.CHECKPOINT_FILE_PATH`、`TRAIN.ENABLE False`，以及正确的数据集路径。测试前请确认 checkpoint 与配置文件对应。

### 4. 显存不够怎么办？

可以优先尝试减少 batch size、降低输入分辨率、减少采样帧数，或者使用更小的模型配置。像 TimeSformer-HR、TimeSformer-L 这类配置通常对显存要求更高。

### 5. 这个仓库适合做什么？

适合做视频动作识别实验、复现论文结果、对比不同注意力机制，以及在 Kinetics、SSv2、HowTo100M 等数据集上开展视频理解研究。

这是我们在 ICML 2021 论文 [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf) 的官方 PyTorch 实现。本仓库提供了用于训练和测试 TimeSformer 模型的 PyTorch 代码。TimeSformer 提供了一个高效的视频分类框架，并在 Kinetics-400 等多个视频动作识别基准上取得了领先结果。

如果 TimeSformer 对你的研究有帮助，欢迎引用以下 BibTeX：

```BibTeX
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```

# 模型库

我们提供了在 Kinetics-400（K400）、Kinetics-600（K600）、Something-Something-V2（SSv2）以及 HowTo100M 数据集上预训练的 TimeSformer 模型。

| 名称 | 数据集 | 帧数 | 空间裁剪 | acc@1 | acc@5 | 下载链接 |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | K400 | 8 | 224 | 77.9 | 93.2 | [model](https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0) |
| TimeSformer-HR | K400 | 16 | 448 | 79.6 | 94.0 | [model](https://www.dropbox.com/s/6f0x172lpqy3oxt/TimeSformer_divST_16x16_448_K400.pyth?dl=0) |
| TimeSformer-L | K400 | 96 | 224 | 80.6 | 94.7 | [model](https://www.dropbox.com/s/r1iuxahif3sgimo/TimeSformer_divST_96x4_224_K400.pyth?dl=0) |

| 名称 | 数据集 | 帧数 | 空间裁剪 | acc@1 | acc@5 | 下载链接 |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | K600 | 8 | 224 | 79.1 | 94.4 | [model](https://www.dropbox.com/s/4h2qt41m2z3aqrb/TimeSformer_divST_8x32_224_K600.pyth?dl=0) |
| TimeSformer-HR | K600 | 16 | 448 | 81.8 | 95.8 | [model](https://www.dropbox.com/s/ft1e92g2vhvxecv/TimeSformer_divST_16x16_448_K600.pyth?dl=0) |
| TimeSformer-L | K600 | 96 | 224 | 82.2 | 95.6 | [model](https://www.dropbox.com/s/857rx6xeclxfhdg/TimeSformer_divST_96x4_224_K600.pyth?dl=0) |

| 名称 | 数据集 | 帧数 | 空间裁剪 | acc@1 | acc@5 | 下载链接 |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | SSv2 | 8 | 224 | 59.1 | 85.6 | [model](https://www.dropbox.com/s/tybhuml57y24wpm/TimeSformer_divST_8_224_SSv2.pyth?dl=0) |
| TimeSformer-HR | SSv2 | 16 | 448 | 61.8 | 86.9 | [model](https://www.dropbox.com/s/9t68uzk8w2fpfnv/TimeSformer_divST_16_448_SSv2.pyth?dl=0) |
| TimeSformer-L | SSv2 | 64 | 224 | 62.0 | 87.5 | [model](https://www.dropbox.com/s/3f1rm2al8mhprwa/TimeSformer_divST_64_224_SSv2.pyth?dl=0) |

| 名称 | 数据集 | 帧数 | 空间裁剪 | 单次 clip 覆盖时长 | acc@1 | 下载链接 |
| --- | --- | --- | --- | --- | --- | --- |
| TimeSformer | HowTo100M | 8 | 224 | 8.5s | 56.8 | [model](https://www.dropbox.com/s/9v8hcm88b9tc6ff/TimeSformer_divST_8x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 32 | 224 | 34.1s | 61.2 | [model](https://www.dropbox.com/s/4roflx4q1gscu85/TimeSformer_divST_32x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 64 | 448 | 68.3s | 62.2 | [model](https://www.dropbox.com/s/15bvqltl1j5vyp3/TimeSformer_divST_64x32_224_HowTo100M.pyth?dl=0) |
| TimeSformer | HowTo100M | 96 | 224 | 102.4s | 62.6 | [model](https://www.dropbox.com/s/t2mzgahnfhgakma/TimeSformer_divST_96x32_224_HowTo100M.pyth?dl=0) |

需要注意的是，这些模型使用了与论文中略有不同的实现重新训练，因此其性能与论文报告结果之间可能会存在少量差异。

你可以按如下方式加载预训练模型：

```python
import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='/path/to/pretrained/model.pyth')

dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)

pred = model(dummy_video,) # (2, 400)
```

# 安装

首先，创建并激活 conda 虚拟环境：

```
conda create -n timesformer python=3.7 -y
source activate timesformer
```

然后安装以下依赖：

- torchvision：`pip install torchvision` 或 `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/)：`pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson：`pip install simplejson`
- einops：`pip install einops`
- timm：`pip install timm`
- PyAV：`conda install av -c conda-forge`
- psutil：`pip install psutil`
- scikit-learn：`pip install scikit-learn`
- OpenCV：`pip install opencv-python`
- tensorboard：`pip install tensorboard`

最后，通过以下命令构建 TimeSformer 代码库：

```
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
python setup.py build develop
```

# 使用方法

## 数据集准备

请参考 [DATASET.md](timesformer/datasets/DATASET.md) 中提供的数据集准备说明。

## 训练默认版 TimeSformer

默认版 TimeSformer 使用 divided space-time attention，输入为 8 帧 clip，空间裁剪分辨率为 224x224。可使用以下命令进行训练：

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

你可能需要在命令行中通过 `DATA.PATH_TO_DATA_DIR path_to_your_dataset` 指定数据集路径；或者也可以直接在 yaml 配置文件中加入：

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```

这样以后每次运行时就不需要重复在命令行传入该参数。

## 使用不同数量的 GPU

如果你想使用更少数量的 GPU，需要修改 [`configs/`](configs/) 下对应的 `.yaml` 配置文件。具体来说，需要调整每个配置文件中的 `NUM_GPUS`、`TRAIN.BATCH_SIZE`、`TEST.BATCH_SIZE`、`DATA_LOADER.NUM_WORKERS` 等字段。

其中，`BATCH_SIZE` 的值应与 `NUM_GPUS` 相同或更大。在 [configs/Kinetics/TimeSformer_divST_8x32_224_4gpus.yaml](configs/Kinetics/TimeSformer_divST_8x32_224_4gpus.yaml) 中，我们提供了一个 4 GPU 配置示例。

## 使用不同的自注意力方案

如果你想尝试不同的时空自注意力方案，例如仅空间注意力（space-only）或联合时空注意力（joint space-time attention），可以分别使用以下命令：

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

以及：

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_jointST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

## 训练不同的 TimeSformer 变体

如果你想训练更强的 TimeSformer 变体，例如 TimeSformer-HR（16 帧输入，448x448 空间分辨率）和 TimeSformer-L（96 帧输入，224x224 空间分辨率），可以使用以下命令：

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_16x16_448.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

以及：

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_96x4_224.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

注意，这些模型通常需要显存约 32GB 的 GPU。

## 推理

可以通过 `TRAIN.ENABLE` 和 `TEST.ENABLE` 控制某次运行是否进行训练或测试。测试时，你还需要通过 `TEST.CHECKPOINT_FILE_PATH` 指定 checkpoint 模型路径。

```
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

## 通过 Slurm 进行单节点训练

如需通过 Slurm 训练 TimeSformer，请参考单节点训练脚本 [slurm_scripts/run_single_node_job.sh](slurm_scripts/run_single_node_job.sh)。

## 通过 Submitit 进行多节点训练

支持通过 Slurm 和 submitit 进行分布式训练。

```
pip install submitit
```

若要在 Kinetics 数据集上使用 4 个节点、每个节点 8 张 GPU 训练 TimeSformer，可使用以下命令：

```
python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/ --num_shards 4 --name ${JOB_NAME} --use_volta32
```

我们还提供了一个用于启动 Slurm 多节点任务的脚本：[slurm_scripts/run_multi_node_job.sh](slurm_scripts/run_multi_node_job.sh)。

## 微调

如果你想基于已有的 PyTorch checkpoint 进行微调，可以在命令行中添加以下参数，也可以将其写入 YAML 配置文件：

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
TRAIN.FINETUNE True
```

## HowTo100M 数据集划分

如果你想在 HowTo100M 上进行长时序视频建模实验，请从[这里](https://www.dropbox.com/sh/ttvsxwqypijjuda/AACmJx1CnddW6cVBoc21eSuva?dl=0)下载训练/测试划分文件。

# 环境说明

本代码基于 Ubuntu 20.04 和 Python 3.7 开发。训练时，我们使用了 4 个 GPU 计算节点，每个节点包含 8 张 Tesla V100 GPU（总计 32 张 GPU）。其他平台或显卡尚未经过充分测试。

# 许可证

本项目大部分内容采用 [CC-NC 4.0 International license](LICENSE) 授权。但项目中的部分代码适用其他许可证： [SlowFast](https://github.com/facebookresearch/SlowFast) 和 [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 采用 Apache 2.0 license。

# 贡献

欢迎提交 Pull Request。更多信息请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

# 致谢

TimeSformer 构建于 [PySlowFast](https://github.com/facebookresearch/SlowFast) 和 [Ross Wightman](https://github.com/rwightman) 开发的 [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 之上。感谢这些作者开源他们的代码。如果你使用了我们的模型，也建议同时引用这些工作：

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
