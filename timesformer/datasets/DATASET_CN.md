# 数据集准备
## Kinetics数据集
可通过下方链接下载Kinetics数据集：[link](https://github.com/cvdfoundation/kinetics-dataset):

全部视频下载完成后，将视频短边分辨率缩放至256像素。接着分别生成训练集、验证集、测试集的CSV文件，命名为`train.csv`、`val.csv`、`test.csv`。

```
CSV文件格式如下：
视频文件路径1 标签1
视频文件路径2 标签2
视频文件路径3 标签3
……
视频文件路径N 标签N
```

## Something-Something V2数据集
前往数据集官方渠道下载数据集及标注文件。[dataset provider](https://20bn.com/datasets/something-something).

通过对应链接下载训练集、验证集的帧序列列表。

以30帧每秒的速率提取视频帧。实验环境使用4.1.3版本FFmpeg，执行提取命令：
`ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"`

提取后的帧文件目录结构，需与帧序列列表保持一致。

将所有标注JSON文件、帧序列列表存放至同一文件夹，配置参数`DATA.PATH_TO_DATA_DIR`指向该文件夹路径；参数`DATA.PATH_PREFIX`设置为帧文件存放目录路径。