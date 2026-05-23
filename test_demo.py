import torch
from timesformer.models.vit import TimeSformer

print("1. 成功导入 TimeSformer 模块！")

# 实例化默认的 Divided Space-Time Attention 模型
model = TimeSformer(
    img_size=224, 
    num_classes=400, 
    num_frames=8, 
    attention_type='divided_space_time'
)
model.eval()
print("2. 模型结构实例化成功！")

# 模拟一个 Batch 的视频输入 (BatchSize=2, Channels=3, Frames=8, H=224, W=224)
dummy_video = torch.randn(2, 3, 8, 224, 224)
print("3. 生成随机视频 Tensor 数据...")

# 执行前向传播
with torch.no_grad():
    predictions = model(dummy_video)

print(f"4. 前向传播成功！输出维度: {predictions.shape}") 
# 预期输出应该是 [2, 400] (对应2个样本，400个类别的预测概率)