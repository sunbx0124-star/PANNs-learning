import torch
import librosa
import csv
from mymodels import Cnn14_4blocks   # 从 mymodels.py 导入您的模型

# 1. 加载类别标签（把数字ID翻译成声音名称）
labels_path = "../metadata/class_labels_indices.csv"  # 注意路径，回到上级目录再进metadata
id_to_label = {}
with open(labels_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        # row[0] 是数字ID（如"0"），row[1] 是显示名称（如"Speech"）
        id_to_label[int(row[0])] = row[2]

# 创建模型实例
model = Cnn14_4blocks(sample_rate=32000, window_size=1024, hop_size=320,
                      mel_bins=64, fmin=50, fmax=14000, classes_num=527)

# 2. 加载预训练权重（strict=False 允许部分层不匹配）
checkpoint_path = "Cnn14_mAP=0.431.pth"  # 确保这个文件在 pytorch 文件夹里
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
print("已加载预训练权重（部分层匹配）")

model.eval()

# 3. 加载真实的音频文件（注意路径）
audio_path = "../resources/my_audio.wav"  # 相对于 pytorch/ 文件夹
waveform, sr = librosa.load(audio_path, sr=32000)  # 加载并重采样到 32000 Hz
waveform = torch.from_numpy(waveform).unsqueeze(0)  # 添加 batch 维度 (1, 32000)

# 4. 推理
with torch.no_grad():
    output = model(waveform)

# 5. 打印结果
probs = output['clipwise_output'].numpy()[0]
top5_indices = probs.argsort()[-5:][::-1]
print("Top 5 预测类别和概率：")
for idx in top5_indices:
    label_name = id_to_label.get(idx, f"未知类别{idx}")  # 关键修改
    print(f"  {label_name}: {probs[idx]:.4f}")