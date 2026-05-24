import os
import librosa
import numpy as np
from panns_inference import AudioTagging

# 要识别的音频文件路径
audio_files = [
    r"D:\audioset_tagging_cnn\resources\test2.m4a",
    r"D:\audioset_tagging_cnn\resources\test3.m4a",
]

print("正在加载模型...")
at = AudioTagging(checkpoint_path=None, device='cuda')

for audio_path in audio_files:
    if not os.path.exists(audio_path):
        print(f"文件不存在: {audio_path}")
        continue
    
    print(f"\n📁 {os.path.basename(audio_path)}")
    
    # 加载音频
    audio, _ = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]  # 添加 batch 维度
    
    # 推理
    clipwise_output, _ = at.inference(audio)
    
    # 获取 Top 5 结果
    top5_idx = np.argsort(clipwise_output[0])[::-1][:5]
    
    print("-" * 40)
    for i, idx in enumerate(top5_idx):
        print(f"  {i+1}. {at.labels[idx]}: {clipwise_output[0][idx]:.4f}")