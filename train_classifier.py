import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import os
import pandas as pd
from mymodels import Cnn14_4blocks

# ========== 配置 ==========
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# ==========训练轮数 ==========
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 数据集路径
US8K_PATH = r"D:\audioset_tagging_cnn\data\UrbanSound8K"

# 加载标注
metadata = pd.read_csv(os.path.join(US8K_PATH, 'metadata', 'UrbanSound8K.csv'))
CSV_PATH = os.path.join(US8K_PATH, 'metadata', 'UrbanSound8K.csv')
metadata = pd.read_csv(CSV_PATH)
AUDIO_DIR = os.path.join(US8K_PATH, 'audio')
train_meta = metadata[metadata['fold'] != 10].reset_index(drop=True)
val_meta = metadata[metadata['fold'] == 10].reset_index(drop=True)

NUM_CLASSES = 10#类别数量
print(f"Number of classes: {NUM_CLASSES}")

# 参数
SAMPLE_RATE = 32000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000

# ========== 独立的预处理函数 ==========
def preprocess_audio_for_model(audio_path, sample_rate=32000, n_fft=1024, 
                                hop_length=320, n_mels=64, fmin=50, fmax=14000):
    """
    音频预处理函数，返回 numpy.ndarray，形状为 (1, time, freq)
    DataLoader 会自动添加 batch 维度，变成 (batch, 1, time, freq)
    """
    # 加载音频
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # 统一长度到 5 秒
    target_len = 5 * sample_rate
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    
    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, 
        hop_length=hop_length, n_mels=n_mels, 
        fmin=fmin, fmax=fmax
    )
    
    # 转换为对数刻度（dB）
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 调整形状： (freq, time) -> (time, freq) -> (1, time, freq)
    log_mel = log_mel.T
    log_mel = log_mel[np.newaxis, :, :]  # (1, time, freq)
    
    return log_mel

# ========== 数据集类 ==========
class UrbanSound8KDataset(Dataset):
    def __init__(self, metadata, audio_dir, is_training=True):
        self.metadata = metadata
        self.audio_dir = audio_dir
        self.is_training = is_training  # 新增：区分训练集和验证集

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fold = f"fold{row['fold']}"
        audio_path = os.path.join(self.audio_dir, fold, row['slice_file_name'])
        
         # 加载音频
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

         # 统一长度到 4 秒
        target_len = 4 * SAMPLE_RATE
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

            # 数据增强
        if self.is_training:
            # 加噪声
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise
        
        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, 
            hop_length=HOP_LENGTH, n_mels=N_MELS, 
            fmin=FMIN, fmax=FMAX
        )
        
        # 转换为对数刻度（dB）
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 调整形状： (freq, time) -> (time, freq) -> (1, time, freq)
        log_mel = log_mel.T
        log_mel = log_mel[np.newaxis, :, :]
        
        # 转换为 Tensor
        log_mel = torch.from_numpy(log_mel).float()
        
        label = row['classID']
        return log_mel, label

# 创建数据集和数据加载器
train_dataset = UrbanSound8KDataset(train_meta, AUDIO_DIR, is_training=True)
val_dataset = UrbanSound8KDataset(val_meta, AUDIO_DIR, is_training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ========== 创建模型 ==========
model = Cnn14_4blocks(classes_num=NUM_CLASSES).to(DEVICE)

# ========== 加载预训练权重 ==========
checkpoint_path = r"D:\audioset_tagging_cnn\Cnn14_mAP=0.431.pth"

print(f"\nLooking for checkpoint at: {checkpoint_path}")
print(f"File exists: {os.path.exists(checkpoint_path)}")

if os.path.exists(checkpoint_path):
    print("\n=== Debug: Check checkpoint structure ===")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Top-level keys: {list(checkpoint.keys())}")
        
        if 'model' in checkpoint:
            print("Found 'model' key, using checkpoint['model']")
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        model_dict = model.state_dict()
        
        # 匹配参数
        matched = 0
        for name in model_dict.keys():
            if name in pretrained_dict:
                if pretrained_dict[name].shape == model_dict[name].shape:
                    model_dict[name] = pretrained_dict[name]
                    matched += 1
        
        print(f"\n✅ Matched {matched} out of {len(model_dict)} layers")
        
        if matched > 0:
            model.load_state_dict(model_dict, strict=False)
            print("✅ Successfully loaded pretrained weights!")

# ========== 冻结/解冻设置 ==========
for param in model.parameters():
    param.requires_grad = True

print(f"\nTrainable params: {[n for n, p in model.named_parameters() if p.requires_grad]}")

# ========== 优化器和损失函数 ==========
# 不同层使用不同学习率
optimizer = optim.Adam([
    {'params': model.conv_block1.parameters(), 'lr': 1e-5},   # 底层：最小学习率
    {'params': model.conv_block2.parameters(), 'lr': 1e-5},   # 底层：最小学习率
    {'params': model.conv_block3.parameters(), 'lr': 5e-5},   # 中层：中等学习率
    {'params': model.conv_block4.parameters(), 'lr': 1e-4},   # 高层：较大学习率
    {'params': model.fc1.parameters(), 'lr': 1e-3},           # 分类层：最大学习率
    {'params': model.fc_audioset.parameters(), 'lr': 1e-3},
], lr=1e-4)  # 默认学习率作为后备
criterion = nn.CrossEntropyLoss()

# ========== 训练 ==========
print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (mel_specs, labels) in enumerate(train_loader):
        mel_specs, labels = mel_specs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = criterion(outputs['clipwise_output'], labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % 20 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/20:.4f}')
            running_loss = 0.0
    
    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for mel_specs, labels in val_loader:
            mel_specs, labels = mel_specs.to(DEVICE), labels.to(DEVICE)
            outputs = model(mel_specs)
            _, pred = torch.max(outputs['clipwise_output'], 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f'Epoch {epoch+1} Validation Acc: {100*correct/total:.2f}%')

# 保存模型
torch.save(model.state_dict(), "UrbanSound8k_final.pth")
print("Training complete! Model saved as UrbanSound8k_final.pth")