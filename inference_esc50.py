import torch
import librosa
import numpy as np
import os
import glob
from mymodels import Cnn14_4blocks

# ========== 删除这段代码 ==========
# import importlib.util
# spec = importlib.util.spec_from_file_location("audio_utils", "train_classifier.py")
# audio_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(audio_utils)
# preprocess_audio_for_model = audio_utils.preprocess_audio_for_model
# ================================

# ========== 把函数直接复制进来 ==========
def preprocess_audio_for_model(audio_path, sample_rate=32000, n_fft=1024, 
                                hop_length=320, n_mels=64, fmin=50, fmax=14000):
    """
    音频预处理函数，返回 numpy.ndarray，形状为 (1, time, freq)
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
    log_mel = log_mel[np.newaxis, :, :]
    
    return log_mel
# =====================================

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 参数（必须和训练时一致）
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 64
FMIN = 50
FMAX = 14000


# ESC-50 类别
CLASSES = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 
    'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing', 
    'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 
    'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing', 
    'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 
    'clock_tick', 'bell', 'train', 'car', 'bus', 'truck', 'boat', 'airplane', 'helicopter', 
    'chainsaw', 'siren', 'gunshot'
]

# ========== 加载模型 ==========
def load_model(model_path, num_classes=50):
    model = Cnn14_4blocks(classes_num=num_classes).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 模型加载成功: {model_path}")
    return model

# ========== 识别单个音频 ==========
def predict_single_topk(model, audio_path, top_k=5):
    # 调用预处理函数，得到 numpy 数组，形状 (1, time, freq)
    input_numpy = preprocess_audio_for_model(
        audio_path, SAMPLE_RATE, N_FFT, 
        HOP_LENGTH, N_MELS, FMIN, FMAX
    )
    # 转换为 Tensor，形状 (1, time, freq)，DataLoader 不需要加 batch 维度
    input_tensor = torch.from_numpy(input_numpy).float().to(DEVICE)
    
    # 手动添加 batch 维度（因为推理时没有 DataLoader）
    input_tensor = input_tensor.unsqueeze(0)  # (1, 1, time, freq)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output['clipwise_output'], dim=1)
    
    # 获取 Top-K 结果
    probs, indices = torch.topk(probabilities, k=top_k, dim=1)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    # 打印结果
    print(f"\n📁 音频文件: {os.path.basename(audio_path)}")
    print("-" * 40)
    for i, (idx, prob) in enumerate(zip(indices, probs)):
        class_name = CLASSES[idx]
        print(f"  {i+1}. {class_name:20s} ({prob*100:.2f}%)")
    
    return indices, probs

# ========== 批量识别文件夹 ==========
def recognize_folder(model, folder_path, top_k=5):
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    # 查找音频文件
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not audio_files:
        print(f"❌ 文件夹中没有找到音频文件: {folder_path}")
        return
    
    print(f"\n📁 文件夹: {folder_path}")
    print(f"📄 找到 {len(audio_files)} 个音频文件")
    print("=" * 60)
    
    for i, file_path in enumerate(sorted(audio_files), 1):
        file_name = os.path.basename(file_path)
        try:
            print(f"\n[{i}] 识别: {file_name}")
            predict_single_topk(model, file_path, top_k)
        except Exception as e:
            print(f"  ❌ 识别失败: {e}")

# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载模型
    model = load_model("esc50_final.pth", num_classes=50)
    
    # 要识别的文件夹路径（改成你的）
    target_folder = r"D:\audioset_tagging_cnn\resources"
    
    # 批量识别
    recognize_folder(model, target_folder, top_k=5)