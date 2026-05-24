import os
import glob
from pydub import AudioSegment

# ========================================
# 1. 配置参数（修改这里）
# ========================================

# 要转换的文件夹路径（改成你的）
input_folder = r"D:\audioset_tagging_cnn\resources"  # ← 改成你的文件夹

# 是否包含子文件夹（True=搜索子文件夹，False=只搜索当前文件夹）
include_subfolders = True

# 转换后是否删除原文件（True=删除，False=保留）
delete_original = True  # 建议先设为 False，确认没问题再改

# 支持的输入格式（自动识别这些格式的文件）
SUPPORTED_FORMATS = ['.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus']

# ========================================
# 2. 查找所有非 WAV 音频文件
# ========================================

def find_audio_files(folder, include_subfolders=True):
    """查找文件夹中所有非 WAV 的音频文件"""
    audio_files = []
    
    if include_subfolders:
        # 搜索所有子文件夹（递归）
        for format_ext in SUPPORTED_FORMATS:
            # ** 表示搜索所有子文件夹
            pattern = os.path.join(folder, "**", f"*{format_ext}")
            audio_files.extend(glob.glob(pattern, recursive=True))
    else:
        # 只搜索当前文件夹
        for format_ext in SUPPORTED_FORMATS:
            pattern = os.path.join(folder, f"*{format_ext}")
            audio_files.extend(glob.glob(pattern))
    
    return audio_files

# ========================================
# 3. 转换单个文件
# ========================================

def convert_to_wav(input_path, output_path):
    """将单个音频文件转换为 WAV 格式"""
    try:
        # 根据文件扩展名选择读取格式
        ext = os.path.splitext(input_path)[1].lower().replace('.', '')
        
        # 读取音频
        audio = AudioSegment.from_file(input_path, format=ext)
        
        # 导出为 WAV
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"  转换失败: {e}")
        return False

# ========================================
# 4. 批量转换主函数
# ========================================

def batch_convert(input_folder, include_subfolders=True, delete_original=False):
    """
    批量转换所有非 WAV 音频文件为 WAV 格式
    
    参数:
        input_folder: 要转换的文件夹路径
        include_subfolders: 是否包含子文件夹
        delete_original: 是否删除原文件
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 文件夹不存在: {input_folder}")
        return
    
    # 查找所有需要转换的文件
    print(f"📁 正在扫描文件夹: {input_folder}")
    print(f"   包含子文件夹: {'是' if include_subfolders else '否'}")
    print("-" * 60)
    
    audio_files = find_audio_files(input_folder, include_subfolders)
    
    if not audio_files:
        print("✅ 没有找到需要转换的非 WAV 音频文件")
        return
    
    print(f"📄 找到 {len(audio_files)} 个需要转换的文件")
    print("-" * 60)
    
    # 统计转换结果
    success_count = 0
    fail_count = 0
    
    # 逐个转换
    for i, input_path in enumerate(sorted(audio_files), 1):
        # 获取文件名（不含路径）
        file_name = os.path.basename(input_path)
        # 生成输出路径（把扩展名改成 .wav）
        output_path = os.path.splitext(input_path)[0] + ".wav"
        
        # 检查是否已经是 WAV（防止重复转换）
        if input_path.lower().endswith('.wav'):
            print(f"{i:3d}. 跳过（已是 WAV）: {file_name}")
            continue
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"{i:3d}. 跳过（WAV 已存在）: {file_name}")
            continue
        
        # 执行转换
        print(f"{i:3d}. 正在转换: {file_name}")
        success = convert_to_wav(input_path, output_path)
        
        if success:
            success_count += 1
            print(f"      ✅ 转换成功: {os.path.basename(output_path)}")
            
            # 如果需要删除原文件
            if delete_original:
                os.remove(input_path)
                print(f"      🗑️ 已删除原文件")
        else:
            fail_count += 1
            print(f"      ❌ 转换失败")
    
    # 打印统计结果
    print("=" * 60)
    print("📊 转换统计:")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {fail_count} 个")
    print(f"  总计: {len(audio_files)} 个")
    
    if success_count > 0:
        print(f"\n✅ 转换完成！WAV 文件保存在原文件夹中")

# ========================================
# 5. 主程序
# ========================================

if __name__ == "__main__":
    
    # 👇👇👇 这里只需要修改这一行 👇👇👇
    # 改成你要转换的文件夹路径
    
    target_folder = r"D:\audioset_tagging_cnn\resources"  # ← 改成你的文件夹
    
    # 运行批量转换
    batch_convert(
        input_folder=target_folder,
        include_subfolders=True,   # 是否搜索子文件夹
        delete_original=True       # 是否删除原文件（建议先设为 False）
    )