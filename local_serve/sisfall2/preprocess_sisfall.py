### Copyright 2026 pzihan. Licensed under the MIT License.

import os
import numpy as np
import glob
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = r'd:\Projects\ModolartsProjects\sisfall\data\SisFall_dataset'  # 数据集根目录
OUTPUT_DIR = 'processed_data'       # 输出目录
WINDOW_SIZE = 256                   # 窗口大小 (约1.28秒 @ 200Hz)
STRIDE = 128                        # 滑动步长 (50% 重叠)
SELECTED_COLUMNS = [0, 1, 2, 3, 4, 5] # 只保留前6列 (ADXL345: 0-2, ITG3200: 3-5)
# ===========================================

def get_label_from_filename(filename):
    """
    根据文件名解析标签
    文件名格式: D01_SA01_R01.txt 或 F01_SA01_R01.txt
    返回: (activity_id, is_fall)
    activity_id: 0-33 (对应 D01-D19, F01-F15)
    is_fall: 0 (ADL) 或 1 (Fall)
    """
    base = os.path.basename(filename)
    code = base.split('_')[0] # 'D01', 'F01' 等
    
    activity_type = code[0]   # 'D' or 'F'
    activity_num = int(code[1:]) # 1, 2, ...
    
    if activity_type == 'D':
        # D01 -> 0, D02 -> 1, ..., D19 -> 18
        label_id = activity_num - 1
        is_fall = 0
    elif activity_type == 'F':
        # F01 -> 19, ..., F15 -> 33
        label_id = (activity_num - 1) + 19
        is_fall = 1
    else:
        return None, None
        
    return label_id, is_fall

def preprocess_sisfall():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_data = []
    all_labels_detailed = [] # 34分类
    all_labels_binary = []   # 2分类 (跌倒/非跌倒)

    # 遍历所有子文件夹 (SA01-SE15)
    search_path = os.path.join(DATA_ROOT, "*", "*.txt")
    files = glob.glob(search_path)
    
    if not files:
        print(f"错误: 在 {DATA_ROOT} 下未找到 .txt 文件，请检查路径。")
        return

    print(f"找到 {len(files)} 个数据文件，开始处理...")

    for file_path in tqdm(files):
        # 1. 解析标签
        label_id, is_fall = get_label_from_filename(file_path)
        if label_id is None:
            continue

        try:
            # 2. 读取数据 (跳过Readme之类的非数据行，SisFall通常没有header，但最后一行可能是空的)
            # 原始数据分隔符可能是逗号或空格，SisFall Readme说是txt，通常是逗号分隔
            # 注意：SisFall原始文件中的数据格式可能是 "  12,  34, ..."
            raw_data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    # 替换可能的分隔符
                    parts = line.replace(';', ',').split(',')
                    try:
                        # 只取前9列中的前6列，转为float
                        # 某些行可能为空或格式错误
                        if len(parts) >= 6:
                            nums = [float(p) for p in parts[:6]] # 取前6列
                            raw_data.append(nums)
                    except ValueError:
                        continue
            
            raw_data = np.array(raw_data)
            
            # 如果文件数据太少，跳过
            if raw_data.shape[0] < WINDOW_SIZE:
                continue

            # 3. 数据清洗 (只保留 SELECTED_COLUMNS)
            # 代码中已经通过 parts[:6] 实现了筛选，这里确保numpy array正确
            # raw_data shape: (N_samples, 6)

            # 4. 滑动窗口切分
            num_samples = raw_data.shape[0]
            for start in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                segment = raw_data[start:end, :] # (256, 6)
                
                # 转置为 (6, 256) 适配 Conv1d 的 (Channel, Length) 格式，或者保持并在模型中转置
                # 这里我们保持 (Channel, Length) 格式方便查看
                segment = segment.transpose(1, 0) # (6, 256)
                
                all_data.append(segment)
                all_labels_detailed.append(label_id)
                all_labels_binary.append(is_fall)

        except Exception as e:
            print(f"处理文件 {file_path} 出错: {e}")

    # 转换为numpy数组
    X = np.array(all_data, dtype=np.float32)
    y_detail = np.array(all_labels_detailed, dtype=np.int32)
    y_binary = np.array(all_labels_binary, dtype=np.int32)

    print(f"\n处理完成!")
    print(f"输入数据形状 X: {X.shape} (样本数, 通道数6, 窗口长度)")
    print(f"标签数据形状 y: {y_detail.shape}")

    # 保存
    np.save(os.path.join(OUTPUT_DIR, 'sisfall_X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'sisfall_y_detail.npy'), y_detail)
    np.save(os.path.join(OUTPUT_DIR, 'sisfall_y_binary.npy'), y_binary)
    print(f"数据已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_sisfall()
