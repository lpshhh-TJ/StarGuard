# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.

import argparse
from pathlib import Path

def generate_fixed_labels(input_dir: str, output_dir: str, label_value: float):
    """
    读取输入文件夹的所有 .npy 文件，并在输出文件夹建立同名的 .txt 文件。
    如果 .txt 文件已存在，则跳过，不进行覆盖。
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not input_path.exists():
        print(f"错误: 输入路径不存在 -> {input_path}")
        return

    # 自动创建输出目录（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)

    created_count = 0
    skipped_count = 0

    # 遍历所有 .npy 文件
    for npy_file in input_path.glob("*.npy"):
        # 保持与 dataset.py 一致的文件名匹配逻辑
        txt_name = npy_file.stem + ".txt"
        target_file = output_path / txt_name
        
        # 检查文件是否存在，防止覆盖
        if target_file.exists():
            skipped_count += 1
            continue
            
        try:
            with target_file.open("w", encoding="utf-8") as f:
                f.write(f"{float(label_value)}\n")
            created_count += 1
        except Exception as e:
            print(f"写入文件 {target_file} 时出错: {e}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"新生成的标签数量: {created_count}")
    print(f"跳过（已存在）的数量: {skipped_count}")
    print(f"输出目录: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="安全地生成 .txt 标签文件（不覆盖已有文件）")
    parser.add_argument("input_dir", type=str, help="包含 .npy 文件的文件夹路径")
    parser.add_argument("output_dir", type=str, help="生成 .txt 标签的文件夹路径")
    parser.add_argument("--value", type=float, default=1.0, help="要写入的标签数值")
    parser.add_argument("--value_str", type=str, help="要写入的标签字符串（如果提供则覆盖数值）")
    


    args = parser.parse_args()
    
        # --- 核心转换步骤 ---
    try:
        # 将字符串 "1.0" 转换为浮点数 1.0
        file_value = float(args.value_str) 
        print(f"当前处理的数值是: {file_value}, 类型是: {type(file_value)}")
    except ValueError:
        print(f"错误：文件名 '{args.value_str}' 无法转换为浮点数")
        file_value = 0.0  # 或者设定一个默认值
    
    generate_fixed_labels(args.input_dir, args.output_dir, file_value)