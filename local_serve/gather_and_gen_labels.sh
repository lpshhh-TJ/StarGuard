#!/bin/bash
# ============================================
# Step 1: Gather features from source directory
# ============================================
# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source_dir="$SCRIPT_DIR/hankel_dataset"
target_dir="$SCRIPT_DIR/Distance_Measurement_Model/dataset/train/features"

# 如果目标文件夹不存在则创建
mkdir -p "$target_dir"

echo "============================================"
echo "Step 1: Gathering features..."
echo "============================================"

# 递归遍历源目录，复制所有文件到目标目录
# find 命令递归查找所有文件
find "$source_dir" -type f | while read -r file; do
    # 获取文件所在的目录
    file_dir=$(dirname "$file")
    # 跳过目标目录中的文件
    if [ "$file_dir" != "$target_dir" ]; then
        echo "Copying \"$file\" ..."
        cp "$file" "$target_dir/"
    fi
done

echo "Features gathered successfully!"
echo ""

# ============================================
# Step 2: Generate labels
# ============================================
# 激活 Conda 环境
echo "============================================"
echo "Step 2: Generating labels..."
echo "============================================"
echo "Activating Conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mindspore

# 检查是否激活成功
if [ $? -ne 0 ]; then
    echo "[Error] Failed to activate environment."
    read -p "按 Enter 键退出..."
    exit 1
fi

# 遍历源目录下的所有子目录
for dir in "$source_dir"/*/; do
    if [ -d "$dir" ]; then
        # 获取目录名称
        dir_name=$(basename "$dir")
        echo "Generating label for $dir ..."
        # 为每个目录创建独立的输出文件夹
        python "$SCRIPT_DIR/Distance_Measurement_Model/gen_labels.py" "$dir" "$SCRIPT_DIR/Distance_Measurement_NPU_hankel_plus/dataset/train/labels" --value_str="$dir_name"
    fi
done

echo ""
echo "============================================"
echo "All done! Features gathered and labels generated."
echo "============================================"
read -p "按 Enter 键退出..."
