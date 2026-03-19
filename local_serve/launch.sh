#!/bin/bash
# 1. 激活 Conda 环境
# 将 'your_env_name' 替换为你的环境名称
echo "Activating Conda environment..."
# 初始化 conda (根据你的 conda 安装路径可能需要调整)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate your_env_name

# 设置编码为 UTF-8 防止中文路径乱码
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

echo "正在启动 7 个数据处理终端..."

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 启动 6 个不同串口设备的定位计算程序
# 在 Linux 下串口设备通常是 /dev/ttyUSB0, /dev/ttyACM0 等
# 需要根据实际情况修改设备名称
gnome-terminal --title="Position_COM5" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/extract_from_serial.py\" /dev/ttyUSB0; exec bash" &
gnome-terminal --title="Position_COM17" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/extract_from_serial.py\" /dev/ttyUSB1; exec bash" &
gnome-terminal --title="Position_COM19" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/extract_from_serial.py\" /dev/ttyUSB2; exec bash" &
gnome-terminal --title="Position_COM24" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/extract_from_serial.py\" /dev/ttyUSB3; exec bash" &
gnome-terminal --title="Position_COM25" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/extract_from_serial.py\" /dev/ttyUSB4; exec bash" &
gnome-terminal --title="Position_COM26" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/extract_from_serial.py\" /dev/ttyUSB5; exec bash" &

# 启动定位计算程序
gnome-terminal --title="Position_Default" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/calculate_position.py\"; exec bash" &

# 启动预处理程序
gnome-terminal --title="Preprocess_Hankel" -- bash -c "python \"$SCRIPT_DIR/Recive_and_Preprocess/preprocess_hankel_middle.py\"; exec bash" &

echo "所有窗口已启动！"
read -p "按 Enter 键退出..."
