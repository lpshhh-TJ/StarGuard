@echo off
:: 1. 激活 Conda 环境 (必须使用 call)
:: 将 'your_env_name' 替换为你的环境名称
echo Activating Conda environment...
call conda activate your_env_name

:: 设置编码为 UTF-8 防止中文路径乱码（如果需要）
chcp 65001

echo 正在启动 6 个数据处理终端...

:: 启动 6 个不同 COM 口的定位计算程序
start "Position_COM5" cmd /k python "%~dp0Recive_and_Preprocess\extract_from_serial.py" COM5
start "Position_COM17" cmd /k python "%~dp0Recive_and_Preprocess\extract_from_serial.py" COM19
start "Position_COM19" cmd /k python "%~dp0Recive_and_Preprocess\extract_from_serial.py" COM23
start "Position_COM24" cmd /k python "%~dp0Recive_and_Preprocess\extract_from_serial.py" COM24
start "Position_COM25" cmd /k python "%~dp0Recive_and_Preprocess\extract_from_serial.py" COM25
start "Position_COM26" cmd /k python "%~dp0Recive_and_Preprocess\extract_from_serial.py" COM26

:: 启动定位计算程序
start "Position_Default" cmd /k python "%~dp0Recive_and_Preprocess\calculate_position.py"

:: 启动预处理程序
start "Preprocess_Hankel" cmd /k python "%~dp0Recive_and_Preprocess\preprocess_hankel_middle.py"

echo 所有窗口已启动！
pause