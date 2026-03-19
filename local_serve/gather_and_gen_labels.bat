@echo off
:: ============================================
:: Step 1: Gather features from source directory
:: ============================================
set "source_dir=%~dp0hankel_dataset"
set "target_dir=%~dp0Distance_Measurement_Model\dataset\train\features"

:: 如果目标文件夹不存在则创建
if not exist "%target_dir%" mkdir "%target_dir%"

echo ============================================
echo Step 1: Gathering features...
echo ============================================

:: /r 表示递归遍历所有子文件夹
:: %%f 是完整路径
for /r "%source_dir%" %%f in (*) do (
    if not "%%~dpf"=="%target_dir%\" (
        echo Copying "%%f" ...
        copy "%%f" "%target_dir%\"
    )
)

echo Features gathered successfully!
echo.

:: ============================================
:: Step 2: Generate labels
:: ============================================
:: 激活 Conda 环境 (必须使用 call)
echo ============================================
echo Step 2: Generating labels...
echo ============================================
echo Activating Conda environment...
call conda activate mindspore

:: 检查是否激活成功
if %ERRORLEVEL% neq 0 (
    echo [Error] Failed to activate environment.
    pause
    exit /b
)

for /d %%f in (%source_dir%\*) do (
    echo Generating label for %%f ...
    :: 为每个文件创建独立的输出文件夹
    python "%~dp0Distance_Measurement_Model\gen_labels.py" "%%f" "%~dp0Distance_Measurement_NPU_hankel_plus\dataset\train\labels" --value_str=%%~nxf
)
echo.
echo ============================================
echo All done! Features gathered and labels generated.
echo ============================================
pause