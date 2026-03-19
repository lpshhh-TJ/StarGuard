# 星闪护佑 - 本地服务端

StartGuard 智护星居项目的核心服务端，负责从硬件 IQ 数据到位置信息和跌倒状态的完整处理流程。

## 快速开始

```bash
# 安装依赖（建议使用 conda 虚拟环境）
pip install -r requirements.txt

# 配置 launch.bat/launch.sh 中的虚拟环境和串口

# 运行项目
launch.bat    # Windows
launch.sh     # Linux
```

## 数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据处理流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  串口IQ数据                                                             │
│      ↓                                                                     │
│  extract_from_serial.py  提取IQ数据                                        │
│      ↓                                                                     │
│  preprocess.py              计算Hankel矩阵                                  │
│      ↓                                                                     │
│  测距神经网络              预测距离                                         │
│      ↓                                                                     │
│  calculate_position.py     计算坐标 (RANSAC + 卡尔曼滤波)                   │
│      ↓                                                                     │
│  realtime_detection.py     跌倒检测融合 (IMU数据)                           │
│      ↓                                                                     │
│  网页服务器                 显示位置 + 跌倒报警                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 目录结构

| 文件夹 | 功能 |
|--------|------|
| `process_flow/` | 主程序源码（日常运行） |
| `distance_measurement_model/` | 测距模型训练代码 |
| `sisfall2/` | IMU 跌倒检测模型训练代码 |
| `hankel_dataset/` | 测距数据集（新场景部署时采集） |
| `Recive_and_Preprocess/` | 串口数据接收与预处理 |

## 功能特性

- **实时定位** - 从 6 个基站的 IQ 数据计算用户坐标
- **跌倒检测** - 融合位置信息和 IMU 数据判断跌倒状态
- **数据采集** - 通过网页引导采集新场景的测距数据集
- **模型训练** - 包含完整的测距和跌倒检测模型训练代码

## 安装

### 建议设备

| 设备类型 | 用途 |
|----------|------|
| 香橙派 | 长期运行 |
| Windows 电脑 | 测试开发 |
| 昇腾 NPU 服务器 | 模型训练 |

### 环境要求

详见 [requirements.txt](requirements.txt)

- Python 3.x
- MindSpore
- numpy, scipy, serial 等

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/username/project.git

# 创建并激活虚拟环境
conda create -n mindspore python=3.9
conda activate mindspore

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 配置启动脚本

编辑 `launch.bat` (Windows) 或 `launch.sh` (Linux)：

```bash
# 配置虚拟环境名称
call conda activate your_env_name

# 配置串口（根据实际情况修改）
# Windows: COM5, COM6, COM7...
# Linux: /dev/ttyUSB0, /dev/ttyUSB1...
```

### 2. 启动服务

```bash
# Windows
launch.bat

# Linux
launch.sh
```

### 3. 数据集采集（可选）

新场景部署时，建议重新采集测距数据集：

1. 启动网页服务端 ([web_serve](../web_serve))
2. 网页上启动「数据集采集」功能
3. 按照指引移动到各坐标点采集数据
4. 运行 `gather_and_gen_label.bat` 汇总数据集
5. 重新训练测距模型

## 配置

### 定位算法配置

| 配置文件 | 可配置项 |
|----------|----------|
| `process_flow/calculate_position.py` | RANSAC 参数、卡尔曼滤波参数 |
| `process_flow/multipath_resistant.py` | 抗多径算法参数 |

### 模型配置

| 配置文件 | 可配置项 |
|----------|----------|
| `realtime_detection.py` | 跌倒检测模型参数、融合策略 |
| `distance_measurement_model/model.py` | 测距模型结构 |
| `distance_measurement_model/src/main.py` | 测距模型训练参数 |

## 常见问题

**Q: 串口无法连接？**

检查串口设备名称并确保设备已连接：
```bash
# Windows
mode COM5

# Linux
ls /dev/ttyUSB*
```

**Q: 模型文件找不到？**

确保模型文件 `distance.ckpt` 和 `sisfall.ckpt` 在正确的位置，或修改代码中的 `MODEL_PATH`。

## 相关链接

- [网页服务端文档](../web_serve/README.md)
- [测距模型训练](distance_measurement_model/README.md)
- [跌倒检测模型训练](sisfall2/README.md)

## 许可证

Licensed under the MIT License.



