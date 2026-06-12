# IQ 数据处理 + AI 测距系统（STM32 端）

> 基于小熊派 BS21E + STM32H747I-DISCO 的端侧 AI 距离测量系统 — SLE 信道探测 IQ 数据采集 → Hankel 矩阵变换 → 神经网络推理 → LCD 实时显示

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-STM32H747I--DISCO-blue)](https://www.st.com/en/evaluation-tools/stm32h747i-disco.html)
[![AI Framework](https://img.shields.io/badge/AI-STM32Cube.AI-brightgreen)](https://www.st.com/en/embedded-software/x-cube-ai.html)

---

## 简介

本系统是 [StarGuard 智护星居](../README.md) 项目的独立子模块，将原本运行在 PC 服务端的 AI 测距模型移植到 **STM32H747I-DISCO** 开发板上，实现纯嵌入式端到端的距离测量。

**数据源**：小熊派 BS21E（HiSilicon BS21E，RISC-V @ 250MHz）通过 SLE（星闪低功耗）无线协议进行 Channel Sounding 信道探测，采集原始 IQ 数据，经由 UART 以二进制帧格式发送给 STM32。

**处理流程**：STM32 接收 IQ 帧 → 解析 692 字节二进制帧 → 构建 40×40 复数 Hankel 矩阵 → 归一化 → AI 推理（40M MACs）→ exp 反变换 → 在 4.3" TFT LCD 上显示距离。

### 适用场景

- 低成本的嵌入式 AI 测距部署（无需 PC / 服务器）
- 室内人员定位、跌倒检测的测距预处理
- STM32Cube.AI 部署 MindSpore 模型的端到端参考设计

---

## 硬件平台

| 组件 | 型号/规格 | 作用 |
|------|----------|------|
| **发送端 MCU** | 小熊派 BS21E — HiSilicon BS21E (RISC-V @ 250MHz) | SLE 信道探测，采集 IQ 数据 |
| **接收端 MCU** | STM32H747XIH6 (Cortex-M7 @ 480MHz + Cortex-M4 @ 240MHz) | 数据处理 + AI 推理 |
| **无线协议** | SLE (Super Low Energy), 2.4GHz, Channel Sounding | 测距信号交互 |
| **板载内存** | 64MB SDRAM (IS42S32800J) | AI 激活缓冲区 + LCD 帧缓冲 |
| **外部存储** | 128MB QSPI NOR Flash (MT25TL01G) | 存储 405KB 模型权重 |
| **显示屏** | 4.3" TFT LCD (NT35510 / OTM8009A, 480×272) | 结果显示 |
| **数据接口** | UART0 (GPIO11/12, 115200bps) → UART8 (Arduino D0/D1, 115200bps) | IQ 数据传输 |
| **调试接口** | USART1 (STLink VCP, 115200bps) | LC_PRINT 调试日志 |

> **注意**：UART8 专门用于 IQ 二进制帧接收，USART1 专用于调试日志输出，双 UART 隔离互不干扰。

---

## 系统架构

```
┌──────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│      小熊派 BS21E (发送端)            │    │             STM32H747I-DISCO (接收端)           │
│                                      │    │                                              │
│ ┌─────────────────────────────────┐  │    │   UART0 ──→ 环形缓冲区 ─TIM6 ISR──→ IQ 帧解析 │
│ │ SLE Channel Sounding 采集 IQ 数据│  │    │  (GPIO11/12)  (4096字节)   (200ms)            │
│ │                                 │  │    │                             ↓                 │
│ │ Local IQ (本端) ─┐              │  │    │                       IQ Pair 组装             │
│ │ Remote IQ (对端) ─┤              │  │    │                             ↓                 │
│ │                 ↓               │  │    │                    计算 Hankel 矩阵             │
│ │          匹配时间戳对            │  │    │                  (40×40 复数, 3200 floats)     │
│ │                 ↓               │  │    │                             ↓                 │
│ │   二进制帧打包 (692B × 2帧)     │──┼──→ │                      归一化 ÷8380418            │
│ │   ┌─ 帧0: Local IQ (行标志=0)  │  │    │                             ↓                 │
│ │   └─ 帧1: Remote IQ (行标志=1) │  │    │               ┌─→ AI 推理 (STM32Cube AI)        │
│ │                                 │  │    │               │   (MAC: 40M, 权重: 405KB)      │
│ │   帧格式:                        │  │    │               │   (推理时间: ~700ms)            │
│ │   0xAA 0xBB + 行标志 +          │  │    │               │         ↓                      │
│ │   7×int32 头部 + 79×IQ对 +      │  │    │               │   exp 反变换 → 距离(m)          │
│ │   XOR 校验 = 692 字节            │  │    │               │         ↓                      │
│ └─────────────────────────────────┘  │    │               │   LCD 显示结果                  │
│                                      │    │               │   (相位图 + 距离 + 统计)         │
│ 关闭所有调试串口输出                  │    │               └─────────────────────────────────│
│ (osal_printk_control → false)        │    │                                                    │
│                                      │    │  USART1 (STLink) → 串口调试日志 (LC_PRINT)         │
└──────────────────────────────────────┘    └──────────────────────────────────────────────────┘
```

### 数据流详解

1. **IQ 采集**：BS21E 通过 SLE 协议与对端设备完成 Channel Sounding 流程，分别在本地和对端采集 80 组 I/Q 原始数据（实际使用 79 对）
2. **时间戳匹配**：Local IQ 和 Remote IQ 到达时间不同，通过时间戳匹配（±20ms 容差）配对
3. **帧打包**：匹配成功后将 Local/Remote IQ 数据分别打包为 692 字节二进制帧，通过 UART0 发送
4. **STM32 接收**：UART8 RXNE 中断将每个字节存入 4096 字节环形缓冲区
5. **帧解析**：TIM6 中断每 200ms 扫描环形缓冲区，检测帧头 `0xAA 0xBB`、解析 IQ 数据、验证 XOR 校验和
6. **IQ 配对**：连续收到 Line1（行标志=0）和 Line2（行标志=1）后组成完整 IQ 样本
7. **Hankel 变换**：从 316 个 int16 IQ 值构建 40×40 复数 Hankel 矩阵（3200 个 float）
8. **AI 推理**：归一化后送入神经网络，输出 log 空间距离值
9. **结果显示**：exp 反变换得到实际距离（米），更新 LCD 相位图和仪表盘

---

## 目录结构

```
dis_measure_STM32/
├── BS21E/                              # 小熊派 BS21E 发送端固件
│   └── sle_measure_dis/               # SLE 测距项目
│       ├── sle_measure_dis.c          # 任务入口，关闭调试输出
│       ├── sle_measure_dis.code-workspace
│       ├── CMakeLists.txt
│       ├── Kconfig
│       ├── sle_measure_dis_client/    # Client 端（对端）代码
│       │   ├── sle_measure_dis_client.c
│       │   ├── sle_measure_dis_client.h
│       │   ├── sle_measure_dis_client_slem.c
│       │   └── sle_measure_dis_client_slem.h
│       └── sle_measure_dis_server/    # Server 端（本端）代码
│           ├── sle_measure_dis_server.c           # 服务端初始化
│           ├── sle_measure_dis_server.h           # 数据结构定义
│           ├── sle_measure_dis_server_alg.c       # IQ 处理 + 帧打包 + UART0 配置
│           ├── sle_measure_dis_server_alg.h       # 算法数据结构
│           ├── sle_measure_dis_server_adv.c       # 广播配置
│           └── sle_measure_dis_server_adv.h
│
├── Model_SourceCode(Python)/           # Python 模型设计与训练
│   └── dm_model_stm32/
│       ├── gen_labels.py              # 标签生成工具
│       ├── logs/                      # 训练日志
│       └── src/
│           ├── main.py                # 训练入口（参数配置 + 训练 + 评估）
│           ├── model.py               # 模型定义（MultiScaleConv + Network）
│           ├── dataset.py             # 数据集加载与增强
│           ├── predict.py             # 推理预测
│           ├── utils.py               # 工具函数
│           └── export_onnx.py         # MindSpore → ONNX 导出
│
└── STM32H747I-DISCO/                   # STM32 接收端项目 (Keil MDK)
    └── disMeasure/
        ├── CM4/                       # Cortex-M4 核代码
        │   └── Core/
        │       ├── Inc/               # main.h, stm32h7xx_hal_conf.h, stm32h7xx_it.h
        │       └── Src/               # main.c, stm32h7xx_hal_msp.c, stm32h7xx_it.c
        ├── CM7/                       # Cortex-M7 核代码（主处理核）
        │   ├── AI/App/               # AI 应用层（核心代码）
        │   │   ├── app_x-cube-ai.c   # 主处理逻辑：归一化、Hankel、推理、LCD、流控
        │   │   ├── app_x-cube-ai.h   # 函数导出
        │   │   ├── app_config.h      # AI 配置（Cache、外部 RAM 开关）
        │   │   ├── bsp_ai.h          # 板级支持
        │   │   ├── network.c/h       # STM32Cube AI 生成的网络模型
        │   │   ├── network_data.c/h  # 权重数据（QSPI Flash 映射）
        │   │   ├── network_details.h # 网络结构描述
        │   │   ├── network_weights.c/h # 权重加载接口
        │   │   ├── user_init.c/h     # 权重初始化
        │   │   ├── multi_test.h      # 3 个自测样本数据
        │   │   ├── image_320x240_argb8888.h
        │   │   └── life_augmented_argb8888.h
        │   └── Core/                 # CM7 核心初始化
        │       ├── Inc/              # main.h（含 LCD 帧缓冲区地址定义）
        │       └── Src/              # main.c, stm32h7xx_it.c, stm32h7xx_hal_msp.c
        ├── Common/                   # 双核共享代码
        │   ├── Inc/
        │   └── Src/
        └── MDK-ARM/                  # Keil 项目文件
            ├── stm32h747xx_flash_CM7.sct  # 链接脚本（RAM_D1 分配）
            └── disMeasure.uvprojx         # 项目文件
```

---

## 二进制帧协议

BS21E 通过 UART0 发送的每一帧为 **692 字节**，格式如下：

| 偏移 | 大小 | 字段 | 说明 |
|:--|:--|:--|:--|
| 0 | 2 | 帧头 | 固定 `0xAA 0xBB` |
| 2 | 1 | 行标志 | `0` = Local IQ, `1` = Remote IQ |
| 3 | 28 | 头部 | 7 × int32 小端序 |
| 31 | 316 | IQ 数据 | 79 对 I/Q 交错，int16 小端序 |
| 347 | 344 | 保留 | 填充 `0` |
| 691 | 1 | 校验和 | Byte[0] ~ Byte[690] 逐字节 XOR |
| **总计** | **692** | | |

一次完整的测量产生两帧（Local + Remote），共 316 个 int16 值。STM32 端将这两帧的 IQ 数据合并排列，作为模型输入。

### 头部字段

| 偏移 | 索引 | 字段名 | 值/来源 |
|:--|:--|:--|:--|
| 3 | [0] | 固定标识 | `g_measure_dis_server_addr[0]` = 1 |
| 7 | [1] | 固定值 | `num` = 0 |
| 11 | [2] | 序列号 | `report->timestamp_sn` |
| 15 | [3] | 时间戳(ms) | `uapi_tcxo_get_ms()` |
| 19 | [4] | 服务器 ID | `g_measure_dis_server_addr[0]` = 1 |
| 23 | [5] | 采样数 | `report->samp_cnt` = 80 |
| 27 | [6] | RSSI/标志 | `report->rssi` |

---

## 功能特性

| 功能 | 描述 | 状态 |
|------|------|:----:|
| **UART 数据接收** | 通过 UART8 (Arduino D0/D1) 接收 692 字节二进制帧，含同步头 (0xAA 0xBB) 和 XOR 校验 | ✅ |
| **IQ 帧解析** | 解析两行 IQ 数据 (Line1/Line2)，配对组成完整样本 | ✅ |
| **Hankel 矩阵计算** | 从 316 个 int16 IQ 值构建 40×40 复数 Hankel 矩阵 | ✅ |
| **输入归一化** | 原始数据 ÷ 8380418.0 映射到 [0,1] 范围 | ✅ |
| **AI 模型推理** | 部署 40M MACs 轻量神经网络，输出 log 空间距离值 | ✅ |
| **输出反归一化** | `exp(raw) - 0.001` 截断到 [0, 10] 米 | ✅ |
| **相位图显示** | 使用 `atan2f` 计算相位、以灰度图方式在 LCD 显示 40×40 矩阵 | ✅ |
| **右侧仪表盘** | 显示距离值、配对编号、处理时间、状态 | ✅ |
| **UART 统计** | 实时显示接收字节数、帧数、配对数和 CRC 错误数 | ✅ |
| **启动自检** | 上电自动用 3 个训练样本测试推理是否正常 | ✅ |
| **流控保护** | 处理中自动丢弃新帧，防止数据覆盖 | ✅ |
| **中断挂起** | 推理期间暂停 UART + TIM6 中断，防止 QSPI 访问死锁 | ✅ |
| **双 UART 隔离** | USART1 用于调试日志，UART8 用于 IQ 数据，互不干扰 | ✅ |
| **权重 SDRAM 缓存** | 启动时将 4.9MB 权重从 QSPI 拷贝到 SDRAM，加速推理 | ✅ |

---

## 性能指标

| 指标 | 数值 |
|------|------|
| 计算量 | **40M MACs** |
| 权重大小 | **405 KB** |
| 推理时间（QSPI 直读） | **~700 ms** |
| 推理时间（SDRAM 缓存） | **~100 ms**（预留优化） |
| 预测范围 | **0–10 m** |
| 输入归一化 | 线性缩放 ÷8380418 |
| 数据帧大小 | 692 字节/帧 |
| 单次测量数据 | 2 帧 / 316 个 int16 IQ 对 |

---

## 快速开始

### 前置要求

| 组件 | 要求 |
|------|------|
| 硬件 | 小熊派 BS21E × 2 + STM32H747I-DISCO × 1 |
| BS21E SDK | HiSilicon BS21E SDK + 交叉编译工具链 |
| STM32 IDE | Keil MDK-ARM (μVision) |
| AI 工具 | STM32Cube.AI (X-CUBE-AI) |
| Python | 3.9+ + MindSpore（模型训练用） |

### BS21E 发送端烧录

```bash
# 在 BS21E SDK 环境中配置项目
cd BS21E/sle_measure_dis

# 配置为 Server 端（数据发送端）
# 在 Kconfig 中启用 CONFIG_SAMPLE_SUPPORT_SLE_MEASURE_DIS_SERVER

# 编译并烧录
# 具体步骤参考 BS21E SDK 官方文档
```

### STM32 接收端烧录

1. 打开 `STM32H747I-DISCO/disMeasure/MDK-ARM/disMeasure.uvprojx`
2. 确认 QSPI Flash 已烧录模型权重（`network_data.c` 编译后默认存入 QSPI）
3. 编译 CM7 工程并下载到开发板
4. 连接小熊派 BS21E 的 UART0（GPIO11/12）到 STM32 的 UART8（Arduino D0/D1）
5. 上电后 LCD 将依次显示：
   - 启动自检结果（3 个测试样本的推理）
   - 等待 IQ 数据
   - 接收到数据后显示距离和相位图

### 模型训练（可选）

```bash
cd Model_SourceCode(Python)/dm_model_stm32/src
python main.py \
  --feature-dir ../dataset/train/features \
  --label-dir ../dataset/train/labels \
  --epochs 100 \
  --augment
```

---

## Python 模型设计

### 模型架构

模型采用 **多尺度卷积 + 全连接回归** 的混合结构，输入为 2×40×40 的 Hankel 矩阵（实部 + 虚部），输出为单一距离值（log 空间）。

```
输入 (2, 40, 40)
    │
    ├─ MultiScaleConv (1×1 + 3×3) → 24 通道
    ├─ ReLU
    ├─ Conv2d 24→48 (3×3) + BN + ReLU
    ├─ AvgPool 2×2 → 20×20
    │
    ├─ Conv2d 48→48 (3×3) + BN + ReLU
    ├─ Conv2d 48→64 (3×3) + BN + ReLU
    ├─ AvgPool 2×2 → 10×10
    │
    ├─ Conv2d 64→64 (3×3) + BN + ReLU
    ├─ GlobalAvgPool → 64-d 特征向量
    │
    ├─ Dense(64→64) + ReLU + Dropout
    ├─ Dense(64→16) + ReLU + Dropout
    └─ Dense(16→1) → log(距离)
```

### 关键设计——MultiScaleConv

并行的 1×1 和 3×3 卷积同时提取局部细节和邻域上下文，输出在通道维拼接。

```python
class MultiScaleConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def construct(self, x):
        c1 = self.conv1(x)       # 局部细节
        c3 = self.conv3(x)       # 邻域上下文
        return self.concat((c1, c3))  # → 2×out_channels
```

### 模型轻量化（417M → 40M MACs）

原版模型面向 Ascend NPU 设计（417M MACs），针对 STM32H7 重设计后降至 **40M MACs**（↓ 90%）：

| 策略 | 说明 |
|------|------|
| 去掉 5×5 分支 | 2 通道输入用大核感受野收益极低 |
| 通道数砍到 1/4 | 每层通道数等比缩减 |
| 全连接层瘦身 | 中间层从 256→64 降到 64→16 |
| 输入输出规格不变 | 仍为 2×40×40 → 1，新旧模型无缝替换 |

---

## 部署流程

完整的端到端 AI 部署链路：

```
MindSpore (训练) → ONNX → STM32Cube AI (量化/优化) → C 代码 → MDK-ARM (编译) → 烧录
```

### ONNX 导出

```python
# Model_SourceCode(Python)/dm_model_stm32/src/export_onnx.py
mindspore.export(net, dummy_input, file_name="model.onnx", file_format="ONNX")
```

### 注意：名字清洗

MindSpore 导出的 ONNX 节点名包含 `/` 和 `-`（如 `features-SequentialCell/0-MultiScaleConv/Concat-op0`），ST Edge AI Core 用这些名字生成 C 函数名，导致 Keil MDK 编译失败。后处理脚本会自动替换为 C 兼容字符：

```
3-Default/features-SequentialCell/0-MultiScaleConv/Concat-op0
→ 3_Default_features_SequentialCell_0_MultiScaleConv_Concat_op0
```

### 归一化参数

保存在 `model_stm32.json` 中，C 端相应地做归一化与反归一化：

```c
/* 输入归一化 */
input[i] /= 8380418.0f;

/* 输出解码 */
float distance = expf(model_output * 1.0f) - 0.001f;
/* 截断到 [0, 10] */
if (distance < 0.0f)  distance = 0.0f;
if (distance > 10.0f) distance = 10.0f;
```

---

## 技术要点与踩坑记录

### QSPI DTR 模式 + D-Cache 导致总线死锁 🔴

**现象**：首次调用 `stai_network_run()` 时系统完全卡死。

**根因**：QSPI 配置为 DTR (Double Transfer Rate) 内存映射模式，D-Cache 的缓存行填充会触发 AHB burst 读取，DTR 模式下 burst 读取无法正确完成，导致 AHB 总线无限等待。

**解决方案**：使用 **DTR + Cacheable** 模式——缓存命中时避免重复 QSPI 访问，推理速度从 ~7s 降到 ~700ms。

```c
MPU_InitStruct.IsCacheable = MPU_ACCESS_CACHEABLE;
MPU_InitStruct.IsBufferable = MPU_ACCESS_BUFFERABLE;
```

### AI 输出恒为 1263535.75 🔴

**根因**：输入数据未经过归一化。训练时使用 `input_scale = 8380418.0` 做线性缩放，但 C 代码直接喂原始值（百万级范围）。

### 数据到达速率过快导致崩溃 🔴

**根因**：数据输入速率（2 帧/秒）超过推理速率（~7s/次）。引入 `new_pair_ready` 流控标志解决。

### TIM6 中断与 LCD 操作竞争 🟡

**根因**：TIM6 ISR 更新统计文字时与主循环的 `draw_phase_map()` 冲突，通过减少不必要的 LCD 写操作解决。

### 链接资源不足 🟡

**根因**：模型 `network.c` 产生大量中间数据段，128KB DTCM 不足。通过修改 scatter 文件将 RW/ZI 段路由到 512KB RAM_D1 (AXI SRAM) 解决。

```ld
RW_RAM_D1 0x24000000 0x80000 {
    .ANY (+RW +ZI)
    *(.AI_RAM_D1)
}
```

---

## 未来改进方向

### 性能优化

| 方向 | 方案 | 预期提升 |
|------|------|---------|
| **权重预加载到 SDRAM** | 启动时将权重拷贝到 SDRAM (cacheable)，已实现 | ~700ms → **~100ms** |
| **DMA2D 双缓冲** | 硬件后台渲染相位图到 off-screen buffer | 消除撕裂 |
| **LTDC 阴影层** | VBlank 期间切换 framebuffer | 完美无撕裂 |

### CM4 核利用

当前 CM4 核几乎闲置，可以分担：

- **IQ 帧解析**：TIM6 ISR 中的帧解析移至 CM4
- **Hankel 计算**：CM4 计算 40×40 复数矩阵
- **LCD 渲染**：CM4 负责相位图渲染
- **UART 管理**：CM4 管理 UART8 环形缓冲区

### 功能扩展

- **实时校准**：通过 USART1 更新归一化参数
- **数据记录**：推理结果保存到 microSD 卡
- **无线传输**：通过 ESP8266/ESP32 上传距离数据
- **GUI 升级**：坐标轴刻度、历史曲线图

### 模型迭代

- INT8 量化推理（进一步减少计算量）
- 更浅的网络结构（目标 < 10M MACs）
- 继续训练缩小验证集误差（当前 MAE ~0.15m）

---

## 依赖与参考

- [STM32H747I-DISCO 官方页面](https://www.st.com/en/evaluation-tools/stm32h747i-disco.html)
- [X-CUBE-AI (STM32Cube.AI)](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [MindSpore 深度学习框架](https://www.mindspore.cn/)
- [HiSilicon BS21E / 小熊派](https://www.bearpi.cn/)
- [StarGuard 智护星居 — 主项目](../README.md)

---

## 许可证

本项目采用 **MIT 协议** 开源。

- [Apache License & MIT License](../LICENSE)

---

## 团队

**StarGuard 智护星居**
