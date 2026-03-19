# 星闪护佑 - 硬件端

基于 BearPi-Pico H2821 的星闪 (SLE) 测距硬件固件，实现 Client/Server 架构的无线测距功能。

**配套服务**: [本地服务端文档](../local_serve/README.md)

---

## 快速开始

```bash
# 1. 搭建开发环境
# 访问: https://www.bearpi.cn/core_board/bearpi/pico/h2821E/software/环境搭建windows_IDE.html

# 2. 替换示例代码
# 用本项目 firmware/sle_measure 文件夹替换 SDK 中的:
# fbb_bs2x\src\application\samples\products\sle_measure_dis 文件夹

# 3. 配置设备地址（见下方）

# 4. 编译烧录
# 在 Kconfig 中启用例程 sle_measure_dis，选择设备类型
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              硬件系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Client 设备 (佩戴端)                                                        │
│      ├── 设备地址: 自由设置                                                  │
│      ├── 连接 1-2 个 Server 设备                                            │
│      └── 仅需供电，不输出数据                                                │
│                                                                             │
│      ↓ SLE 无线测距                                                          │
│                                                                             │
│  Server 设备 (基站) x 6                                                      │
│      ├── 设备地址: 自由设置（不可与Client重合）                                │
│      ├── 计算与 Client 的距离                                                │
│      └── 串口输出 IQ 数据字符串                                              │
│                                                                             │
│      ↓ 串口通信                                                              │
│                                                                             │
│  本地服务端 (local_serve)                                                    │
│      ├── extract_from_serial.py 接收 IQ 数据                                │
│      ├── 计算坐标和跌倒检测                                                  │
│      └── 上传到网页服务端                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 设备配置

### Client 设备（佩戴端）

文件：`sle_measure_dis_client/sle_measure_dis_client.c`

```c
// Client 设备地址（可自定义）
uint8_t g_measure_dis_client_addr[SLE_ADDR_LEN] = { 12, 12, 12, 12, 12, 12 };

// 目标 Server 设备地址列表
// 建议连接 1-2 个 Server，否则测距频率可能不稳定
#define TARGET_SERVER_NUM 2
uint8_t g_target_server_list[TARGET_SERVER_NUM][SLE_ADDR_LEN] = {
    { 0x4, 0x4, 0x4, 0x4, 0x4, 0x4 },  // Server 1
    { 0x5, 0x5, 0x5, 0x5, 0x5, 0x5 }   // Server 2
};
```

| 配置项 | 说明 | 注意事项 |
|--------|------|----------|
| `g_measure_dis_client_addr` | Client 设备地址 | 6 字节，需唯一 |
| `TARGET_SERVER_NUM` | 连接的 Server 数量 | 建议 1-2 个 |
| `g_target_server_list` | 目标 Server 地址列表 | 需与 Server 端配置一致 |

### Server 设备（基站）

文件：`sle_measure_dis_server/sle_measure_dis_server.c`

```c
// Server 设备地址（每个基站不同）
uint8_t g_measure_dis_server_addr[SLE_ADDR_LEN] = { 5, 5, 5, 5, 5, 5 };
```

| 基站编号 | 推荐地址示例 | 位置 |
|----------|--------------|------|
| Server 1 | `{ 4, 4, 4, 4, 4, 4 }` | 左下角 |
| Server 2 | `{ 5, 5, 5, 5, 5, 5 }` | 右下角 |
| Server 3 | `{ 6, 6, 6, 6, 6, 6 }` | 右上角 |
| Server 4 | `{ 7, 7, 7, 7, 7, 7 }` | 左上角 |
| Server 5 | `{ 8, 8, 8, 8, 8, 8 }` | 中心 |
| Server 6 | `{ 9, 9, 9, 9, 9, 9 }` | 中心底部 |

## 使用方法

### 1. 环境搭建

参照 [BearPi 官方文档](https://www.bearpi.cn/core_board/bearpi/pico/h2821E/software/环境搭建windows_IDE.html) 完成开发环境搭建和 SDK 下载。

### 2. 替换示例代码

```bash
# 用本项目中的 firmware/sle_measure 文件夹替换 SDK 中的:
fbb_bs2x\src\application\samples\products\sle_measure_dis
```

### 3. 编译烧录

#### Client 设备

```bash
# 1. 在 Kconfig 中启用例程 sle_measure_dis
# 2. 设备类型选择: Client
# 3. 编译并烧录到设备
```

> **注意**: Client 端不输出 IQ 信息，只需保持供电即可

#### Server 设备

```bash
# 1. 在 Kconfig 中启用例程 sle_measure_dis
# 2. 设备类型选择: Server
# 3. 编译并烧录到设备
# 4. 通过串口连接到电脑或无线通信设备
```

> **注意**: Server 端以字符串形式输出 IQ 信息，需要与本地服务端通信

### 4. 连接本地服务端

Server 设备通过串口输出 IQ 数据，由本地服务端接收处理：

```bash
# local_serve/launch.bat 中配置串口
# Windows: COM5, COM6, COM7...
# Linux: /dev/ttyUSB0, /dev/ttyUSB1...
```

## 目录结构

```
firmware/
└── sle_measure/
    ├── sle_measure_dis_client/    # Client 设备代码
    │   └── sle_measure_dis_client.c
    └── sle_measure_dis_server/    # Server 设备代码
        └── sle_measure_dis_server.c
```

## 硬件连接

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              硬件连接                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Client 设备 (佩戴端)                                                        │
│      ├── 电池供电                                                            │
│      ├── 老人佩戴                                                          │
│      └── 与 Server 设备进行 SLE 无线测距                                     │
│                                                                             │
│  Server 设备 x 6 (基站)                                                      │
│      ├── 固定安装在房间四个角落及中心                                        │
│      ├── USB 串口 ←→ 电脑/香橙派                                            │
│      │             ↓                                                        │
│      │         本地服务端 (local_serve)                                      │
│      │             extract_from_serial.py 接收解析                          │
│      │                                                                      │
│      └── 与 Client 设备进行 SLE 无线测距                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 常见问题

**Q: 测距频率不稳定？**

减少 Client 连接的 Server 数量，建议连接 1-2 个 Server。

**Q: 串口无法接收数据？**

检查 Server 设备的串口连接和波特率设置，确保本地服务端的串口配置正确。

**Q: 设备地址冲突？**

每个设备的地址必须唯一，Client 和 Server 的地址不能重复。

## 参考资源

- [BearPi-Pico H2821 官方文档](https://www.bearpi.cn/core_board/bearpi/pico/h2821E/software/环境搭建windows_IDE.html)
- [SDK 源码](https://gitcode.com/HiSpark/fbb_bs2x)
- [本地服务端文档](../local_serve/README.md)

## 许可证

Licensed under the Apache License.
