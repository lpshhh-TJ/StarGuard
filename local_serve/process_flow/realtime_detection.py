# Copyright 2026 pzhihan. Licensed under the MIT License.
#这是调用IMU模型和坐标融合判断是否跌倒的程序

import os
import sys
import time
import serial
import re
import socket
import requests
import numpy as np
import collections
import mindspore
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net
import mindspore.ops as ops

# 导入你的模型定义
# 假设 sisfall_resnet.py 在同级目录下
try:
    from sisfall_resnet import SisFallResNet
except ImportError:
    # 如果你把模型定义放在了 model 文件夹下
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
    from sisfall_resnet import SisFallResNet

# ================= 配置区域 =================
SERIAL_PORT = 'COM13'       # 串口号
BAUD_RATE = 115200          # 波特率 (根据你的ESP32设置调整)
MODEL_PATH = r'sisfall.ckpt'

# 模型参数
IN_CHANNELS = 6
NUM_CLASSES = 2             # 二分类
WINDOW_SIZE = 256           # 窗口大小
STRIDE = 32                 # 推理步长 (每收到32个新数据进行一次判断)
DEVICE_TARGET = "CPU"       # 在本地PC上通常使用CPU进行推理

# 定位融合参数
POS_HOST = 'localhost'      # 定位服务器地址
POS_PORT = 5002             # 定位服务器端口
POS_Y_THRESHOLD = 50        # cm, 低于此高度且模型判断跌倒才算跌倒
POS_SAMPLE_COUNT = 100      # 跌倒判定时采集定位数据的次数

# 报警服务器配置
ALERT_SERVER_URL = 'http://124.70.161.112:3000/update'
# ===========================================

class SerialFallDetector:
    def __init__(self):
        print("正在初始化跌倒检测系统...")

        # 定位数据缓存 (cm)
        self.current_pos_y = None
        self.last_pos_update_time = 0 # 上次更新位置的时间
        self.pos_socket = None        # 定位 socket 连接

        # 1. 初始环境
        context.set_context(mode=context.GRAPH_MODE, device_target=DEVICE_TARGET)
        
        # 2. 加载模型
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")
            
        self.net = SisFallResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
        print(f"加载模型参数: {MODEL_PATH}")
        param_dict = load_checkpoint(MODEL_PATH)
        load_param_into_net(self.net, param_dict)
        self.model = Model(self.net)
        
        # 3. 初始化数据缓冲
        # 使用 deque 自动维护固定长度的窗口
        self.buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.new_samples_count = 0
        
        # 4. 编译正则表达式
        # 匹配格式: A:0,176,198 G:-52,28,10
        # 容错：允许数字前后有空格，允许负号
        self.pattern = re.compile(r"A:\s*(-?\d+),\s*(-?\d+),\s*(-?\d+).*?G:\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)")
        
        # 5. 串口
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"成功打开串口: {SERIAL_PORT} @ {BAUD_RATE}")
        except Exception as e:
            print(f"无法打开串口 {SERIAL_PORT}: {e}")
            sys.exit(1)

        # 6. 连接定位数据 socket
        try:
            self.pos_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.pos_socket.settimeout(1.0)  # 设置超时
            self.pos_socket.connect((POS_HOST, POS_PORT))
            print(f"成功连接定位服务器: {POS_HOST}:{POS_PORT}")
        except Exception as e:
            print(f"警告: 无法连接定位服务器 {POS_HOST}:{POS_PORT}: {e}")
            print("系统将在无定位数据模式下运行")
            self.pos_socket = None

    def update_position(self, y_cm):
        """
        [定位接口] 更新当前的Y轴坐标
        :param y_cm: Y轴坐标值 (cm)。如果传入 None，则表示暂时无定位数据。
        """
        self.current_pos_y = y_cm

    def _read_position_from_socket(self):
        """
        从 socket 读取一次定位数据，返回 y 值
        预期数据格式: JSON 或 "x,y,z" 形式
        """
        if self.pos_socket is None:
            return None

        try:
            data = self.pos_socket.recv(1024).decode('utf-8').strip()
            if not data:
                return None

            # 尝试解析 JSON 格式: {"x": 1.0, "y": 2.0, "z": 3.0}
            try:
                import json
                pos = json.loads(data)
                if 'y' in pos:
                    return float(pos['y'])
            except json.JSONDecodeError:
                pass

            # 尝试解析 CSV 格式: x,y,z
            parts = data.split(',')
            if len(parts) >= 2:
                try:
                    return float(parts[1])  # y 是第二个值
                except ValueError:
                    pass

            return None
        except socket.timeout:
            return None
        except Exception:
            return None

    def _get_position_median(self, count=POS_SAMPLE_COUNT):
        """
        读取多次定位数据并返回中位数
        :param count: 采样次数
        :return: 中位数值，失败返回 None
        """
        if self.pos_socket is None:
            return None

        y_values = []
        for _ in range(count):
            y = self._read_position_from_socket()
            if y is not None:
                y_values.append(y)

        if not y_values:
            return None

        return float(np.median(y_values))

    def _send_fall_alert(self):
        """
        向服务器发送跌倒报警信息
        """
        try:
            response = requests.post(
                ALERT_SERVER_URL,
                json={'isFallen': True},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            print(f"  → 报警已发送: 状态码 {response.status_code}")
        except Exception as e:
            print(f"  → 报警发送失败: {e}")

    def parse_line(self, line):
        """解析串口行数据"""
        try:
            line_str = line.decode('utf-8', errors='ignore').strip()
            # 只有包含数据的行才处理
            if "A:" in line_str and "G:" in line_str:
                match = self.pattern.search(line_str)
                if match:
                    # 提取6个数值: Ax, Ay, Az, Gx, Gy, Gz
                    # 注意：MindSpore Tensor 需要 float32
                    values = [float(g) for g in match.groups()]
                    return values
        except Exception as e:
            # 忽略解析错误
            pass
        return None

    def predict(self):
        """执行推理"""
        # 转换数据格式
        # buffer shape: (256, 6) -> list of lists
        data_np = np.array(self.buffer, dtype=np.float32)
        
        # 模型需要 (Batch, Channels, Length) -> (1, 6, 256)
        # 当前 data_np 是 (256, 6)，需要转置并增加维度
        data_np = data_np.transpose(1, 0) # -> (6, 256)
        data_np = np.expand_dims(data_np, axis=0) # -> (1, 6, 256)
        
        input_tensor = Tensor(data_np, mindspore.float32)
        
        # 推理
        logits = self.model.predict(input_tensor)
        
        # 获取结果
        # logits shape: (1, 2)
        probs = ops.Softmax()(logits).asnumpy()[0]
        prediction = np.argmax(probs)
        
        return prediction, probs

    def start(self):
        print("开始监听数据... (按 Ctrl+C 停止)")
        print(f"模式: [传感器模型] + [定位融合 (阈值<{POS_Y_THRESHOLD}cm, 采样{POS_SAMPLE_COUNT}次中位数)]")
        print("-" * 50)

        try:
            print("正在等待传感器数据输入...")
            while True:
                if self.ser.in_waiting:
                    line = self.ser.readline()
                    data = self.parse_line(line)

                    if data:
                        self.buffer.append(data)
                        self.new_samples_count += 1

                        # 当缓冲区填满，且累积了足够多的新数据(Stride)时进行推理
                        if len(self.buffer) == WINDOW_SIZE and self.new_samples_count >= STRIDE:
                            self.new_samples_count = 0 # 重置计数器

                            # 1. 模型推理
                            pred_idx, probs = self.predict()
                            fall_conf = probs[1] * 100

                            # 2. 定位融合逻辑
                            final_decision = False
                            reason = ""
                            median_y = None

                            if pred_idx == 1:  # 模型认为跌倒
                                print(f"\n[模型检测到跌倒 (置信度:{fall_conf:.1f}%)，正在采集定位数据...]")
                                median_y = self._get_position_median(POS_SAMPLE_COUNT)

                                if median_y is not None:
                                    # 如果有定位数据：必须同时满足高度 < 阈值
                                    if median_y < POS_Y_THRESHOLD:
                                        final_decision = True
                                        reason = f"模型({fall_conf:.1f}%) + 定位中位数({median_y:.1f}cm < {POS_Y_THRESHOLD}cm)"
                                    else:
                                        final_decision = False
                                        reason = f"误报拦截: 模型判断跌倒但定位中位数({median_y:.1f}cm >= {POS_Y_THRESHOLD}cm)"
                                else:
                                    # 如果没有定位数据：完全信任模型
                                    final_decision = True
                                    reason = f"仅模型判断({fall_conf:.1f}%) [定位数据获取失败]"
                            else:
                                final_decision = False
                                reason = "正常"

                            # 3. 打印逻辑
                            status_str = "!!! 确认跌倒 !!!" if final_decision else "正常"
                            timestamp = time.strftime("%H:%M:%S", time.localtime())

                            # 如果最终判定跌倒，或者是模型检测到但被高度拦截了，都打印详细日志
                            if final_decision or (pred_idx == 1):
                                print(f"[{timestamp}] 状态: {status_str:<15} | 详情: {reason}")
                            else:
                                # 正常状态使用行覆盖刷新
                                sys.stdout.write(f"\r[{timestamp}] 状态: 正常  | ADL:{probs[0]:.2f} Fall:{probs[1]:.2f}  ")
                                sys.stdout.flush()

                            if final_decision:
                                self._send_fall_alert()

        except KeyboardInterrupt:
            print("\n停止监控。")
        finally:
            if self.ser.is_open:
                self.ser.close()
            if self.pos_socket is not None:
                self.pos_socket.close()

if __name__ == "__main__":
    detector = SerialFallDetector()
    detector.start()
