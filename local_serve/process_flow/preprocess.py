# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.
#这是把IQ数据预处理得到Hankel矩阵，然后把Hankel矩阵输入神经网络预测得到距离的程序

import socket
import json
import argparse
import sys
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime

# MindSpore 相关导入
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

# 导入最新网络定义

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ds_model import Network


def compute_covariance_matrix(part1, part2):
    """
    计算 40xK 汉克尔矩阵 (复数)
    """
    p1 = np.array(part1, dtype=np.float64)
    p2 = np.array(part2, dtype=np.float64)

    z_local = p1[0::2] + 1j * p1[1::2]
    z_remote = p2[0::2] + 1j * p2[1::2]
    z_diff = z_remote * np.conj(z_local)

    N, M = len(z_diff), 40
    K = N - M + 1

    C = np.zeros((M, K), dtype=complex)
    for m in range(M):
        C[m, :] = z_diff[m : m + K]

    return C

def load_scaling_config(ckpt_path):
    """
    从 model.json 自动加载缩放因子
    """
    config_path = Path(ckpt_path).with_suffix(".json")
    scales = {"input_scale": 1.0, "label_scale": 1.0}
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                scales["input_scale"] = config.get("input_scale", 1.0)
                scales["label_scale"] = config.get("label_scale", 1.0)
                print(f"[Config] Loaded input_scale: {scales['input_scale']}, label_scale: {scales['label_scale']}")
        except Exception as e:
            print(f"[Warning] Config read error: {e}")
    return scales

def init_predictor(ckpt_path, use_cnn_gru=False):
    """
    初始化推理环境 (强制 CPU)

    Args:
        use_cnn_gru: 是否使用 CNN-GRU 模型（需要40帧序列）
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    if use_cnn_gru:
        raise ValueError("CNN-GRU model is not available in this project. Please set use_cnn_gru=False or use a model checkpoint trained with CNN-GRU architecture.")

    net = Network()
    print("[Model] Using CNN-only (single frame)")

    param_dict = load_checkpoint(str(ckpt_path))

    # 加入诊断逻辑：检查参数加载完整性
    param_not_load, _ = load_param_into_net(net, param_dict)
    if param_not_load:
        print(f"[Warning] Parameters not fully loaded: {param_not_load}")
    else:
        print("[Info] All parameters loaded successfully.")

    net.set_train(False)
    return net


class ServerFrameBuffer:
    """为每个 server_id 维护的历史帧缓冲区"""
    def __init__(self, max_size=40):
        self.max_size = max_size
        self.frames = []  # 存储帧数据 (real_part, imag_part)

    def add_frame(self, real_part, imag_part):
        """添加新帧，返回是否已满"""
        self.frames.append((real_part, imag_part))
        if len(self.frames) > self.max_size:
            self.frames.pop(0)  # 移除最老的帧
        return len(self.frames) == self.max_size

    def get_sequence(self):
        """获取序列数据，返回 (max_size, 2, 40, 40)"""
        if len(self.frames) < self.max_size:
            return None

        # 构建序列
        sequence = []
        for real_part, imag_part in self.frames:
            sequence.append(np.stack([real_part, imag_part], axis=0))
        return np.stack(sequence, axis=0)  # (40, 2, 40, 40)

    def is_ready(self):
        """检查缓冲区是否已满"""
        return len(self.frames) >= self.max_size

def do_inference(net, cov_matrix, input_scale, label_scale, inference_lock=None, use_cnn_gru=False):
    """
    推理逻辑：还原物理距离
    使用线程锁保护多线程推理，防止 MindSpore 递归错误

    Args:
        use_cnn_gru: 是否使用 CNN-GRU 模型（需要序列输入）
    """
    if inference_lock:
        inference_lock.acquire()
    try:
        real_part = (cov_matrix.real / input_scale).astype(np.float32)
        imag_part = (cov_matrix.imag / input_scale).astype(np.float32)

        if use_cnn_gru:
            # CNN-GRU 需要序列输入，这里返回预处理后的帧
            # 调用者需要先收集40帧后再推理
            return real_part, imag_part
        else:
            # 单帧推理（原有逻辑）
            input_data = np.stack([real_part, imag_part], axis=0)
            input_tensor = Tensor(input_data).expand_dims(0)
            output = net(input_tensor)
            pred_log = float(output.asnumpy()[0, 0])

            eps = 1e-3
            distance = np.exp(pred_log * label_scale) - eps
            return float(np.clip(distance, 0.0, 10.0))
    finally:
        if inference_lock:
            inference_lock.release()


def do_sequence_inference(net, sequence, input_scale, label_scale, inference_lock=None):
    """
    CNN-GRU 序列推理

    Args:
        sequence: (40, 2, 40, 40) 序列数据
    """
    if inference_lock:
        inference_lock.acquire()
    try:
        # sequence 已经是 (40, 2, 40, 40) 格式
        # input_scale 已在添加帧时应用过
        input_tensor = Tensor(sequence).expand_dims(0)  # (1, 40, 2, 40, 40)
        output = net(input_tensor)
        pred_log = float(output.asnumpy()[0, 0])

        eps = 1e-3
        distance = np.exp(pred_log * label_scale) - eps
        return float(np.clip(distance, 0.0, 10.0))
    finally:
        if inference_lock:
            inference_lock.release()

def handle_client(client_socket, addr, predictor, scales, forward_manager, auto_id_counter, id_lock, inference_lock, use_cnn_gru=False, use_median=True):
    """
    处理单个客户端连接的线程函数
    限制处理队列上限为20，防止数据积压

    Args:
        use_cnn_gru: 是否使用 CNN-GRU 模型（为每个 server_id 维护40帧序列）
        use_median: 是否使用中位数平滑（默认 True）
    """
    # 为每个客户端维护独立的预测结果缓存（仅当 use_median=True 时使用）
    prediction_buffer = []
    WINDOW_SIZE = 10
    MAX_QUEUE_SIZE = 20  # 最大处理队列上限

    # 为每个 server_id 维护独立的历史帧缓冲区（CNN-GRU 模式）
    server_buffers = {}  # {server_id: ServerFrameBuffer}

    client_id = f"{addr[0]}:{addr[1]}"
    print(f"[+] Client {client_id} connected, thread started")

    # 设置socket为非阻塞模式
    client_socket.setblocking(False)

    try:
        while True:
            # 一次性读取所有可用数据，但只处理最新的20条
            lines_buffer = []

            while True:
                try:
                    chunk = client_socket.recv(4096).decode('utf-8', errors='ignore')
                    if not chunk:
                        break
                    lines_buffer.append(chunk)
                except BlockingIOError:
                    # 没有更多数据
                    break
                except Exception:
                    # 连接断开
                    break

            if not lines_buffer:
                # 没有收到任何数据，短暂休眠
                time.sleep(0.01)
                continue

            # 合并所有收到的数据
            all_data = ''.join(lines_buffer)
            lines = all_data.split('\n')

            # 只保留最后MAX_QUEUE_SIZE条有效数据
            valid_lines = [l.strip() for l in lines if l.strip()]
            if len(valid_lines) > MAX_QUEUE_SIZE:
                skipped = len(valid_lines) - MAX_QUEUE_SIZE
                if skipped > 0:
                    print(f"[Client:{client_id}] Skipped {skipped} backlog messages, processing latest {MAX_QUEUE_SIZE}")
                valid_lines = valid_lines[-MAX_QUEUE_SIZE:]

            # 处理筛选后的数据
            for line in valid_lines:
                try:
                    req = json.loads(line)
                    p1, p2 = req.get("part1", []), req.get("part2", [])
                    sid = req.get("server_id", "unknown")

                    if len(p1) == 158 and len(p2) == 158:
                        cov_matrix = compute_covariance_matrix(p1, p2)

                        if use_cnn_gru:
                            # CNN-GRU 模式：为每个 server_id 维护独立的历史窗口
                            if sid not in server_buffers:
                                server_buffers[sid] = ServerFrameBuffer(max_size=40)

                            buffer = server_buffers[sid]
                            real_part = (cov_matrix.real / scales["input_scale"]).astype(np.float32)
                            imag_part = (cov_matrix.imag / scales["input_scale"]).astype(np.float32)

                            is_ready = buffer.add_frame(real_part, imag_part)

                            if is_ready:
                                # 缓冲区已满，进行序列推理
                                sequence = buffer.get_sequence()
                                prediction = do_sequence_inference(predictor, sequence, scales["input_scale"], scales["label_scale"], inference_lock)
                            else:
                                # 缓冲区未满，暂时跳过或使用单帧推理作为后备
                                # 这里选择跳过，等待足够的帧数
                                print(f"[Buffer] Server:{sid} | Frames: {len(buffer.frames)}/40 | Waiting...")
                                continue
                        else:
                            # 单帧推理模式（原有逻辑）
                            prediction = do_inference(predictor, cov_matrix, scales["input_scale"], scales["label_scale"], inference_lock)

                        gid = req.get("group_id")
                        if gid is None:
                            with id_lock:
                                gid = f"auto_{auto_id_counter[0]}"
                                auto_id_counter[0] += 1

                        # 决定输出值：中位数平滑 或 原始预测值
                        if use_median:
                            # 将预测结果加入缓存
                            prediction_buffer.append(prediction)
                            if len(prediction_buffer) > WINDOW_SIZE:
                                prediction_buffer.pop(0)
                            # 计算中位数
                            output_value = np.median(prediction_buffer)
                            log_str = f"Median({len(prediction_buffer)}): {output_value:.4f}m"
                        else:
                            # 直接使用原始预测值
                            output_value = prediction
                            log_str = f"Raw: {output_value:.4f}m"

                        response = {
                            "server_id": sid,
                            "label": round(output_value, 4),
                            "status": "success"
                        }
                        # 转发到 calculate_position（如果配置了）
                        if forward_manager:
                            forward_manager.send(response)

                        # 打印日志
                        if use_median:
                            print(f"[Inference] Client:{client_id} | GID: {gid} | Server: {sid} | Raw: {prediction:.4f}m | {log_str}")
                        else:
                            print(f"[Inference] Client:{client_id} | GID: {gid} | Server: {sid} | {log_str}")
                except Exception as e:
                    print(f"[Error Decoding JSON] Client:{client_id} | {e}")
    except Exception as e:
        print(f"[Error] Client:{client_id} | {e}")
    finally:
        client_socket.close()
        print(f"[-] Client {client_id} disconnected")


class ForwardManager:
    """转发管理器，支持自动重连"""
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.lock = threading.Lock()
        self.enabled = bool(host and port and port > 0)
        self._connect()

    def _connect(self):
        """建立连接"""
        if not self.enabled:
            print(f"[Forward] Disabled (host={self.host}, port={self.port})")
            return

        max_retries = 5
        for i in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5)
                self.socket.connect((self.host, self.port))
                print(f"[Forward] Connected to {self.host}:{self.port}")
                return
            except Exception as e:
                print(f"[Forward] Connection attempt {i+1}/{max_retries} failed: {e}")
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"[Forward] All connection attempts failed. Forwarding will be disabled.")
                    self.enabled = False
                    self.socket = None

    def send(self, data):
        """发送数据"""
        if not self.enabled:
            return

        with self.lock:
            if not self.socket:
                return

            try:
                self.socket.sendall((json.dumps(data) + "\n").encode('utf-8'))
            except Exception as e:
                print(f"[Warning] Forward failed: {e}. Attempting to reconnect...")
                self.socket = None
                self._connect()

    def close(self):
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


def start_server(host, port, output_dir, ckpt_path, forward_host=None, forward_port=None, use_cnn_gru=False, use_median=True):
    """
    实时监听：实时预测并回传结果，支持多客户端并发连接
    支持转发到另一个服务（如 calculate_position）
    输出累计10个数据的中位数

    Args:
        use_cnn_gru: 是否使用 CNN-GRU 模型（为每个 server_id 维护40帧序列）
        use_median: 是否使用中位数平滑（默认 True）
    """
    scales = load_scaling_config(ckpt_path)
    predictor = init_predictor(ckpt_path, use_cnn_gru=use_cnn_gru)

    # 建立转发连接管理器
    forward_manager = ForwardManager(forward_host, forward_port) if forward_host and forward_port else None

    # 实时模式下的自动 ID 计数器（使用列表使其可变）
    auto_id_counter = [50000]
    # 用于保护 auto_id_counter 的锁
    id_lock = threading.Lock()
    # 用于保护 MindSpore 推理的锁（防止多线程并发导致递归错误）
    inference_lock = threading.Lock()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    mode_str = "CNN-GRU (40-frame sequences)" if use_cnn_gru else "CNN-only (single frame)"
    median_str = "Median smoothing ON" if use_median else "Median smoothing OFF"
    forward_info = ""
    if forward_manager and forward_manager.enabled:
        forward_info = f" → Forward to {forward_host}:{forward_port}"
    print(f"[Server] Ready on {host}:{port}{forward_info} | Mode: {mode_str} | {median_str} | Max queue: 20")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            # 为每个客户端创建新线程
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, addr, predictor, scales, forward_manager, auto_id_counter, id_lock, inference_lock, use_cnn_gru, use_median),
                daemon=True
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("\n[!] Server shutting down...")
    finally:
        if forward_manager:
            forward_manager.close()
            print("[Forward] Connection closed")
        server_socket.close()

def process_batch(input_file, output_dir):
    """
    批处理模式：加入全局计数器防止文件覆盖
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    count = 0
    auto_id_counter = 10000

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                cov = compute_covariance_matrix(data['part1'], data['part2'])

                # ID 补全逻辑
                gid = data.get("group_id")
                if gid is None:
                    gid = f"auto_{auto_id_counter}"
                    auto_id_counter += 1

                ts = data.get("timestamp", "0")
                np.save(out_path / f"batch_{gid}_{ts}.npy", cov)
                count += 1
            except: continue
    print(f"[Batch] Done. Processed {count} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default=str(Path(__file__).parent / "distance.ckpt"))
    parser.add_argument("--file", type=str, help="Run Batch Mode if provided")
    parser.add_argument("--output_dir", type=str, default="output_npy")
    parser.add_argument("--forward-host", type=str, default="127.0.0.1", help="Forward results to another host")
    parser.add_argument("--forward-port", type=int, default=5001, help="Forward results to another port")
    parser.add_argument("--use-cnn-gru", default=False, help="Use CNN-GRU model (requires 40-frame sequences per server)")
    parser.add_argument("--no-median", action="store_true", help="Disable median smoothing (output raw predictions)")

    args = parser.parse_args()
    if args.file:
        process_batch(args.file, args.output_dir)
    else:
        start_server(args.host, args.port, args.output_dir, args.ckpt, args.forward_host, args.forward_port, args.use_cnn_gru, use_median=not args.no_median)
