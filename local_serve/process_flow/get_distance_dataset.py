# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.
#这是在网页上选择采集数据集时运行的程序

#!/usr/bin/env python3
"""
Distance Dataset Collector
监听5010端口，接收IQ数据并保存为Hankel矩阵(.npy格式)
"""
import argparse
import json
import socket
import threading
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import requests


def compute_covariance_matrix(part1, part2):
    """
    计算 Hankel 矩阵（从 preprocess_hankel_middle.py 复用）

    Args:
        part1: 长度为158的整数列表（本地信号，实部虚部交错）
        part2: 长度为158的整数列表（远程信号，实部虚部交错）

    Returns:
        40×40 复数 Hankel 矩阵
    """
    # 转换为 numpy 数组
    part1 = np.array(part1)
    part2 = np.array(part2)

    # 1. 分离实部/虚部，重构复数信号
    z_local = part1[0::2] + 1j * part1[1::2]   # 长度79
    z_remote = part2[0::2] + 1j * part2[1::2]  # 长度79

    # 2. 计算互相关
    z_diff = z_remote * np.conj(z_local)

    # 3. 构造 Hankel 矩阵 (40×40)
    N, M = len(z_diff), 40
    K = N - M + 1  # K = 40
    C = np.zeros((M, K), dtype=complex)
    for m in range(M):
        C[m, :] = z_diff[m : m + K]
    return C


# --- 数据类 ---
@dataclass
class Coordinate:
    """三维坐标"""
    x: float
    y: float
    z: float


# --- 网页协调器 ---
class WebCoordinator:
    """网页服务器协调器：处理 API 通信和距离计算"""

    def __init__(self, api_base: str = "http://localhost:3000"):
        self.api_base = api_base
        self.anchors: List[Coordinate] = []
        self.coordinate: Optional[Coordinate] = None
        self.is_collecting = False
        self.is_completed = False
        self.distances: List[float] = []
        self.lock = threading.Lock()

    def fetch_anchors(self) -> bool:
        """获取基站位置（每5秒调用）"""
        try:
            response = requests.get(f"{self.api_base}/anchors", timeout=3)
            if response.status_code == 200:
                data = response.json()
                anchors_list = data.get("anchors", [])
                with self.lock:
                    self.anchors = [Coordinate(a["x"], a["y"], a["z"]) for a in anchors_list]
                return True
        except requests.RequestException as e:
            print(f"[API Error] Failed to fetch anchors: {e}")
        return False

    def check_is_collecting(self) -> Optional[bool]:
        """检查是否正在采集数据集"""
        try:
            response = requests.get(f"{self.api_base}/dataset-is-collecting", timeout=3)
            if response.status_code == 200:
                data = response.json()
                return data.get("isCollectingDataset", False)
        except requests.RequestException as e:
            print(f"[API Error] Failed to check collecting status: {e}")
        return None

    def fetch_coordinate(self) -> bool:
        """获取当前采集坐标（每1秒调用）"""
        try:
            response = requests.get(f"{self.api_base}/dataset-current-coordinate", timeout=3)
            if response.status_code == 200:
                data = response.json()
                coord = data.get("coordinate", {})
                with self.lock:
                    self.coordinate = Coordinate(coord["x"], coord["y"], coord["z"])
                    self.is_collecting = data.get("isCollecting", False)
                    self.is_completed = data.get("isCompleted", False)
                return True
        except requests.RequestException as e:
            print(f"[API Error] Failed to fetch coordinate: {e}")
        return False

    def calculate_distances(self) -> List[float]:
        """计算当前坐标到四个基站的距离"""
        with self.lock:
            if not self.anchors or not self.coordinate:
                return []
            coord = self.coordinate
            anchors_copy = self.anchors[:]

        distances = []
        for anchor in anchors_copy:
            dist = math.sqrt(
                (coord.x - anchor.x) ** 2 +
                (coord.y - anchor.y) ** 2 +
                (coord.z - anchor.z) ** 2
            )
            distances.append(dist)
        return distances

    def mark_complete(self):
        """标记坐标采集完成"""
        try:
            requests.post(f"{self.api_base}/dataset-coordinate/complete", timeout=3)
            print("[API] Marked coordinate as complete")
        except requests.RequestException as e:
            print(f"[API Error] Failed to mark complete: {e}")


# --- 数据收集器 ---
class DataCollector:
    """数据收集管理器：管理文件夹和数据保存"""

    def __init__(self, output_dir: Path, target_count: int = 100, num_stations: int = 4):
        self.output_dir = output_dir
        self.target_count = target_count
        self.num_stations = num_stations  # 基站数量
        self.folders = {}  # {server_id: Path}
        self.counts = {i: 0 for i in range(num_stations)}  # 动态初始化
        self.lock = threading.Lock()
        self._current_distances = None

    def update_folders(self, distances: List[float]):
        """更新采集文件夹（根据距离创建N个文件夹）"""
        with self.lock:
            self.folders = {}
            self.counts = {i: 0 for i in range(len(distances))}  # 根据距离数量动态初始化
            self._current_distances = distances
            for server_id, dist in enumerate(distances):
                folder_name = f"{dist:.2f}"
                folder_path = self.output_dir / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                self.folders[server_id] = folder_path
            print(f"[Collector] Updated folders ({len(distances)} stations): {[f'{d:.2f}' for d in distances]}")

    def save_data(self, server_id: int, cov_matrix, timestamp: int) -> bool:
        """保存数据到对应基站文件夹，返回是否全部完成"""
        with self.lock:
            if server_id not in self.folders:
                return False

            filename = f"hankel_{timestamp}.npy"
            np.save(self.folders[server_id] / filename, cov_matrix)
            self.counts[server_id] += 1

            # 动态生成进度字符串
            num_stations = len(self.counts)
            progress = f"{{{', '.join(f's{i}:{self.counts[i]}/{self.target_count}' for i in range(num_stations))}}}"
            print(f"[Saved] s{server_id} -> {self._current_distances[server_id]:.2f}/ {progress}")

            return self.is_complete()

    def is_complete(self) -> bool:
        """检查是否所有基站都收集满"""
        return all(count >= self.target_count for count in self.counts.values())


# --- 后台线程函数 ---
def anchor_update_thread(coordinator: WebCoordinator, stop_event: threading.Event):
    """基站位置更新线程（每5秒）"""
    while not stop_event.is_set():
        # 获取基站位置
        if coordinator.fetch_anchors():
            anchors_info = [(a.x, a.y, a.z) for a in coordinator.anchors]
            print(f"[Anchors] Updated: {anchors_info}")

        # 检查采集状态
        is_collecting = coordinator.check_is_collecting()
        if is_collecting is False:
            print("[Dataset] isCollectingDataset = false, shutting down...")
            stop_event.set()
            break
        elif is_collecting is True:
            print("[Dataset] isCollectingDataset = true, continuing...")

        time.sleep(5)


def coordinate_update_thread(coordinator: WebCoordinator, collector: DataCollector,
                             stop_event: threading.Event):
    """采集坐标更新线程（每1秒）"""
    last_is_collecting = False

    while not stop_event.is_set():
        if coordinator.fetch_coordinate():
            # 检查是否开始采集
            if coordinator.is_collecting and not last_is_collecting:
                distances = coordinator.calculate_distances()
                if distances:
                    collector.update_folders(distances)
                    coord = coordinator.coordinate
                    print(f"[Coordinate] Start collecting at ({coord.x}, {coord.y}, {coord.z})")

            # 检查是否完成
            if coordinator.is_completed and last_is_collecting:
                print(f"[Coordinate] Collection completed")

            last_is_collecting = coordinator.is_collecting

        time.sleep(1)


def handle_client(client_sock: socket.socket, client_addr: tuple,
                  coordinator: WebCoordinator, collector: DataCollector,
                  stop_event: threading.Event):
    """
    处理单个客户端连接

    Args:
        client_sock: 客户端socket
        client_addr: 客户端地址
        coordinator: 网页协调器
        collector: 数据收集器
        stop_event: 停止事件
    """
    print(f"[Connection] {client_addr} connected")
    buffer = b""
    count = 0

    try:
        while not stop_event.is_set():
            chunk = client_sock.recv(4096)
            if not chunk:
                break
            buffer += chunk

            # 处理完整的 JSON 行（每行一个数据包）
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode('utf-8'))
                    part1 = data.get('part1')
                    part2 = data.get('part2')

                    # 验证数据
                    if part1 is None or part2 is None:
                        print(f"[Warning] Missing part1/part2 in data")
                        continue
                    if len(part1) != 158 or len(part2) != 158:
                        print(f"[Warning] Invalid data length: {len(part1)}, {len(part2)}")
                        continue

                    # 生成 Hankel 矩阵
                    cov_matrix = compute_covariance_matrix(part1, part2)

                    # 获取元数据
                    server_id = data.get('server_id')
                    timestamp = data.get('timestamp', int(time.time() * 1000))
                    print(f"[Received] IQ data, server_id: {server_id}")

                    # 动态验证 server_id
                    if server_id is None or server_id < 0:
                        print(f"[Warning] Invalid server_id: {server_id}")
                        continue

                    # 如果正在采集，保存数据
                    if coordinator.is_collecting:
                        # 验证 server_id 是否在有效范围内
                        if server_id in collector.folders:
                            is_complete = collector.save_data(server_id, cov_matrix, timestamp)
                            count += 1

                            # 检查是否完成
                            if is_complete:
                                coordinator.mark_complete()
                                print(f"[Complete] Collected {collector.target_count} samples for all {len(collector.folders)} stations!")
                        else:
                            print(f"[Warning] server_id {server_id} not in configured stations: {list(collector.folders.keys())}")

                except json.JSONDecodeError as e:
                    print(f"[Error] JSON decode error: {e}")
                except Exception as e:
                    print(f"[Error] Processing error: {e}")

    finally:
        client_sock.close()
        print(f"[Connection] {client_addr} disconnected (total: {count} files)")


def start_server(host: str, port: int, output_dir: Path,
                 coordinator: WebCoordinator, collector: DataCollector):
    """
    启动 TCP 服务器

    Args:
        host: 监听地址
        port: 监听端口
        output_dir: 输出目录
        coordinator: 网页协调器
        collector: 数据收集器
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)

    print(f"[Server] Listening on {host}:{port}")
    print(f"[Output] Saving to {output_dir.absolute()}")
    print(f"[API] Connecting to {coordinator.api_base}")
    print("[Info] Press Ctrl+C to stop\n")

    stop_event = threading.Event()
    threads = []

    # 启动后台线程
    anchor_thread = threading.Thread(
        target=anchor_update_thread,
        args=(coordinator, stop_event),
        daemon=True
    )
    coord_thread = threading.Thread(
        target=coordinate_update_thread,
        args=(coordinator, collector, stop_event),
        daemon=True
    )
    anchor_thread.start()
    coord_thread.start()
    threads.extend([anchor_thread, coord_thread])

    try:
        while not stop_event.is_set():
            try:
                server_sock.settimeout(1.0)
                client_sock, client_addr = server_sock.accept()
                t = threading.Thread(
                    target=handle_client,
                    args=(client_sock, client_addr, coordinator, collector, stop_event),
                    daemon=True
                )
                t.start()
                threads.append(t)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                print("\n[Server] Shutting down...")
                break
    finally:
        stop_event.set()
        server_sock.close()
        # 等待所有客户端线程结束
        for t in threads:
            t.join(timeout=1.0)
        print("[Server] Stopped")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Receive IQ data and save Hankel matrices as .npy files"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5010, help="Server port (default: 5010)")
    parser.add_argument("--output-dir", default="hankel_dataset", help="Output directory (default: hankel_dataset)")
    parser.add_argument("--api-url", default="http://localhost:3000", help="API server URL (default: http://localhost:3000)")
    parser.add_argument("--target-count", type=int, default=100, help="Target samples per anchor (default: 100)")
    parser.add_argument("--num-stations", type=int, default=None,
                        help="Number of stations to use (default: auto-detect from API)")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化协调器
    coordinator = WebCoordinator(api_base=args.api_url)

    # 确定基站数量
    num_stations = args.num_stations
    if num_stations is None:
        # 尝试从API获取基站数量
        print("[Info] Detecting number of stations from API...")
        if coordinator.fetch_anchors():
            num_stations = len(coordinator.anchors)
            print(f"[Info] Detected {num_stations} stations from API")
        else:
            print("[Warning] Could not detect stations from API, using default (4)")
            num_stations = 4

    print(f"[Setup] Configured for {num_stations} stations")

    # 初始化收集器
    collector = DataCollector(output_dir=output_dir, target_count=args.target_count, num_stations=num_stations)

    start_server(args.host, args.port, output_dir, coordinator, collector)


if __name__ == "__main__":
    main()
