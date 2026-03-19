# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.
#这是通过用户与各个基站距离解算坐标的程序

import socket
import json
import argparse
import time
import threading
import sys
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests
import subprocess
import platform

# 导入抗多径定位模块
try:
    from multipath_resistant import MultipathResistantPositioner, create_positioner
    MULTIPATH_RESISTANT_AVAILABLE = True
except ImportError:
    MULTIPATH_RESISTANT_AVAILABLE = False
    print("[Warning] multipath_resistant module not found, using standard positioning only")

# 全局保存状态标志
save_enabled = False
save_lock = threading.Lock()

# 基站坐标同步配置
ANCHORS_SERVER_URL = "http://124.70.161.112:3000"
ANCHORS_UPDATE_INTERVAL = 5  # 每5秒更新一次
anchors_lock = threading.Lock()  # 保护 STATIONS 的锁

# HTTP 服务器配置
HTTP_SERVER_URL = "http://124.70.161.112:3000/update"
HTTP_ENABLED = True  # 默认开启 HTTP 发送
# 线程池用于异步 HTTP 请求
http_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="http_sender")

# HTTP 发送限流：每秒最多1次
_last_http_send_time = 0
HTTP_MIN_INTERVAL = 1.0

def send_position_to_http_async(x, y, z):
    """
    异步发送位置信息到 HTTP 服务器，不阻塞主线程
    限流：每秒最多1次
    """
    global _last_http_send_time

    if not HTTP_ENABLED:
        return

    current_time = time.time()

    # 检查发送间隔
    if current_time - _last_http_send_time < HTTP_MIN_INTERVAL:
        return  # 距离上次发送不足1秒，跳过

    _last_http_send_time = current_time

    # 提交到线程池异步执行
    http_executor.submit(_send_http_request, x, y, z)

def _send_http_request(x, y, z):
    """
    实际执行 HTTP 请求的函数（在后台线程中运行）
    """
    try:
        data = json.dumps({
            "x": x,
            "y": y,
            "z": z,
            "isFallen": False
        }).encode('utf-8')

        req = urllib.request.Request(
            HTTP_SERVER_URL,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Content-Length': len(data)
            },
            method='POST'
        )

        # 设置更短的超时时间
        with urllib.request.urlopen(req, timeout=1) as response:
            response_data = response.read().decode('utf-8')
            if response.status == 200:
                print(f"[HTTP] OK - Position sent: x={x:.4f}, y={y:.4f}, z={z:.4f} | Response: {response_data}")
    except urllib.error.URLError as e:
        print(f"[HTTP] Network Error: {e}")
    except Exception as e:
        print(f"[HTTP] Error: {e}")


# ============ 配置 ============
# 矩形区域大小 (米)
AREA_X = 2.0   # x方向
AREA_Y = 2.0  # y方向
AREA_Z = 1.0 # z方向

# 基站坐标配置 (server_id -> (x, y, z))，初始为空，启动时从服务器获取
STATIONS = {}


def get_anchors(server_url=ANCHORS_SERVER_URL):
    """从服务器获取基站坐标"""
    try:
        response = requests.get(f"{server_url}/anchors", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("anchors", [])
    except Exception as e:
        print(f"[Anchors] 获取基站坐标失败: {e}")
    return None


# 数据集采集进程管理
dataset_process = None
dataset_process_lock = threading.Lock()
DATASET_SERVER_URL = ANCHORS_SERVER_URL  # 默认使用与基站相同的服务器地址


def check_dataset_collecting():
    """检查是否正在采集数据集"""
    global DATASET_SERVER_URL
    try:
        response = requests.get(f"{DATASET_SERVER_URL}/dataset-is-collecting", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("isCollectingDataset", False)
    except Exception as e:
        print(f"[Dataset] 检查采集状态失败: {e}")
    return False


def manage_dataset_collection_process():
    """管理数据集采集进程"""
    global dataset_process, DATASET_SERVER_URL
    is_collecting = check_dataset_collecting()

    with dataset_process_lock:
        process_exists = dataset_process is not None and dataset_process.poll() is None

        if is_collecting and not process_exists:
            # 需要采集但进程不存在，启动进程
            script_path = Path(__file__).parent / "get_distance_dataset.py"
            if script_path.exists():
                try:
                    if platform.system() == "Windows":
                        # Windows: 在新控制台窗口启动
                        dataset_process = subprocess.Popen(
                            [sys.executable, str(script_path), "--api-url", DATASET_SERVER_URL],
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                        print(f"[Dataset] 启动数据集采集程序 (PID: {dataset_process.pid})")
                    else:
                        # Linux/macOS: 尝试在新终端启动
                        # 按优先级尝试常见的终端模拟器
                        terminal_commands = [
                            ["gnome-terminal", "--"],
                            ["konsole", "-e"],
                            ["xfce4-terminal", "-e"],
                            ["xterm", "-e"],
                            # macOS
                            ["open", "-a", "Terminal.app"],
                        ]
                        terminal_found = False
                        for term_cmd in terminal_commands:
                            try:
                                # 检查终端是否可用（简单检查命令是否存在）
                                import shutil
                                if shutil.which(term_cmd[0]):
                                    cmd = term_cmd + [sys.executable, str(script_path), "--api-url", DATASET_SERVER_URL]
                                    # macOS 使用 open 命令时需要特殊处理
                                    if platform.system() == "Darwin" and term_cmd[0] == "open":
                                        cmd = ["open", "-a", "Terminal.app", sys.executable, str(script_path), "--api-url", DATASET_SERVER_URL]
                                    dataset_process = subprocess.Popen(cmd)
                                    print(f"[Dataset] 启动数据集采集程序 (使用终端: {term_cmd[0]})")
                                    terminal_found = True
                                    break
                            except (subprocess.CalledProcessError, FileNotFoundError):
                                continue

                        if not terminal_found:
                            # 没有找到可用的终端，使用后台启动方式
                            print("[Dataset] 未找到可用的终端模拟器，使用后台启动")
                            dataset_process = subprocess.Popen(
                                [sys.executable, str(script_path), "--api-url", DATASET_SERVER_URL],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            print(f"[Dataset] 启动数据集采集程序 (PID: {dataset_process.pid}, 后台模式)")
                except Exception as e:
                    print(f"[Dataset] 启动失败: {e}")
            else:
                print(f"[Dataset] 脚本不存在: {script_path}")
        elif not is_collecting and process_exists:
            # 不需要采集但进程存在，终止进程
            dataset_process.terminate()
            dataset_process.wait(timeout=5)
            dataset_process = None
            print(f"[Dataset] 数据集采集程序已停止")


def init_anchors_from_server(server_url=ANCHORS_SERVER_URL, num_stations=None):
    """从服务器初始化基站坐标，启动时调用一次

    Args:
        server_url: 服务器URL
        num_stations: 要读取的基站数量，None表示读取所有
    """
    global STATIONS
    print(f"[Anchors] 正在从服务器获取基站坐标...")
    anchors = get_anchors(server_url)
    if anchors and len(anchors) >= 4:
        with anchors_lock:
            # 动态读取所有基站或指定数量
            count = num_stations if num_stations is not None else len(anchors)
            STATIONS = {}
            for i in range(min(count, len(anchors))):
                STATIONS[i] = (
                    float(anchors[i]['x']),
                    float(anchors[i]['y']),
                    float(anchors[i]['z'])
                )
        print(f"[Anchors] 基站坐标初始化成功: 共 {len(STATIONS)} 个基站")
        for i, pos in STATIONS.items():
            print(f"  基站{i}: x={pos[0]}, y={pos[1]}, z={pos[2]}")
        return True
    else:
        print(f"[Anchors] 错误: 无法从服务器获取基站坐标，程序无法运行！")
        return False


def update_anchors_periodically():
    """后台线程：定期更新基站坐标"""
    global STATIONS
    while True:
        anchors = get_anchors()
        if anchors and len(anchors) >= 4:
            with anchors_lock:
                # 获取当前基站数量
                current_count = len(STATIONS)
                # 动态更新所有基站
                new_stations = {}
                for i in range(min(current_count, len(anchors))):
                    new_stations[i] = (
                        float(anchors[i]['x']),
                        float(anchors[i]['y']),
                        float(anchors[i]['z'])
                    )
                STATIONS = new_stations

                print(f"[Anchors] 基站坐标已更新: 共 {len(STATIONS)} 个基站")
                for i, pos in STATIONS.items():
                    print(f"  基站{i}: x={pos[0]}, y={pos[1]}, z={pos[2]}")

        # 检查并管理数据集采集进程
        manage_dataset_collection_process()

        time.sleep(ANCHORS_UPDATE_INTERVAL)


def start_anchors_updater():
    """启动基站坐标更新线程"""
    updater_thread = threading.Thread(target=update_anchors_periodically, daemon=True)
    updater_thread.start()
    print(f"[Anchors] 后台更新线程已启动，每 {ANCHORS_UPDATE_INTERVAL} 秒更新一次")


# 全局定位器实例
multipath_positioner = None

def calculate_position(distances, use_multipath=False):
    """
    计算三维位置，支持标准最小二乘法和抗多径算法

    原理：对于每个基站 i，有 (x - xi)^2 + (y - yi)^2 + (z - zi)^2 = di^2
    展开后消去二次项，得到线性方程 Ax = b。

    Args:
        distances: dict {server_id: distance}
        use_multipath: 是否使用抗多径算法

    Returns:
        (x, y, z) or (None, None, None) if failed
    """
    global STATIONS, multipath_positioner

    # 获取当前基站坐标的快照（使用锁保护）
    with anchors_lock:
        stations = STATIONS.copy()

    # 检查基站坐标是否已初始化
    if not stations or len(stations) < 4:
        return None, None, None

    # 提取有数据的基站（只使用非None的距离）
    valid_ids = [k for k in distances if distances[k] is not None and k in stations]

    if len(valid_ids) < 4:
        return None, None, None  # 3D定位至少需要4个基站

    # 使用抗多径算法
    if use_multipath and MULTIPATH_RESISTANT_AVAILABLE and multipath_positioner is not None:
        valid_stations = {k: stations[k] for k in valid_ids}
        valid_distances = {k: distances[k] for k in valid_ids}

        position, _ = multipath_positioner.compute_position(valid_stations, valid_distances)

        if position is not None:
            # 移除了 RANSAC 调试信息输出
            return float(position[0]), float(position[1]), float(position[2])

    # 标准最小二乘法
    # 构建线性方程组 Ax = b
    # 以第一个基站为参考，与其他基站做差
    ref_id = valid_ids[0]
    ref_x, ref_y, ref_z = stations[ref_id]
    ref_d = distances[ref_id]

    A = []
    b = []

    for sid in valid_ids[1:]:
        xi, yi, zi = stations[sid]
        di = distances[sid]

        # 方程: 2*(xi - ref_x)*x + 2*(yi - ref_y)*y + 2*(zi - ref_z)*z
        #      = xi^2 + yi^2 + zi^2 - ref_x^2 - ref_y^2 - ref_z^2 + ref_d^2 - di^2
        A.append([2 * (xi - ref_x), 2 * (yi - ref_y), 2 * (zi - ref_z)])
        b.append(xi**2 + yi**2 + zi**2 - ref_x**2 - ref_y**2 - ref_z**2 + ref_d**2 - di**2)

    A = np.array(A)
    b = np.array(b)

    try:
        # 最小二乘法求解
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        return float(result[0]), float(result[1]), float(result[2])
    except np.linalg.LinAlgError:
        return None, None, None


def init_multipath_positioner(ransac_iterations=100, ransac_threshold=0.5,
                               process_noise=0.1, measure_noise=0.5,
                               enable_ransac=True, enable_kalman=True):
    """
    初始化抗多径定位器

    Args:
        ransac_iterations: RANSAC 迭代次数 (默认100)
        ransac_threshold: RANSAC 残差阈值/米 (默认0.5)
        process_noise: 卡尔曼过程噪声 (默认0.1)
        measure_noise: 卡尔曼测量噪声 (默认0.5)
        enable_ransac: 是否启用RANSAC (默认True)
        enable_kalman: 是否启用卡尔曼滤波 (默认True)
    """
    global multipath_positioner

    if not MULTIPATH_RESISTANT_AVAILABLE:
        print("[Error] multipath_resistant module not available")
        return False

    try:
        multipath_positioner = MultipathResistantPositioner(
            ransac_iterations=ransac_iterations,
            ransac_threshold=ransac_threshold,
            process_noise=process_noise,
            measure_noise=measure_noise,
            enable_ransac=enable_ransac,
            enable_kalman=enable_kalman
        )
        print(f"[Multipath] Initialized: RANSAC={'ON' if enable_ransac else 'OFF'}, "
              f"Kalman={'ON' if enable_kalman else 'OFF'}, "
              f"Iterations={ransac_iterations}, Threshold={ransac_threshold}m")
        return True
    except Exception as e:
        print(f"[Error] Failed to initialize multipath positioner: {e}")
        return False


def save_position_to_jsonl(filepath, x, y, z, distances):
    """将位置信息保存为 JSONL 格式"""
    data = {
        "timestamp": time.time(),
        "x": x,
        "y": y,
        "z": z,
        "distances": dict(distances)
    }
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"[Warning] Failed to save to file: {e}")


def start_server(host, port, output_file=None, forward_host=None, forward_port=None, http_enabled=True,
                 use_multipath=False, ransac_config=None, num_stations=None):
    """
    作为服务器监听端口，接收 preprocess 处理好的距离数据并计算位置
    每收到任何基站的新数据就立即更新位置

    Args:
        use_multipath: 是否启用抗多径定位算法
        ransac_config: RANSAC配置字典 (可选)
    """
    global HTTP_ENABLED
    HTTP_ENABLED = http_enabled

    # 初始化抗多径定位器
    if use_multipath:
        if ransac_config is None:
            ransac_config = {}
        init_multipath_positioner(**ransac_config)

    # 首先从服务器初始化基站坐标
    if not init_anchors_from_server(num_stations=num_stations):
        print("[Server] 无法获取基站坐标，退出程序")
        return

    # 动态初始化距离字典和标志（根据实际基站数量）
    num_stations = len(STATIONS)
    distances = {i: 1.0 for i in range(num_stations)}
    received_flags = {i: False for i in range(num_stations)}
    print(f"[Server] 初始化 {num_stations} 个基站的距离字典")

    # 默认转发到5002端口
    if not forward_host:
        forward_host = "127.0.0.1"
    if not forward_port:
        forward_port = 5002

    # 建立转发连接
    forward_socket = None
    try:
        forward_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        forward_socket.connect((forward_host, forward_port))
        print(f"[Forward] Connected to {forward_host}:{forward_port}")
    except Exception as e:
        print(f"[Warning] Cannot connect to forward destination {forward_host}:{forward_port}: {e}")

    if output_file:
        print(f"[Output] File: {output_file}")

    print(f"[Server] Starting on {host}:{port}")
    if HTTP_ENABLED:
        print(f"[Server] HTTP POST to: {HTTP_SERVER_URL} (async)")
    print(f"[Server] Station positions:")
    for sid, (x, y, z) in STATIONS.items():
        print(f"  Station {sid}: ({x}, {y}, {z})")
    print("-" * 50)

    # 启动基站坐标更新线程
    start_anchors_updater()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"[Server] Ready and listening on {host}:{port}")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            client_id = f"{addr[0]}:{addr[1]}"
            print(f"[+] Client connected from {client_id}")

            # 为每个客户端创建处理线程
            threading.Thread(
                target=handle_client_connection,
                args=(client_socket, client_id, distances, received_flags, forward_socket, output_file, use_multipath),
                daemon=True
            ).start()

    except KeyboardInterrupt:
        print("\n[Server] Interrupted by user")
    finally:
        if forward_socket:
            forward_socket.close()
            print("[Forward] Connection closed")
        server_socket.close()
        print("[Server] Exited")


def handle_client_connection(client_socket, client_id, distances, received_flags, forward_socket, output_file, use_multipath=False):
    """处理单个客户端连接"""
    try:
        # 设置非阻塞模式
        client_socket.settimeout(1.0)

        while True:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break

                lines = data.decode('utf-8', errors='ignore').split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        msg = json.loads(line)

                        if msg.get("status") == "success":
                            server_id = msg.get("server_id")
                            distance = msg.get("label")

                            if server_id in distances:
                                # 更新该基站的距离
                                distances[server_id] = distance
                                received_flags[server_id] = True

                                # 检查是否所有基站都已收到数据
                                all_received = all(received_flags.values())
                                total_stations = len(received_flags)
                                status_str = "All stations" if all_received else f"{sum(received_flags.values())}/{total_stations} stations"

                                # 尝试计算位置（只要有任何更新就尝试计算）
                                x, y, z = calculate_position(distances, use_multipath=use_multipath)
                                if x is not None:
                                    with save_lock:
                                        is_saving = save_enabled

                                    # 异步发送位置到 HTTP 服务器（不阻塞）
                                    if HTTP_ENABLED:
                                        send_position_to_http_async(x, y, z)

                                    if output_file and is_saving:
                                        save_position_to_jsonl(output_file, x, y, z, distances)

                                    # 转发位置到5002端口（每次计算都发送）
                                    if forward_socket:
                                        try:
                                            position_data = {
                                                "timestamp": time.time(),
                                                "x": x,
                                                "y": y,
                                                "z": z,
                                                "distances": dict(distances)
                                            }
                                            forward_socket.sendall((json.dumps(position_data) + "\n").encode('utf-8'))
                                        except Exception as e:
                                            print(f"[Warning] Forward failed: {e}")

                                    # 每次都打印位置信息
                                    save_status = "[SAVING]" if is_saving else "[NOT SAVING]"
                                    print(f"{save_status} Position [{status_str}]: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
                                    print(f"  Current Distances: {distances}")
                                    print("-" * 50)
                            else:
                                print(f"[Warning] Unknown server_id: {server_id}")

                    except json.JSONDecodeError as e:
                        print(f"[Error] JSON decode error: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                break

    except Exception as e:
        print(f"[Error] Client {client_id}: {e}")
    finally:
        client_socket.close()
        print(f"[-] Client {client_id} disconnected")


def start_client(host, port, output_file=None, forward_host=None, forward_port=None, http_enabled=True,
                 use_multipath=False, ransac_config=None, num_stations=None):
    """
    作为客户端连接到服务端，接收距离数据并计算位置
    支持将位置转发到另一个服务
    支持自动重连

    Args:
        use_multipath: 是否启用抗多径定位算法
        ransac_config: RANSAC配置字典 (可选)
    """
    global HTTP_ENABLED
    HTTP_ENABLED = http_enabled

    # 初始化抗多径定位器
    if use_multipath:
        if ransac_config is None:
            ransac_config = {}
        init_multipath_positioner(**ransac_config)

    # 首先从服务器初始化基站坐标
    if not init_anchors_from_server(num_stations=num_stations):
        print("[Client] 无法获取基站坐标，退出程序")
        return

    # 动态初始化距离字典和标志（根据实际基站数量）
    num_stations = len(STATIONS)
    distances = {i: 1.0 for i in range(num_stations)}
    received_flags = {i: False for i in range(num_stations)}
    print(f"[Client] 初始化 {num_stations} 个基站的距离字典")

    # 默认转发到5002端口
    if not forward_host:
        forward_host = "127.0.0.1"
    if not forward_port:
        forward_port = 5002

    # 建立转发连接
    forward_socket = None
    try:
        forward_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        forward_socket.connect((forward_host, forward_port))
        print(f"[Forward] Connected to {forward_host}:{forward_port}")
    except Exception as e:
        print(f"[Warning] Cannot connect to forward destination {forward_host}:{forward_port}: {e}")

    if output_file:
        print(f"[Output] File: {output_file}")

    print(f"[Client] Connecting to {host}:{port}")
    if HTTP_ENABLED:
        print(f"[Client] HTTP POST to: {HTTP_SERVER_URL} (async)")
    print(f"[Client] Station positions:")
    for sid, (x, y, z) in STATIONS.items():
        print(f"  Station {sid}: ({x}, {y}, {z})")
    print("-" * 50)

    # 启动基站坐标更新线程
    start_anchors_updater()

    reconnect_delay = 1
    max_reconnect_delay = 10

    while True:
        client_socket = None
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)  # 设置10秒超时
            client_socket.connect((host, port))
            print(f"[Client] Connected to server")

            # 将 socket 包装成文件流，按行读取
            f = client_socket.makefile('r', encoding='utf-8')
            reconnect_delay = 1  # 重置重连延迟

            while True:
                line = f.readline()
                if not line:
                    print("[Client] Server closed connection")
                    break

                try:
                    data = json.loads(line)

                    if data.get("status") == "success":
                        server_id = data.get("server_id")
                        distance = data.get("label")

                        # 更新对应基站的距离
                        if server_id in distances:
                            distances[server_id] = distance
                            received_flags[server_id] = True

                            # 检查是否所有基站都已收到数据
                            all_received = all(received_flags.values())
                            total_stations = len(received_flags)
                            status_str = "All stations" if all_received else f"{sum(received_flags.values())}/{total_stations} stations"

                            # 尝试计算位置（只要有任何更新就尝试计算）
                            x, y, z = calculate_position(distances, use_multipath=use_multipath)
                            if x is not None:
                                with save_lock:
                                    is_saving = save_enabled

                                # 异步发送位置到 HTTP 服务器（不阻塞）
                                if HTTP_ENABLED:
                                    send_position_to_http_async(x, y, z)

                                # 保存到 JSONL 文件（仅在启用保存时）
                                if output_file and is_saving:
                                    save_position_to_jsonl(output_file, x, y, z, distances)

                                # 转发位置到5002端口（每次计算都发送）
                                if forward_socket:
                                    try:
                                        position_data = {
                                            "timestamp": time.time(),
                                            "x": x,
                                            "y": y,
                                            "z": z,
                                            "distances": dict(distances)
                                        }
                                        forward_socket.sendall((json.dumps(position_data) + "\n").encode('utf-8'))
                                    except Exception as e:
                                        print(f"[Warning] Forward failed: {e}")

                                # 每次都打印位置信息
                                save_status = "[SAVING]" if is_saving else "[NOT SAVING]"
                                print(f"{save_status} Position [{status_str}]: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
                                print(f"  Current Distances: {distances}")
                                print("-" * 50)
                        else:
                            print(f"[Warning] Unknown server_id: {server_id}")

                except json.JSONDecodeError as e:
                    print(f"[Error] JSON decode error: {e}")
                except Exception as e:
                    print(f"[Error] {e}")

        except ConnectionRefusedError:
            print(f"[Error] Cannot connect to {host}:{port}. Retrying in {reconnect_delay}s...")
        except (ConnectionResetError, socket.timeout) as e:
            print(f"[Error] Connection lost: {e}. Retrying in {reconnect_delay}s...")
        except KeyboardInterrupt:
            print("\n[Client] Interrupted by user")
            break
        except Exception as e:
            print(f"[Error] Unexpected error: {e}. Retrying in {reconnect_delay}s...")
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass

        # 等待后重连
        time.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    # 清理转发连接
    if forward_socket:
        forward_socket.close()
        print("[Forward] Connection closed")
    print("[Client] Exited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate position from base stations with multipath resistance")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on (server mode) or connect to (client mode)")
    parser.add_argument("--output", type=str, default=None, help="Save positions to JSONL file")
    parser.add_argument("--forward-host", type=str, default=None, help="Forward position to this host")
    parser.add_argument("--forward-port", type=int, default=None, help="Forward position to this port")
    parser.add_argument("--client-mode", action="store_true", help="Run in client mode (connect to server) instead of server mode")
    parser.add_argument("--disable-http", action="store_true", help="Disable HTTP POST to server")
    parser.add_argument("--http-url", type=str, default="http://localhost:3000/update", help="HTTP server URL for POST requests")
    parser.add_argument("--anchors-url", type=str, default="http://124.70.161.112:3000", help="Server URL for fetching anchors coordinates")
    parser.add_argument("--dataset-url", type=str, default=ANCHORS_SERVER_URL, help="Server URL for checking dataset collecting status")
    parser.add_argument("--disable-dataset-manager", action="store_true", help="Disable automatic dataset collection process management")

    # 抗多径定位相关参数
    parser.add_argument("--use-multipath", default=True, help="Enable multipath-resistant positioning (RANSAC + Kalman filter)")
    parser.add_argument("--ransac-iterations", type=int, default=100, help="RANSAC iterations (default: 100)")
    parser.add_argument("--ransac-threshold", type=float, default=0.5, help="RANSAC residual threshold in meters (default: 0.5)")
    parser.add_argument("--kalman-process-noise", type=float, default=0.01, help="Kalman filter process noise (default: 0.01, smaller = smoother)")
    parser.add_argument("--kalman-measure-noise", type=float, default=2.0, help="Kalman filter measurement noise (default: 2.0, larger = smoother)")
    parser.add_argument("--disable-ransac", action="store_true", help="Disable RANSAC, only use Kalman filter")
    parser.add_argument("--disable-kalman", action="store_true", help="Disable Kalman filter, only use RANSAC")

    # 基站数量相关参数
    parser.add_argument("--num-stations", type=int, default=None,
                        help="Number of stations to use (default: all available from server)")

    args = parser.parse_args()

    # 配置抗多径定位参数
    ransac_config = None
    if args.use_multipath:
        ransac_config = {
            'ransac_iterations': args.ransac_iterations,
            'ransac_threshold': args.ransac_threshold,
            'process_noise': args.kalman_process_noise,
            'measure_noise': args.kalman_measure_noise,
            'enable_ransac': not args.disable_ransac,
            'enable_kalman': not args.disable_kalman
        }

    # 更新全局配置
    if args.http_url:
        HTTP_SERVER_URL = args.http_url
    if args.anchors_url:
        ANCHORS_SERVER_URL = args.anchors_url
    if args.dataset_url:
        DATASET_SERVER_URL = args.dataset_url

    http_enabled = not args.disable_http
    dataset_manager_enabled = not args.disable_dataset_manager

    if not dataset_manager_enabled:
        # 禁用数据集管理功能
        def manage_dataset_collection_process():
            pass

    if args.client_mode:
        start_client(args.host, args.port, args.output, args.forward_host, args.forward_port, http_enabled,
                    use_multipath=args.use_multipath, ransac_config=ransac_config, num_stations=args.num_stations)
    else:
        start_server(args.host, args.port, args.output, args.forward_host, args.forward_port, http_enabled,
                    use_multipath=args.use_multipath, ransac_config=ransac_config, num_stations=args.num_stations)
