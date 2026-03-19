# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.
#这是从串口输出字符串中提取IQ数据的程序

import argparse
import sys
import time
import re
import queue
import threading
import json
import socket
from pathlib import Path
from typing import Optional
import platform

import serial

# --- 核心正则定义 ---
BRACKET_LABEL = re.compile(r"\[(?:0|[1-9]|[1-6]\d|7[0-8])\]:")
HEX_PATTERN = re.compile(r"0x[0-9A-Fa-f]+")
IQ_TO_ZERO = re.compile(r"\[iq_data\].*?\[\d+\]:?\s*", re.DOTALL)

def extract_header_info(segment: str):
    """提取 [iq_data] 后的 server_id (第1个数字) 和 timestamp (第5个数字)"""
    try:
        start_marker = "[iq_data]"
        idx = segment.find(start_marker)
        if idx == -1: return None, None
        nums = re.findall(r'\d+', segment[idx + len(start_marker):idx + 100])
        return (int(nums[0]), int(nums[4])) if len(nums) >= 5 else (None, None)  # (server_id, timestamp)
    except: return None, None

def process_segment(segment: str) -> str:
    segment = IQ_TO_ZERO.sub("", segment)
    segment = BRACKET_LABEL.sub("", segment)
    segment = segment.replace(";", ",").replace(".", ",")
    segment = HEX_PATTERN.sub(lambda m: str(int(m.group(0), 16)), segment)
    lines = [l.strip() for l in segment.splitlines() if l.strip()]
    return " ".join(lines)

def get_third_char(block: str) -> str | None:
    idx = block.find("[iq_data]")
    return block[idx + len("[iq_data]") + 2] if idx != -1 and idx + 12 < len(block) else None

# --- 核心服务线程 ---

def socket_sender_task(s_q: queue.Queue, stop_ev: threading.Event, host: str, primary_port: int, secondary_port: int = None):
    """【长连接】发送线程：支持多端口并发发送，每个端口独立重连"""
    class SocketConn:
        def __init__(self, host, port, name):
            self.host = host
            self.port = port
            self.name = name
            self.sock = None
            self.last_error_time = 0
            self.send_queue = queue.Queue(maxsize=50)  # 每个连接独立的发送队列
            self.running = True

        def connect(self):
            if self.sock is not None:
                return True
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(2.0)
                self.sock.connect((self.host, self.port))
                print(f"\n[Socket] [{self.name}] Connected to {self.host}:{self.port}")
                return True
            except (socket.error, ConnectionRefusedError, socket.timeout) as e:
                # 减少错误日志刷屏，每30秒最多打印一次
                current_time = time.time()
                if current_time - self.last_error_time > 30:
                    print(f"\n[Socket] [{self.name}] Cannot connect to {self.host}:{self.port} ({e})")
                    self.last_error_time = current_time
                return False

        def send(self, data):
            """非阻塞：将数据放入队列，由后台线程处理"""
            try:
                self.send_queue.put_nowait(data)
                return True
            except queue.Full:
                # 队列满时丢弃（防止阻塞主流程）
                return False

        def _sender_loop(self):
            """后台线程：从队列取数据并实际发送"""
            while self.running or not self.send_queue.empty():
                try:
                    data = self.send_queue.get(timeout=0.5)
                    if self.sock is None and not self.connect():
                        continue
                    try:
                        self.sock.sendall(data)
                    except (socket.error, ConnectionResetError, socket.timeout) as e:
                        print(f"\n[Socket] [{self.name}] Connection lost ({e}). Reconnecting...")
                        if self.sock:
                            self.sock.close()
                        self.sock = None
                        # 失败的数据不放回队列（避免堆积）
                except queue.Empty:
                    continue

        def start_sender_thread(self):
            """启动独立的后台发送线程"""
            threading.Thread(target=self._sender_loop, daemon=True).start()

        def close(self):
            self.running = False
            if self.sock:
                self.sock.close()
                self.sock = None

    # 创建连接对象
    connections = [SocketConn(host, primary_port, "Primary")]
    if secondary_port is not None:
        connections.append(SocketConn(host, secondary_port, "Secondary"))

    # 为每个连接启动独立的后台发送线程
    for conn in connections:
        conn.start_sender_thread()

    while not stop_ev.is_set():
        try:
            # 获取解析好的 JSON 字符串
            payload_str = s_q.get(timeout=1.0)
            data = (payload_str + "\n").encode('utf-8')

            # 并发向所有可用连接发送数据（非阻塞）
            for conn in connections:
                conn.send(data)

        except queue.Empty:
            continue

def file_writer_task(w_q: queue.Queue, stop_ev: threading.Event):
    """本地文件写入线程"""
    while not stop_ev.is_set() or not w_q.empty():
        try:
            p, c = w_q.get(timeout=0.2)
            with p.open("a", encoding="utf-8") as f: f.write(c)
        except: continue

# --- 处理逻辑 ---

def write_output(lines: list[str], output_file: Optional[Path], json_file: Optional[Path],
                 ts: Optional[int], server_id: Optional[int], write_q: queue.Queue, s_q: Optional[queue.Queue]):
    
    content = "\n".join(lines) + "\n"
    nums_str = re.findall(r'-?\d+', content.split('组')[-1] if '组' in content else content)
    count = len(nums_str)
    
    # 控制台实时反馈
    print(f"[{time.strftime('%H:%M:%S')}] Received: {count} pts | ServerID: {server_id}", end=' ')

    if count == 316:
        print("-> [MATCHED!]", flush=True)
        payload = {
            "group_id": next((int(re.search(r'\d+', l).group()) for l in lines if "组" in l), None),
            "server_id": server_id, 
            "part1": [int(x) for x in nums_str[:158]], 
            "part2": [int(x) for x in nums_str[158:]], 
            "timestamp": ts or int(time.time())
        }
        # 发送到 Socket 队列
        if s_q is not None:
            s_q.put(json.dumps(payload))
        # 写入 JSONL 队列
        if json_file:
            write_q.put((json_file, json.dumps(payload) + "\n"))
    else:
        print("-> (Skip)", flush=True)

    if output_file:
        write_q.put((output_file, content))

def process_task(data_q: queue.Queue, stop_ev: threading.Event, output_file: Optional[Path], 
                 json_file: Optional[Path], write_q: queue.Queue, s_q: Optional[queue.Queue]):
    buffer = b""
    pending_zero, capturing_match, match_buf, match_count = None, False, [], 1
    
    while not stop_ev.is_set() or not data_q.empty():
        try: chunk = data_q.get(timeout=0.1)
        except queue.Empty: continue
        
        buffer += chunk
        if b'\n' in buffer:
            parts = buffer.split(b'\n')
            buffer = parts[-1]
            for line_b in parts[:-1]:
                if b"Matched!" in line_b:
                    capturing_match, match_buf = True, []
                if b"[iq_data]" not in line_b: continue
                
                try: line_s = line_b.decode("utf-8", errors="ignore").strip()
                except: continue

                if capturing_match:
                    match_buf.append(line_s)
                    if len(match_buf) == 2:
                        server_id, ts = extract_header_info(match_buf[0])
                        write_output([f"第{match_count}组"] + [process_segment(s) for s in match_buf],
                                     output_file, json_file, ts, server_id, write_q, s_q)
                        match_count += 1
                        capturing_match = False
                    continue

                ctype = get_third_char(line_s)
                if ctype == "1" and pending_zero:
                    server_id, ts = extract_header_info(pending_zero)
                    write_output([process_segment(pending_zero), process_segment(line_s)],
                                 output_file, json_file, ts, server_id, write_q, s_q)
                    pending_zero = None
                elif ctype == "0": pending_zero = line_s

def serial_read_task(ser_inst, dq, stop_ev):
    while not stop_ev.is_set():
        try:
            if ser_inst.in_waiting:
                data = ser_inst.read(ser_inst.in_waiting)
                if data: dq.put(data)
            else: time.sleep(0.005)
        except: break

# --- 主入口 ---

def main():
    parser = argparse.ArgumentParser(description="Professional IQ Data Extractor")
    parser.add_argument("ports", nargs='*', help="Serial port(s)")
    parser.add_argument("-i", "--input-file", type=Path, help="Extract from local TXT")
    parser.add_argument("-o", "--output", type=Path, help="Save to TXT")
    parser.add_argument("--json-output", type=Path, help="Save to JSONL")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--disable-socket", action="store_true", help="Disable socket sending (enabled by default)")
    parser.add_argument("--host", default="127.0.0.1", help="Socket server host")
    parser.add_argument("--socket-port", type=int, default=5000, help="Primary socket port (default: 5000)")
    parser.add_argument("--secondary-port", type=int, default=5010, help="Secondary socket port (default: 5010)")
    args = parser.parse_args()

    # 全局控制量
    stop_event = threading.Event()
    write_q = queue.Queue() # 文件队列
    socket_q = queue.Queue() if not args.disable_socket else None # Socket 队列

    # 1. 启动文件写入线程
    threading.Thread(target=file_writer_task, args=(write_q, stop_event), daemon=True).start()

    # 2. 启动 Socket 发送线程（长连接模式）
    if not args.disable_socket:
        threading.Thread(
            target=socket_sender_task,
            args=(socket_q, stop_event, args.host, args.socket_port, args.secondary_port),
            daemon=True
        ).start()

    # 3. 运行模式切换
    if args.input_file:
        print(f"[*] Mode: File Extraction ({args.input_file})")
        dq = queue.Queue()
        threading.Thread(target=process_task, args=(dq, stop_event, args.output, args.json_output, write_q, socket_q), daemon=True).start()
        with open(args.input_file, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk: break
                dq.put(chunk)
        while not dq.empty(): time.sleep(0.5)
        print("[+] File processing complete.")
    else:
        # 根据操作系统设置默认串口
        default_port = "COM17" if platform.system() == "Windows" else "/dev/ttyUSB0"
        target_ports = args.ports if args.ports else [default_port]
        for p in target_ports:
            try:
                ser = serial.Serial(p, args.baud, timeout=0.1)
                dq = queue.Queue()
                threading.Thread(target=serial_read_task, args=(ser, dq, stop_event), daemon=True).start()
                threading.Thread(target=process_task, args=(dq, stop_event, args.output, args.json_output, write_q, socket_q), daemon=True).start()
                print(f"[*] Monitoring {p}...")
            except Exception as e: print(f"[!] Error opening {p}: {e}")
        
        try:
            while not stop_event.is_set(): time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()
            print("\n[!] Stopping...")
            time.sleep(1)

if __name__ == "__main__":
    main()