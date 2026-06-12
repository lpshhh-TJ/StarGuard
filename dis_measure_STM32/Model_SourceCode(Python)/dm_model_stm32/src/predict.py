# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mindspore
from mindspore import context
import numpy as np

from model import Network
from utils import load_model, predict_dir


def setup_logging(log_dir: Path = None):
    """Setup logging for prediction."""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"predict_{timestamp}.log"

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Prediction logging initialized. Log file: {log_file}")
    return log_file


def _parse_args():
    parser = argparse.ArgumentParser(description="Load checkpoint and run batch predictions on .npy files")
    parser.add_argument("--ckpt", type=Path, default=Path("model.ckpt"), help="Checkpoint file to load")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .npy files to predict")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write .txt prediction files")
    parser.add_argument("--input-scale", type=float, default=0.0, help="Scale used to divide input features. 0.0 = Use config or auto-detect.")
    parser.add_argument("--label-scale", type=float, default=1.0, help="Scale used to multiply predicted labels")
    parser.add_argument("--device-target", type=str, default="CPU", help="MindSpore device target, e.g., CPU or Ascend")
    return parser.parse_args()


def check_input_files(input_dir: Path):
    """检查输入文件夹并确认数据格式"""
    for file_name in sorted(input_dir.iterdir()):
        if file_name.suffix == ".npy":
            arr = np.load(file_name)
            # 允许 (16, 16) 或 (1, 16, 16) 等形状
            if arr.ndim not in [2, 3]:
                raise ValueError(f"Unsupported feature shape {arr.shape} in {file_name}. Expected 2D or 3D array.")
            return arr.shape
    raise FileNotFoundError(f"No .npy files found in {input_dir}")


def main():
    args = _parse_args()

    # Setup logging for prediction
    setup_logging()

    # 设置运行环境
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    logging.info(f"Arguments: {args}")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    logging.info(f"Checking input files in {input_dir}...")
    sample_shape = check_input_files(input_dir)
    logging.info(f"Detected input shape: {sample_shape}")

    # 注意：Network() 在 model.py 中定义时不接收参数
    model = Network()

    logging.info(f"Loading checkpoint {args.ckpt} into model...")
    load_model(model, str(args.ckpt))

    # Try to load corresponding config file for scaling factors
    input_scale = args.input_scale
    label_scale = args.label_scale
    config_loaded = False
    
    config_path = args.ckpt.with_suffix(".json")
    if config_path.exists():
        logging.info(f"Loading scaling config from {config_path}...")
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
                input_scale = config.get("input_scale", input_scale) if args.input_scale <= 0 else args.input_scale
                label_scale = config.get("label_scale", label_scale)
                logging.info(f"Using loaded scaling factors -> input_scale: {input_scale}, label_scale: {label_scale}")
                config_loaded = True
        except Exception as e:
            logging.error(f"Error loading config: {e}")

    if not config_loaded:
        logging.warning(f"Config file {config_path} missing or invalid.")
        if input_scale <= 0:
            logging.warning("No input_scale provided and no config found. Will try to auto-detect from first sample (RISKY!).")

    # 再次检查输入数据大小
    # 随机抽查一个文件
    sample_file = next(input_dir.glob("*.npy"))
    sample_data = np.load(sample_file)
    sample_max = np.max(np.abs(sample_data))
    
    # 如果 input_scale 仍然是 0 (未配置且未加载成功)，则使用样本最大值
    if input_scale <= 0:
        logging.warning(f"[Auto-Detect] input_scale was 0. Setting based on sample file: {sample_max}")
        input_scale = float(sample_max)

    if sample_max > input_scale * 10:
        logging.warning(f"[DANGER] Data max value ({sample_max:.2f}) is MUCH larger than input_scale ({input_scale}).")
        logging.warning(f"Your input features will be > 10.0 after scaling, which usually causes output explosion.")
        logging.warning(f"Recommendation: Ensure you have a valid model.json from training.")

    logging.info(f"Running predictions for .npy -> .txt: {input_dir} -> {output_dir}")
    # utils.py 中的 predict_dir 会自动处理 2D/3D 的 reshape
    predict_dir(model, input_dir, output_dir, input_scale=input_scale, label_scale=label_scale)


if __name__ == "__main__":
    main()