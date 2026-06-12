"""
Export the STM32-optimised model to ONNX format.

Usage:
    cd dm_model_stm32/src
    python export_onnx.py
    python export_onnx.py --ckpt ../../model_stm32.ckpt --onnx ../../model_stm32.onnx
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import mindspore
from model import Network


def _sanitize_onnx(onnx_path):
    """Replace `/` → `.` and `-` → `_` in all ONNX node/tensor names."""
    try:
        import onnx
    except ImportError:
        logging.warning(
            "onnx package not found -- name sanitisation skipped.\n"
            "Run: pip install onnx, then re-run this script."
        )
        return

    model = onnx.load(onnx_path)
    rename = {}

    def _safe(n):
        new = n.replace("/", ".").replace("-", "_")
        if new != n:
            rename[n] = new
        return new

    for v in model.graph.input:               v.name = _safe(v.name)
    for v in model.graph.output:              v.name = _safe(v.name)
    for v in model.graph.value_info:          v.name = _safe(v.name)
    for t in model.graph.initializer:         t.name = _safe(t.name)

    for node in model.graph.node:
        node.name = _safe(node.name)
        for i in range(len(node.output)):
            old = node.output[i]
            if old in rename:
                node.output[i] = rename[old]
            else:
                node.output[i] = _safe(node.output[i])  # direct fallback
        for i in range(len(node.input)):
            old = node.input[i]
            if old in rename:
                node.input[i] = rename[old]
            else:
                node.input[i] = _safe(node.input[i])    # direct fallback

    if rename:
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
        logging.info(f"Sanitised {len(rename)} names (-> . and _)")
    else:
        logging.info("No names needed sanitising")


def export(ckpt_path="../../model_stm32.ckpt",
           onnx_path="../../model_stm32.onnx"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt.resolve()}")

    # Load config (input_scale from .json)
    input_scale = 1.0
    config_path = ckpt.with_suffix(".json")
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        input_scale = config.get("input_scale", input_scale)
        logging.info(f"Loaded config: input_scale={input_scale}")

    # Build net & load weights
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE,
                                  device_target="CPU")
    net = Network()
    param_dict = mindspore.load_checkpoint(str(ckpt))
    param_not_load, _ = mindspore.load_param_into_net(net, param_dict)
    if param_not_load:
        logging.warning(f"Params not loaded: {param_not_load}")
    net.set_train(False)
    logging.info(f"Loaded checkpoint: {ckpt}")

    # Dummy input (shape matches training)
    dummy = (np.random.randn(1, 2, 40, 40).astype(np.float32)
             / input_scale)
    dummy_tensor = mindspore.Tensor(dummy, mindspore.float32)

    # Export
    mindspore.export(net, dummy_tensor,
                     file_name=str(onnx_path), file_format="ONNX")
    logging.info(f"Exported: {onnx_path}")

    # Sanitize names (replace / -> ., - -> _)
    _sanitize_onnx(onnx_path)

    # Verify
    out = Path(onnx_path)
    if out.exists():
        mb = out.stat().st_size / (1024 * 1024)
        logging.info(f"Size: {mb:.2f} MB")
        try:
            import onnx
            model = onnx.load(str(out))
            onnx.checker.check_model(model)
            logging.info("ONNX check: passed")
            for ipt in model.graph.input:
                dims = [d.dim_value for d in ipt.type.tensor_type.shape.dim]
                logging.info(f"  Input:  {ipt.name}  {dims}")
            for opt in model.graph.output:
                dims = [d.dim_value for d in opt.type.tensor_type.shape.dim]
                logging.info(f"  Output: {opt.name}  {dims}")
        except ImportError:
            logging.info("onnx package not installed -- skipping validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="../../model_stm32.ckpt")
    parser.add_argument("--onnx", default="../../model_stm32.onnx")
    args = parser.parse_args()
    export(args.ckpt, args.onnx)
