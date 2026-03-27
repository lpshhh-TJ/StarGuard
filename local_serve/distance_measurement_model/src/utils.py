# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.

import logging
import os
from pathlib import Path

import numpy as np
import mindspore


LABEL_LOG_EPS = 1e-3

def _decode_distance(y_scaled: np.ndarray, label_scale: float = 1.0):
    """Decode model output back to distance.

    Training target is: y = log(d + eps) / label_scale  (if label scaling enabled).
    So inference distance is: d = exp(y * label_scale) - eps, then clip to [0, 10].
    """
    y = y_scaled * float(label_scale)
    d = np.exp(y) - LABEL_LOG_EPS
    return np.clip(d, 0.0, 10.0)


def save_model(model, filepath: str):
    mindspore.save_checkpoint(model, filepath)
    logging.info(f"Saved model to {filepath}")


def load_model(model, filepath: str):
    param_dict = mindspore.load_checkpoint(filepath)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    if param_not_load:
        logging.warning(f"Parameters not loaded: {param_not_load}")
    else:
        logging.info("All parameters loaded successfully.")
    return model


def print_predictions(model, dataset, label_scale: float = 1.0, limit: int = 5):
    if dataset is None:
        logging.warning("Dataset is empty; no predictions to display.")
        return

    model.set_train(False)
    shown = 0
    for batch in dataset.create_dict_iterator():
        data = batch["data"]
        label = batch["label"]
        pred = model(data)

        pred_vals = _decode_distance(pred.asnumpy().reshape(-1), label_scale)
        label_vals = _decode_distance(label.asnumpy().reshape(-1), label_scale)

        for pred_value, label_value in zip(pred_vals, label_vals):
            logging.info(f"pred {pred_value:.4f} target {label_value:.4f}")
            shown += 1
            if shown >= limit:
                return


def predict_dir(model, input_dir, output_dir, input_scale: float = 1.0, label_scale: float = 1.0):
    """Run model predictions on all .npy files in `input_dir` and write same-named .txt files to `output_dir`.

    - `input_dir` may be a `str` or `Path` and should contain .npy files matching training feature format.
    - Each .npy is loaded, optionally scaled by `input_scale` (divide), passed through `model`, then
      the predicted scalar is rescaled by `label_scale` (multiply) and written to a .txt file.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Predict input directory not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    model.set_train(False)

    for file_name in sorted(os.listdir(input_path)):
        if not file_name.endswith(".npy"):
            continue

        feature_path = input_path / file_name
        arr = np.load(feature_path)
        
        # Handle complex input (e.g. covariance matrix)
        if np.iscomplexobj(arr):
            # Separate real and imaginary parts
            arr = np.stack([arr.real, arr.imag], axis=0)

        # Handle 2D array (e.g. 80x80 image)
        if arr.ndim == 2:
            # Assume (H, W), reshape to (1, 1, H, W) for batch inference
            arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
        # Handle 3D array (C, H, W)
        elif arr.ndim == 3:
             # Reshape to (1, C, H, W)
            arr = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
        else:
            raise ValueError(f"Unsupported feature shape {arr.shape} in {feature_path}. Expected 2D (H, W) or 3D (C, H, W).")

        arr = arr.astype(np.float32) / float(input_scale)

        tensor = mindspore.Tensor(arr)
        pred = model(tensor)
        pred_vals = _decode_distance(pred.asnumpy().reshape(-1), label_scale)

        # write first prediction value for this file
        pred_value = float(pred_vals[0])
        out_path = output_path / (Path(file_name).stem + ".txt")
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(f"{pred_value}\n")

    logging.info(f"Wrote predictions for .npy files in {input_path} to {output_path}")