# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mindspore.dataset as ds
import numpy as np


class DataAugmenter:
    """复数域数据增强器"""

    def __init__(self, enable: bool = True, noise_std: float = 0.01):
        self.enable = enable
        self.noise_std = noise_std

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        if not self.enable:
            return feature

        # 随机选择增强方式
        aug_type = np.random.randint(0, 4)

        if aug_type == 0:  # 水平翻转
            return feature[:, :, :: -1]
        elif aug_type == 1:  # 垂直翻转
            return feature[:, :: -1, :]
        elif aug_type == 2:  # 添加高斯噪声
            noise = np.random.normal(0, self.noise_std, feature.shape).astype(np.float32)
            return feature + noise
        else:  # 不增强
            return feature

DEFAULT_INPUT_SCALE = 4096.0
DEFAULT_LABEL_SCALE = 1.0
LABEL_LOG_EPS = 1e-3


@dataclass
class DatasetMeta:
    count: int
    feature_shape: Tuple[int, ...]
    feature_dir: Path
    label_dir: Path
    input_scale: float
    label_scale: float
    inputs_scaled: bool
    labels_scaled: bool


def _load_feature_and_label(feature_path: Path, label_path: Path) -> Tuple[np.ndarray, float]:
    feature = np.load(feature_path)
    
    # Handle complex input (e.g. covariance matrix)
    if np.iscomplexobj(feature):
        # Separate real and imaginary parts
        # Feature shape likely (H, W), result (2, H, W)
        feature = np.stack([feature.real, feature.imag], axis=0)
    # Handle 2D array (e.g. 80x80 image)
    elif feature.ndim == 2:
        # Assume (H, W), reshape to (1, H, W)
        feature = feature.reshape(1, feature.shape[0], feature.shape[1])
    # Handle 3D array (C, H, W)
    elif feature.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported feature shape {feature.shape} in {feature_path}. Expected 2D (H, W) or 3D (C, H, W).")

    with label_path.open("r", encoding="utf-8") as handle:
        label_value = float(handle.read().strip())

    # Train in log-distance space to reduce scale imbalance.
    # The model learns y = log(d + eps).
    label_value = float(np.log(label_value + LABEL_LOG_EPS))

    return feature, label_value


def _collect_pairs(feature_dir: Path, label_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    first_shape = None
    first_file = None

    for file_name in sorted(os.listdir(feature_dir)):
        if not file_name.endswith(".npy"):
            continue

        feature_path = feature_dir / file_name
        label_path = label_dir / (Path(file_name).stem + ".txt")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file missing for {feature_path.name}")

        feature, label_value = _load_feature_and_label(feature_path, label_path)
        
        if first_shape is None:
            first_shape = feature.shape
            first_file = file_name
        elif feature.shape != first_shape:
            raise ValueError(f"数据形状不一致！文件 '{file_name}' 的形状为 {feature.shape}，但之前的（如 '{first_file}'）形状为 {first_shape}。\n请检查数据目录下是否混杂了不同尺寸的数据。")

        features.append(feature.astype(np.float32))
        labels.append(np.array([label_value], dtype=np.float32))

    if not features:
        raise ValueError(f"No feature files found in {feature_dir}")

    stacked_features = np.stack(features, axis=0)
    stacked_labels = np.stack(labels, axis=0)
    return stacked_features, stacked_labels


# 修改 _build_dataset 函数
def _build_dataset(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool,
                   augment: bool = False, augmenter=None) -> ds.Dataset:

    # 如果启用数据增强，对训练数据进行增强
    if augment and augmenter is not None:
        augmented_features = []
        for feat in features:
            # 50% 概率应用增强
            if np.random.random() < 0.5:
                augmented_features.append(augmenter(feat))
            else:
                augmented_features.append(feat)
        features = np.stack(augmented_features, axis=0)

    dataset = ds.NumpySlicesDataset((features, labels), column_names=["data", "label"], shuffle=shuffle)

    # 增加 num_parallel_workers 和 prefetch_size
    # drop_remainder=True 在 Ascend 上能保证 Batch Shape 固定，避免重复编译
    dataset = dataset.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_workers=8,
    )
    return dataset


def prepare_regression_datasets(
    feature_dir: str,
    label_dir: str,
    batch_size: int = 64,
    test_split: float = 0.2,
    val_split: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
    scale_inputs: bool = True,
    input_scale: float = DEFAULT_INPUT_SCALE,
    scale_labels: bool = True,
    label_scale: float = DEFAULT_LABEL_SCALE,
    augment: bool = False,
) -> Tuple[ds.Dataset, Optional[ds.Dataset], Optional[ds.Dataset], DatasetMeta]:
    feature_path = Path(feature_dir).expanduser().resolve()
    label_path = Path(label_dir).expanduser().resolve()

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label directory not found: {label_path}")

    features, labels = _collect_pairs(feature_path, label_path)

    # === 智能归一化逻辑 ===
    # 1. 计算数据实际最大值
    max_val = np.max(np.abs(features))
    actual_input_scale = 1.0

    if scale_inputs:
        # 如果数据中有非零值
        if max_val > 0:
            # 检查传入的 input_scale 是否合理
            # 如果 input_scale 未设置(<=0) 或者 数据最大值远超 input_scale (例如 > 10倍)，
            # 则强制使用 max_val 作为 scale，防止输入爆炸
            if input_scale <= 0 or max_val > input_scale * 2.0:
                print(f"[Dataset] Warning: Data max value ({max_val:.2f}) is significantly larger than input_scale ({input_scale}). Using max value as scale.")
                actual_input_scale = float(max_val)
            else:
                actual_input_scale = float(input_scale)
        
        features /= actual_input_scale

    if scale_labels:
        labels /= float(label_scale)

    num_samples = features.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    # 三分划分: train / val / test = (1 - val_split - test_split) / val_split / test_split
    # 默认: 70% / 15% / 15%
    val_split_idx = int(num_samples * (1.0 - val_split - test_split)) if num_samples > 1 else num_samples
    val_split_idx = max(1, val_split_idx) if num_samples > 1 else val_split_idx

    test_split_idx = int(num_samples * (1.0 - test_split)) if num_samples > 1 else num_samples
    test_split_idx = max(val_split_idx + 1, test_split_idx) if num_samples > 1 else test_split_idx

    train_indices = indices[:val_split_idx]
    val_indices = indices[val_split_idx:test_split_idx]
    test_indices = indices[test_split_idx:]

    # 创建数据增强器
    augmenter = DataAugmenter(enable=augment) if augment else None

    train_dataset = _build_dataset(features[train_indices], labels[train_indices], batch_size, shuffle,
                                    augment=augment, augmenter=augmenter)

    val_dataset: Optional[ds.Dataset] = None
    if val_indices.size > 0:
        val_dataset = _build_dataset(features[val_indices], labels[val_indices], batch_size, False)

    test_dataset: Optional[ds.Dataset] = None
    if test_indices.size > 0:
        test_dataset = _build_dataset(features[test_indices], labels[test_indices], batch_size, False)

    meta = DatasetMeta(
        count=num_samples,
        feature_shape=tuple(features.shape[1:]),
        feature_dir=feature_path,
        label_dir=label_path,
        input_scale=actual_input_scale,
        label_scale=label_scale if scale_labels else 1.0,
        inputs_scaled=scale_inputs,
        labels_scaled=scale_labels,
    )

    return train_dataset, val_dataset, test_dataset, meta


# Backwards-compatible alias for callers expecting the old name.
prepare_datasets = prepare_regression_datasets