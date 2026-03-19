# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.

import argparse
import json
from pathlib import Path

import mindspore
from mindspore import context, nn

from dataset import prepare_regression_datasets
from model import Network, train, test
from utils import load_model, print_predictions, save_model, predict_dir


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a regression model on 1x316 feature vectors.")
    parser.add_argument("--feature-dir", type=Path, default=project_root / "dataset/train/features", help="Directory containing .npy feature files")
    parser.add_argument("--label-dir", type=Path, default=project_root / "dataset/train/labels", help="Directory containing .txt label files")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction of samples reserved for test")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of samples reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and splits")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling before splitting the dataset")
    parser.add_argument("--no-scale-inputs", action="store_true", help="Disable scaling input vectors by 1/4096")
    parser.add_argument("--input-scale", type=float, default=0.0, help="Value used to scale input vectors. 0.0 = Auto-detect max.")
    parser.add_argument("--no-scale-labels", action="store_true", help="Disable scaling labels")
    parser.add_argument("--label-scale", type=float, default=1.0, help="Value used to scale labels when enabled")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation for training")
    parser.add_argument("--log-interval", type=int, default=10, help="Print average loss every N steps")
    parser.add_argument("--save-path", type=Path, default=Path("model.ckpt"), help="Checkpoint path for saving the trained model")
    parser.add_argument("--load-checkpoint", action="store_true", help="Reload the saved checkpoint into a new model instance")
    parser.add_argument("--print-predictions", action="store_true", help="Print a few sample predictions after training")
    parser.add_argument("--prediction-limit", type=int, default=5, help="Maximum number of predictions to display")
    parser.add_argument("--predict-dir", type=Path, default=None, help="Directory containing .npy files to run predictions on")
    parser.add_argument("--predict-out", type=Path, default=None, help="Output directory to write .txt prediction files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

# 强制设置为 Ascend
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    train_dataset, val_dataset, test_dataset, meta = prepare_regression_datasets(
        feature_dir=args.feature_dir,
        label_dir=args.label_dir,
        batch_size=args.batch_size,
        test_split=args.test_split,
        val_split=args.val_split,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        scale_inputs=not args.no_scale_inputs,
        input_scale=args.input_scale,
        scale_labels=not args.no_scale_labels,
        label_scale=args.label_scale,
        augment=args.augment,
    )

    # === 保存配置提前 ===
    # 只要数据集准备好，就立即保存这一组参数的 scale，防止训练中途崩溃导致 predict 拿不到正确的 scale
    if args.save_path:
        config_path = args.save_path.with_suffix(".json")
        config_data = {
            "input_scale": meta.input_scale,
            "label_scale": meta.label_scale
        }
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        print(f"Saved training config to {config_path} (input_scale={meta.input_scale:.2f}, label_scale={meta.label_scale})")

    step_size = train_dataset.get_dataset_size()

    # 创建一个动态学习率的 Parameter，用于 ReduceLROnPlateau
    # 使用 Parameter 可以在训练过程中修改学习率
    import mindspore as ms
    from mindspore import Parameter
    initial_lr = args.learning_rate  # 默认 1e-3
    lr_schedule = Parameter(ms.Tensor(initial_lr, dtype=ms.float32), name="learning_rate")
    
    model = Network()
    loss_fn = getattr(nn, 'SmoothL1Loss', None)
    if loss_fn is not None:
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        # Fallback for older MindSpore: HuberLoss
        loss_fn = nn.HuberLoss(delta=1.0)
    # 将动态学习率传入优化器，添加 L2 正则化防止过拟合
    optimizer = mindspore.nn.Adam(model.trainable_params(), learning_rate=lr_schedule, weight_decay=1e-4)

    # 训练，带验证集和早停
    train(model, train_dataset, loss_fn, optimizer, epochs=args.epochs, log_interval=args.log_interval,
          save_path=str(args.save_path), val_dataset=val_dataset, early_stopping_patience=10)

    metrics = test(model, test_dataset, loss_fn, label_scale=meta.label_scale)
    if metrics:
        print(f"Evaluation metrics: {metrics}")

    if args.save_path:
        save_model(model, str(args.save_path))
        # Config was already saved at the start

    if args.load_checkpoint:
        reloaded_model = Network()
        load_model(reloaded_model, str(args.save_path))

    if args.print_predictions:
        target_dataset = test_dataset or train_dataset
        print_predictions(model, target_dataset, label_scale=meta.label_scale, limit=args.prediction_limit)

    # Batch predict .npy files if requested
    if args.predict_dir and args.predict_out:
        predict_dir(model, args.predict_dir, args.predict_out, input_scale=meta.input_scale, label_scale=meta.label_scale)
    elif args.predict_dir or args.predict_out:
        print("Both --predict-dir and --predict-out must be provided to run batch predictions.")
    


if __name__ == "__main__":
    main()