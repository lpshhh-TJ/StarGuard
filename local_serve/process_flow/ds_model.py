# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.

import logging
import numpy as np
import mindspore
from mindspore import nn, ops, save_checkpoint # 导入保存接口
from mindspore import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor


class MultiScaleConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 优化：针对 40x40 小图输入，去掉 7x7 和 9x9 大核，节省内存
        # 增加 1x1 卷积，保留局部特征
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same')
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='same')
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, pad_mode='same')
        # 拼接操作
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        # 并行计算
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        # 沿通道轴拼接 (N, C, H, W) -> axis=1
        return self.concat((c1, c3, c5))


class Network(nn.Cell):
    def __init__(self):
        super().__init__()

        self.features = nn.SequentialCell([

            # ===== Block 1 =====
            MultiScaleConv(in_channels=2, out_channels=32),  # -> 96 channels
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, pad_mode='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 只做一次下采样
            nn.AvgPool2d(kernel_size=2, stride=2),  # 40 → 20

            # ===== Block 2 =====
            nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, pad_mode='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 第二次轻度下采样
            nn.AvgPool2d(kernel_size=2, stride=2),  # 20 → 10

            # ===== Block 3 =====
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])

        # 用全局平均池化代替 flatten
        self.global_pool = nn.AvgPool2d(10,10)


        self.regressor = nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(256, 256),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.8),  # 防止过拟合
            nn.Dense(256, 64),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.8),  # 防止过拟合
            nn.Dense(64, 1)
        ])

    def construct(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x


def _match_label_shape(label, target_shape):
    if label.shape != target_shape:
        label = ops.reshape(label, target_shape)
    return label


@mindspore.jit
def forward_fn(model, data, label, loss_fn):
    prediction = model(data)
    label = _match_label_shape(label, prediction.shape)
    loss = loss_fn(prediction, label)
    return loss, prediction


def train_step(grad_fn, model, data, label, optimizer, loss_fn):
    (loss, _), grads = grad_fn(model, data, label, loss_fn)
    optimizer(grads)
    return loss


def train(model, dataset, loss_fn, optimizer, epochs: int = 1, log_interval: int = 10, save_path: str = "model.ckpt",
          val_dataset=None, early_stopping_patience: int = 10):
    """
    训练模型，支持验证集和早停

    Args:
        val_dataset: 验证集，用于早停判断
        early_stopping_patience: 早停耐心值，验证集loss连续patience轮不下降则停止
    """
    if dataset is None:
        raise ValueError("Training dataset is required.")

    # 封装 Model
    train_net = Model(model, loss_fn=loss_fn, optimizer=optimizer)

    # 配置保存回调 (每轮保存一次)
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(), keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="distance_net", directory="./checkpoints", config=config_ck)

    # 日志回调
    loss_cb = LossMonitor(per_print_times=log_interval)
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())

    logging.info(f"Starting training on Ascend...")
    logging.info(f"Total steps per epoch: {dataset.get_dataset_size()}")

    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # 如果没有验证集，使用原有训练方式
    if val_dataset is None:
        train_net.train(epochs, dataset, callbacks=[ckpoint_cb, loss_cb, time_cb], dataset_sink_mode=True)
    else:
        # 有验证集时，手动逐epoch训练以支持早停和学习率动态调整
        logging.info(f"Training with early stopping (patience={early_stopping_patience})...")
        logging.info(f"Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

        # ReduceLROnPlateau 参数
        lr_scheduler_patience = 3  # 验证集loss 3轮不下降则降低学习率
        lr_factor = 0.5  # 学习率衰减因子
        min_lr = 1e-6  # 最小学习率
        scheduler_patience_counter = 0

        # 获取当前学习率 Parameter
        lr_param = optimizer.learning_rate
        if isinstance(lr_param, (int, float)):
            current_lr = float(lr_param)
        else:
            # 如果是 Tensor/Parameter，获取其值
            current_lr = float(lr_param.asnumpy())

        for epoch in range(epochs):
            model.set_train(True)
            train_net.train(1, dataset, callbacks=[loss_cb], dataset_sink_mode=True)

            # 验证集评估
            val_metrics = test(model, val_dataset, loss_fn, label_scale=1.0)
            val_loss = val_metrics.get("loss", float('inf'))

            # ReduceLROnPlateau 逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                scheduler_patience_counter = 0
                # 保存最佳模型
                save_checkpoint(model, save_path)
                logging.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e} [New best model saved]")
            else:
                patience_counter += 1
                scheduler_patience_counter += 1
                logging.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e} [No improvement for {patience_counter}/{early_stopping_patience} epochs]")

                # 如果连续 scheduler_patience_counter 轮验证loss不下降，降低学习率
                if scheduler_patience_counter >= lr_scheduler_patience and current_lr > min_lr:
                    new_lr = max(current_lr * lr_factor, min_lr)

                    # 更新学习率 Parameter 的值
                    lr_param.set_data(mindspore.Tensor(new_lr, dtype=mindspore.float32))

                    logging.info(f"  -> Reducing LR: {current_lr:.2e} -> {new_lr:.2e}")
                    current_lr = new_lr
                    scheduler_patience_counter = 0  # 重置scheduler计数器

            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                logging.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                break
def test(model, dataset, loss_fn, label_scale=1.0):
    model.set_train(False)

    total_loss = 0.0
    total_sq_error = 0.0
    total_abs_error = 0.0
    total_count = 0

    for data in dataset.create_dict_iterator():
        features = data["data"]
        labels_log = data["label"]  # 这是 log(d+eps) / label_scale

        preds_log = model(features)

        # ===== 1. loss 在 log 空间算 =====
        loss = loss_fn(preds_log, labels_log)

        # 如果是 batch tensor，强制求 mean
        if len(loss.shape) > 0:
            loss = loss.mean()

        # 保证是标量
        loss_value = float(loss.asnumpy())
        total_loss += loss_value

        # ===== 2. 反变换回真实距离域 =====
        preds_log_np = preds_log.asnumpy()
        labels_log_np = labels_log.asnumpy()

        # 先还原 scale
        preds_log_np = preds_log_np * label_scale
        labels_log_np = labels_log_np * label_scale

        # exp 还原
        eps = 1e-3
        preds = np.exp(preds_log_np) - eps
        labels = np.exp(labels_log_np) - eps

        # clip
        preds = np.clip(preds, 0.0, 10.0)

        # 展平
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        # ===== 3. 计算真实距离域误差 =====
        sq_error = np.sum((preds - labels) ** 2)
        abs_error = np.sum(np.abs(preds - labels))

        total_sq_error += sq_error
        total_abs_error += abs_error
        total_count += len(preds)

    mean_loss = total_loss / total_count
    rmse = np.sqrt(total_sq_error / total_count)
    mae = total_abs_error / total_count

    logging.info(f"Test metrics: loss={mean_loss:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")

    return {
        "loss": float(mean_loss),
        "rmse": float(rmse),
        "mae": float(mae)
    }

    if dataset is None:
        print("Test dataset not provided; skipping evaluation.")
        return {}

    model.set_train(False)
    num_batches = dataset.get_dataset_size()
    if num_batches == 0:
        print("Test dataset empty; skipping evaluation.")
        return {}

    total_loss = 0.0
    sq_error = 0.0
    abs_error = 0.0
    sample_count = 0

    for batch in dataset.create_dict_iterator():
        data = batch["data"]
        label = batch["label"]
        prediction = model(data)
        label = _match_label_shape(label, prediction.shape)
        loss = loss_fn(prediction, label)
        total_loss += loss.asnumpy()

        pred_log = prediction.asnumpy().reshape(-1) * label_scale
        label_log = label.asnumpy().reshape(-1) * label_scale

        # Decode from log-distance space and clip to [0, 10]
        pred_np = np.clip(np.exp(pred_log) - 1e-3, 0.0, 10.0)
        label_np = np.clip(np.exp(label_log) - 1e-3, 0.0, 10.0)
        diff = pred_np - label_np
        sq_error += np.square(diff).sum()
        abs_error += np.abs(diff).sum()
        sample_count += pred_np.size

    mse = sq_error / max(1, sample_count)
    rmse = np.sqrt(mse)
    mae = abs_error / max(1, sample_count)
    mean_loss = total_loss / num_batches

    mean_loss = float(mean_loss)
    rmse = float(rmse)
    mae = float(mae)


    logging.info(f"Test metrics: loss={mean_loss:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
    return {"loss": mean_loss, "rmse": rmse, "mae": mae}