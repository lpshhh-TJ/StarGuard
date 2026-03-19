## Copyright 2026 pzihan. Licensed under the MIT License.

import os
import sys
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context, Model, save_checkpoint, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 添加当前目录到 sys.path，确保可以导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.sisfall_resnet import SisFallResNet

# ================= 配置区域 =================
DATA_DIR = 'dataset'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE_TARGET = "Ascend" # 如果在ModelArts Ascend上运行，请改为 "Ascend"；如果在GPU上，改为 "GPU"
# ===========================================

class SisFallDataset:
    """
    自定义数据集类，用于加载预处理后的Numpy数据
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

def create_dataset(data, labels, batch_size=32, is_train=True):
    """
    创建MindSpore数据集
    """
    dataset_generator = SisFallDataset(data, labels)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=is_train)
    
    # 类型转换
    type_cast_op = ds.transforms.TypeCast(mindspore.int32)
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def evaluate_metrics(model, network, X_val, y_val, batch_size=64):
    """
    计算验证集的 Precision、Recall、F1 等指标
    """
    # 对验证集进行预测
    all_preds = []
    num_samples = len(X_val)
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_data = Tensor(X_val[i:batch_end].astype(np.float32))
        
        # 使用网络进行预测
        output = network(batch_data)
        # 获取预测类别（取概率最大的类）
        preds = output.argmax(axis=1).asnumpy()
        all_preds.extend(preds)
    
    all_preds = np.array(all_preds)
    
    # 计算各项指标
    precision = precision_score(y_val, all_preds, zero_division=0)
    recall = recall_score(y_val, all_preds, zero_division=0)
    f1 = f1_score(y_val, all_preds, zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_val, all_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds
    }

def train():
    # 1. 设置运行环境
    context.set_context(mode=context.GRAPH_MODE, device_target=DEVICE_TARGET)
    
    # 2. 加载数据
    print("正在加载数据...")
    
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 尝试多种可能的路径
    possible_paths = [
        os.path.join(script_dir, 'dataset'),
        os.path.join(script_dir, 'processed_data'),
        os.path.join(os.getcwd(), 'dataset'),
        os.path.join(os.getcwd(), 'processed_data')
    ]
    
    X_path = ""
    y_path = ""
    found = False
    
    for base_dir in possible_paths:
        X_p = os.path.join(base_dir, 'sisfall_X.npy')
        y_p = os.path.join(base_dir, 'sisfall_y_binary.npy')
        if os.path.exists(X_p) and os.path.exists(y_p):
            X_path = X_p
            y_path = y_p
            print(f"找到数据路径: {base_dir}")
            found = True
            break
            
    if not found:
        print(f"错误: 未找到数据文件 sisfall_X.npy 或 sisfall_y_detail.npy")
        print(f"搜索过的路径包括: {possible_paths}")
        print("请确认您上传的文件夹名称是 'dataset' 还是 'processed_data'，并确保它在正确的目录下。")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    
    # 简单切分训练集和验证集 (80% 训练, 20% 验证)
    # 为了演示简单起见，这里使用随机切分
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split = int(num_samples * 0.8)
    train_indices, val_indices = indices[:split], indices[split:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")
    
    # 创建Dataset对象
    ds_train = create_dataset(X_train, y_train, BATCH_SIZE, is_train=True)
    ds_val = create_dataset(X_val, y_val, BATCH_SIZE, is_train=False) # 验证集
    
    # 3. 定义网络
    network = SisFallResNet(in_channels=6, num_classes=2)
    
    # 4. 定义损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=0.001)
    
    # 5. 定义模型
    model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    # 6. 配置回调函数 (保存模型检查点)
    # 使用ds_train获取步骤数
    steps_per_epoch = ds_train.get_dataset_size()
    
    # 设置模型保存路径
    ckpt_save_dir = "./output"
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * 5, # 每5个epoch保存一次
                                 keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="sisfall_resnet", directory=ckpt_save_dir, config=config_ck)
    
    print("开始训练...")
    # 7. 开始训练
    model.train(EPOCHS, ds_train, callbacks=[LossMonitor(per_print_times=100), TimeMonitor(), ckpoint_cb])
    
    print("训练完成!")
    
    # 8. 验证模型并计算各项指标
    print("\n开始验证...")
    acc = model.eval(ds_val)
    print(f"验证集准确率: {acc['accuracy']:.4f}")
    
    # 计算 Precision、Recall、F1 等指标
    print("\n计算详细评估指标...")
    metrics_result = evaluate_metrics(model, network, X_val, y_val, batch_size=BATCH_SIZE)
    
    print(f"\n=== 验证集评估指标 ===")
    print(f"Precision: {metrics_result['precision']:.4f}")
    print(f"Recall:    {metrics_result['recall']:.4f}")
    print(f"F1 Score:  {metrics_result['f1']:.4f}")
    print(f"\n混淆矩阵:")
    print(metrics_result['confusion_matrix'])
    print(f"\n分类报告:")
    print(classification_report(y_val, metrics_result['predictions'], target_names=['非跌倒', '跌倒']))

    # 保存最终模型
    save_checkpoint(network, os.path.join(ckpt_save_dir, "sisfall_resnet_final.ckpt"))
    print(f"\n模型已保存为 {os.path.join(ckpt_save_dir, 'sisfall_resnet_final.ckpt')}")

if __name__ == "__main__":
    train()
