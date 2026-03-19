# SisFall 跌倒检测模型（基于 MindSpore）

本仓库实现了基于 ResNet 的跌倒检测模型，使用 **SisFall 公开数据集** 作为训练和评估数据。

> ✅ **重点说明：本项目采用的是 SisFall 公开数据集**（可自行在网上寻找）。

---

## 📌 项目结构

- `preprocess_sisfall.py`：负责从原始 SisFall 数据中读取、清洗、滑动窗口切片，并保存为 `*.npy` 格式。
- `train_binary.py`：基于预处理数据训练二分类（跌倒 / 非跌倒）模型，并评估 Precision/Recall/F1。
- `model/sisfall_resnet.py`：ResNet 1D 网络结构定义（适配 6 通道加速度/陀螺数据）。
- `processed_data/`：预处理输出目录（`sisfall_X.npy`、`sisfall_y_binary.npy`、`sisfall_y_detail.npy`）。
- `output/`：训练过程中保存的 checkpoint 文件。

---

## ✅ 依赖环境

本项目使用 MindSpore 训练（目前代码以 MindSpore API 编写）。

建议在虚拟环境中安装：

```bash
pip install mindspore -f https://www.mindspore.cn/zh-CN/install
pip install numpy scikit-learn tqdm
```

> ⚠️ 注意：MindSpore 在不同硬件（Ascend / GPU / CPU）上安装方式不同，请参考官方安装文档。


## 🛠 数据预处理

1. 下载并解压 **SisFall 公开数据集**。
2. 修改 `preprocess_sisfall.py` 中 `DATA_ROOT` 变量为你下载后数据的根目录（包含 `SA01`、`SA02` 等文件夹）。
3. 运行预处理脚本：

```bash
python preprocess_sisfall.py
```

执行后会生成 `processed_data/` 文件夹，包含：

- `sisfall_X.npy`：训练输入（样本数, 6, 窗口长度）
- `sisfall_y_binary.npy`：二分类标签（0 = 非跌倒, 1 = 跌倒）
- `sisfall_y_detail.npy`：34 类活动标签（19 ADL + 15 Fall）

---

## 🧠 模型训练（跌倒/非跌倒二分类）

```bash
python train_binary.py
```

训练会自动加载 `processed_data/` 下的数据，并输出训练过程中的 checkpoint 文件到 `output/`。

### 可配置项（脚本顶部）

- `DEVICE_TARGET`：`Ascend` / `GPU` 等
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` 等

---

## 📌 结果输出

训练完成后，会保存：

- `output/sisfall_resnet_final.ckpt`：最终模型权重
- 训练过程中还会保存多个 checkpoint，如 `sisfall_resnet_1-30_1462.ckpt` 等

---

## 📎 备注

- 若希望做多分类（34 类）训练，可改写训练脚本，使用 `sisfall_y_detail.npy` 作为标签，并将 `SisFallResNet(num_classes=34)`。
- 若使用自己的数据格式，只需保证输入为 `N×6×L`（通道6，窗口长度L）即可。

---

如有模型改进或更复杂训练策略（如 K 折、学习率调度、数据增强等）需求，可在现有框架上扩展。
