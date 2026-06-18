# **人工智能纯源代码 & 环境配置**

> **项目全称**：基于改进注意力机制与边界监督的显著性目标检测  
> **英文标题**：Improved Salient Object Detection with Enhanced Attention and Edge Supervision  
> **领域**：计算机图形学 / 显著性目标检测 (SOD) / 深度学习

---

## 📦 目录结构

```
图形学/
├── README.md                          # 本文件 — 项目说明 + 环境配置指南
├── requirements.txt                   # pip 依赖清单
│
├── clean_source/                      # ✅ 改进版核心代码（推荐使用）
│   ├── dataset.py                     #   数据加载与增强（TTA 多尺度变换）
│   ├── train.py                       #   训练脚本（含边界损失监督）
│   ├── test.py                        #   测试/推理脚本（含 TTA 推理融合）
│   ├── model/
│   │   ├── __init__.py
│   │   ├── get_model.py               #   模型工厂函数
│   │   └── icon/
│   │       ├── icon.py                #   ICON 网络主结构（含边界预测分支）
│   │       ├── modules.py             #   CBAM / smAR / RRGN / FPN / PSP 模块
│   │       ├── pvtv2_encoder.py       #   PVTv2 编码器 ⭐ 推荐
│   │       ├── swin_encoder.py        #   Swin Transformer 编码器
│   │       ├── resnet_encoder.py      #   ResNet 编码器
│   │       ├── vgg_encoder.py         #   VGG-16 编码器
│   │       └── cyclemlp_encoder.py    #   CycleMLP 编码器
│   └── evaltool/
│       ├── sod_eval.py                #   评估工具（F/S/E-measure / MAE）
│       └── sod_valid_sod.py           #   SOD 验证脚本
│
└── TAMF_FRNet/                       # 基准模型（原始 ATMF_FRNet）
    ├── LICENSE                        #   MIT 许可证
    ├── main/
    │   ├── dataset.py
    │   ├── train.py
    │   ├── test.py
    │   └── model/                     #   原始 ICON 模型代码
    │       ├── __init__.py
    │       ├── get_model.py
    │       └── icon/                  #   8个模型文件
    │           ├── icon.py
    │           ├── modules.py
    │           ├── dataset.py
    │           ├── pvtv2_encoder.py
    │           ├── swin_encoder.py
    │           ├── resnet_encoder.py
    │           ├── vgg_encoder.py
    │           └── cyclemlp_encoder.py
    └── util/
        ├── evaltool/                  #   评估工具（含 .sh 脚本）
        │   ├── metrics.py
        │   ├── sod_eval.py
        │   ├── sod_valid_sod.py
        │   ├── run_sod_eval.sh
        │   └── run_sod_valid.sh
        └── scripts/                   #   训练 / 测试启动脚本
            ├── train_icon.sh
            ├── test_icon.sh
            └── run_sod_eval.sh
```

---

## 🔬 技术改进要点

本项目在 **[ATMF_FRNet](https://github.com/zbbany/ATMF_FRNet)**（PLoS ONE 2026）的基础上，针对两大挑战提出三项改进：

| # | 改进点 | 技术方案 | 在代码中的位置 |
|---|--------|---------|---------------|
| 1 | **测试时增强（TTA）** | 多尺度变换（0.75×/1.25×）+ 水平翻转融合 | `clean_source/test.py` |
| 2 | **CBAM 注意力替换** | 将 smAR 通道注意力 → CBAM（通道+空间注意力） | `clean_source/model/icon/modules.py` |
| 3 | **边界损失监督** | Sobel 算子提取边界真值 → 联合多任务训练 | `clean_source/train.py` + `icon.py` |

---

## ⚙️ 环境配置

### 方式一：Conda 环境（推荐）

```bash
# 1. 创建虚拟环境
conda create --name ICON python=3.8.5
conda activate ICON

# 2. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. （可选）安装 NVIDIA Apex 混合精度训练
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 方式二：pip 直接安装

```bash
pip install torch==1.10.0 torchvision==0.11.0
pip install -r requirements.txt
```

### 环境验证

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
```

---

## 🚀 快速开始

### 1. 下载数据与权重（必需，未包含在代码目录中）

- **数据集**：[百度网盘 提取码:SOD1](https://pan.baidu.com/s/1LQ3v7Xc5dqkn_i-b9wXAYg) | [Google Drive](https://drive.google.com/file/d/1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu/view?usp=sharing)
- **预训练权重**：[百度网盘 提取码:SOD1](https://pan.baidu.com/s/1m1GOcd1bHkIEwfOOD4y7MQ) | [Google Drive](https://drive.google.com/file/d/1L_wWTvscAhkhnRteg_UX1laD66ItLMMG/view?usp=drive_link)

解压后按如下结构放置：
```
项目根目录/
├── datasets/          # 数据集（DUTS, ECSSD, HKU-IS, PASCAL-S, DUT-OMRON, SOC, SOD）
├── checkpoint/        # 模型权重（Backbone/ + ICON/）
├── clean_source/      # ← 改进版代码
└── TAMF_FRNet/        # ← 基准模型
```

### 2. 训练（改进版）

```bash
cd clean_source
python train.py --data_root ../datasets --save_path ../checkpoint/ICON/ICON-P
```

### 3. 测试

```bash
cd clean_source
python test.py --data_root ../datasets --pth_path ../checkpoint/ICON/ICON-P/model-best.pth --save_path ./results
```

### 4. 评估

```bash
cd clean_source/evaltool
python sod_eval.py --pred_root ../results --gt_root ../../datasets/DUTS-TE/mask
```

---

## 📊 实验性能（DUTS-TE 测试集）

| 方法 | MAE ↓ | maxF↑ (β²=0.3) | 骨干网络 |
|------|-------|----------------|---------|
| BASNet | 0.048 | 0.860 | ResNet-34 |
| EGNet | 0.040 | 0.889 | ResNet-50 |
| F3Net | 0.036 | 0.898 | ResNet-50 |
| ICON-R | 0.0364 | 0.8322 | ResNet-50 |
| ICON-P | 0.0260 | 0.9346 | PVTv2-B4 |
| ICON-P + TTA | 0.0257 | 0.9388 | PVTv2-B4 |
| **ICON-P + TTA + CBAM + Edge** | **0.0287** | **0.9373** | PVTv2-B4 |

---

## 📖 论文引用

```bibtex
@article{wei2024improved,
  title={Improved Salient Object Detection with Enhanced Attention and Edge Supervision},
  author={Geng Wei and Mi Zhou and Jian Sun and Xiao Shi and Ming Yin and Xinran Zhao and Xueyao Lin},
  year={2024}
}

@article{wei2026robust,
  title={Robust Salient Object Detection Based on Triple Attention-guided Multi-resolution Fusion and Feature Refinement},
  author={Geng Wei and Mi Zhou and Jian Sun and Xiao Shi and Ming Yin and Xinran Zhao and Xueyao Lin},
  journal={PLoS ONE},
  year={2026}
}
```

---

## 📄 许可证

本项目代码基于 [MIT License](TAMF_FRNet/LICENSE) 开源。

---

## 📝 文件统计

| 目录 | Python文件 | Shell脚本 | 说明 |
|------|-----------|----------|------|
| `clean_source/` | 13 | 0 | 改进版核心代码 |
| `TAMF_FRNet/` | 18 | 5 | 基准模型代码 + 工具脚本 |
| **合计** | **31** | **5** | **总计 36 个源代码文件** |

> ⚠️ **注意**：此目录仅包含纯源代码和配置文档。**数据集（~2.1GB）和模型权重（~11.7GB）不在此目录中**，请从上方的网盘链接下载。
