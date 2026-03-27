# Qinchuan Case Study: Street View & Eye-Tracking Analysis

本项目是一个结合眼动热力图、街景图像，利用 SAM2（Segment Anything Model 2）进行图像分割，并使用 CLIP 进行语义分类的自动化分析流水线。

## 📁 项目架构

```text
QINCHUAN_CASE_STUDY/
├── data/                  # 核心数据与模型权重 (需从百度网盘下载)
│   ├── CV-眼动数据
│   ├── experiment_data    # 热力图与原始 CSV 记录
│   ├── input_streetview   # 原始街景图像
│   ├── results            
│   └── weights            # CLIP 等本地模型权重
├── output/                # 结果输出目录 (包含已跑好的缓存与最终结果)
├── sam2_repo/             # SAM2 官方代码库分支 (需独立安装)
├── src/                   # 核心源代码 (适配器与数据结构)
│   ├── clip_adapter.py
│   ├── sam2_adapter.py
│   ├── heatmap_extractor.py
│   ├── QINCHUAN_LABELS_MAP.py
│   └── new_data_structures.py
├── batch_run_experiment.py       # 批处理主程序
├── stage1_sam_segmentation.py    # 阶段一：SAM2 分割处理
└── stage2_clip_classification.py # 阶段二：CLIP 分类与结果汇总
```
*(注：项目不同模块代码位于 `src/` 目录下。)*

---

## 🚀 快速开始 (Quick Start)

### 步骤 1：下载数据与预设结果（必做）
为了避免重复计算和配置环境，**请务必先从百度网盘下载完整的 `data` 和 `output` 文件夹**。
* `data/` 包含了甲方原始数据、眼动热力图以及 CLIP 的本地模型权重。
* `output/` 包含了**已经跑好的中间缓存和最终结果**，无需再次消耗算力重复运行。

👉 **百度网盘下载链接：** `[在这里填写你的网盘链接]`
👉 **提取码：** `[填写提取码]`

下载完成后，请将 `data` 、 `output` 和 `sam2_repo` 文件夹放置在本项目根目录下, 并直接覆盖已有文件夹。

### 步骤 2：环境配置
建议使用 Conda 创建独立的虚拟环境：

```bash
conda create -n qinchuan_env python=3.10
conda activate qinchuan_env

# 安装基础依赖
pip install -r requirements.txt
```

**⚠️ 重点提示：SAM2 的独立安装**
由于 SAM2 需要编译部分 CUDA 算子，请进入 `sam2_repo` 目录进行本地安装：
```bash
cd sam2_repo
pip install -e .
cd ..
```

---

## 💻 代码运行说明

如果你需要基于新数据重新运行流水线（或者想测试断点续跑逻辑），可以通过主程序入口执行：

```bash
python batch_run_experiment.py
```

### 处理流程解析
主程序分为两个阶段（可通过代码中的注释开关控制）：

1. **Phase 1 (Stage 1): SAM2 分割 (`main1`)**
   * 读取 `data/input_streetview` 和 `data/experiment_data/gaze_heatmap`。
   * 提取热力核心采样点，调用 SAM2 生成目标物体 Mask。
   * 结果序列化保存至 `output/intermediate_cache`（已在网盘提供）。

2. **Phase 2 (Stage 2): CLIP 语义分类与汇总 (`main2`)**
   * 读取 Phase 1 的 `.pkl` 缓存文件。
   * 针对生成的 Mask，调用 CLIP 进行 Zero-shot 图像分类（映射到中文标签）。
   * 结合像素投票机制进行 Top-K 筛选与去重。
   * 生成最终的可视化结果与 `Global_Analysis_Result_Batch.xlsx` 统计表，保存至 `output/final_results/`。
```

