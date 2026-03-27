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
│   ├── intermediate_cache/# Stage 1 生成的 SAM2 分割缓存 (.pkl文件)
│   ├── final_results/     # 单用户独立分析全量数据
│   │   └── Final_Run_20260101_145115/ # ✨包含 001~040 共40个志愿者的独立数据及全局 Excel 表格
│   └── analysis/          # 多用户叠合终极分析结果
│       └── 20260109_003500/           # ✨100个场景图，每个场景叠合了40个用户数据的完整分析结果
├── sam2_repo/             # SAM2 官方代码库分支 (需独立安装)
├── src/                   # 核心源代码 (模型适配与数据处理)
│   ├── clip_adapter.py           # CLIP 模型封装类：负责接收 SAM2 的 Mask，裁剪图像并进行 Zero-shot 语义分类。
│   ├── sam2_adapter.py           # SAM2 模型封装类：负责加载预训练权重，接收点提示 (Point Prompts) 并输出分割掩码 (Masks)。
│   ├── heatmap_extractor.py      # 热力图处理模块：提取眼动热力图中的核心高关注红区 (Red Zones)，并生成密集的网格采样点。
│   ├── image_processing.py       # 图像基础处理库：负责热力图与街景图的尺寸对齐、裁剪及可视化叠加融合。
│   ├── new_data_structures.py    # 核心数据结构定义：定义了从采样点、候选物体、热力簇到最终验证物体的全套 Dataclass 容器。
│   ├── QINCHUAN_LABELS_MAP.py    # 语义标签字典：将 CLIP 输出的英文描述映射为本项目定义的 10 类中文研究标签。
│   ├── semantic_engine.py        # [已弃用] 旧版 TF/DeepLab 语义分割引擎，保留作参考。
│   ├── data_structures.py        # [已弃用] 旧版核心数据结构.
│   └── viz_util.py               # [辅助模块, 基本用不到] 数据可视化库：用于生成 SCI 论文级别的图表 (人口统计、语义占比等)。
├── batch_run_experiment.py       # 🚀 批处理主程序：控制全量数据的流水线执行，输出各用户独立结果与 Excel 表格。
├── generate_analysis_maps.py     # 🌟 终极聚合脚本：读取 Excel 全局表格，将分散的多用户数据按场景叠合，生成最终的热力与情感分布图。
├── stage1_sam_segmentation.py    # 阶段一：热力区提取 + SAM2 智能分割，生成中间缓存。
├── stage2_clip_classification.py # 阶段二：候选物体筛选 + CLIP 语义分类 + 结果可视化与 Excel 导出。
├── main_process_heatmap_to_object_scores.py # 新版单次运行主程序：SAM2 + CLIP 完整流水线的单图/单次运行版本。
├── demo_sam2_segment.py          # 测试 Demo：用于快速验证 SAM2 提取热力图焦点并进行分割的效果。
├── demo_sam_segment.py           # [历史归档] 第一代 SAM (segment_anything) 模型的单图测试脚本。
├── main_analysis.py              # [已弃用] 早期主流程：基于 TF/DeepLab 进行全局语义分割与热力图交集计算。
└── main_segment_raw_streetscape_results.py  # [已弃用] 早期脚本：用于全量街景图的 DeepLab 语义分割并导出图例与报表。
```
*(注：项目不同模块代码位于 `src/` 目录下。)*

---

## 🚀 快速开始 (Quick Start)

### 步骤 1：下载数据与预设结果（必做）
为了避免重复计算和配置环境，**请务必先从百度网盘下载完整的 `data` 和 `output` 文件夹**。
* `data/` 包含了甲方原始数据、眼动热力图以及 CLIP 的本地模型权重。
* `output/` 包含了**已经跑好的中间缓存和最终结果**，无需再次消耗算力重复运行。

👉 **百度网盘下载链接：** 
`通过网盘分享的文件：QINCHUAN_CASE_STUDY
链接: https://pan.baidu.com/s/1MNpw5_pMmxwFMRPz6Vj03A?pwd=kvi8 提取码: kvi8 
--来自百度网盘超级会员v7的分享`

下载完成后，请将 `data` 、 `output` 和 `sam2_repo` 文件夹放置在本项目根目录下, 并直接覆盖已有文件夹。 当然也可以直接运行原项目

### 步骤 2：环境配置
建议使用 Conda 创建独立的虚拟环境：

```bash
conda create -n qinchuan_env python=3.10
conda activate qinchuan_env
```

**⚠️ 重点提示：SAM2 的独立安装**
由于 SAM2 需要编译部分 CUDA 算子，请进入 `sam2_repo` 目录进行本地安装：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/facebookresearch/sam2.git（或直接下载https://github.com/facebookresearch/sam2）
cd sam2_repo
pip install -e .
cd ..
```

**再安装基础依赖：**
```bash
pip install -r requirements.txt
```

---

## 💻 代码运行与高级配置

### 1. 单图测试 (Debug 模式)
如果你只想测试一张图片的分割与分类效果，请运行单次主程序：
```bash
python main_process_heatmap_to_object_scores.py
```

### 2. 全量批处理运行
如果你需要基于新数据重新运行流水线，生成单用户的详尽数据与全局 Excel 表格，可以通过主程序入口执行：
```bash
python batch_run_experiment.py
```

#### ⚙️ 批处理高级配置 (在 `batch_run_experiment.py` 中修改)
* **断点续跑**：如果程序中断，无需从头开始。在 `if __name__ == "__main__":` 下修改 `start_user` 变量即可从指定志愿者的 ID 继续运行。
* **阶段控制**：主程序分为 `main1()` (Stage 1: SAM2 缓存生成) 和 `main2()` (Stage 2: CLIP 分类汇总)。
* **可视化开关 (Visualization Toggles)**：在 `main2()` 函数内部，可通过 `viz_settings` 字典控制可视化输出。**⚠️ 警告：开启 `viz_step1` 会产生海量图片，仅供 Debug 使用。**

### 3. 多用户数据叠合与终极分析出图
当通过批处理程序获得 `Global_Analysis_Result_Batch.xlsx` 表格后，运行聚合脚本生成论文所需的场景叠合图：
```bash
python generate_analysis_maps.py
```
*(注：该脚本将自动解析 Excel 中的 WKT 空间多边形数据和情感评分，生成以每个场景为单位的全局关注度热力图和情感分布图。)*

---

## 📊 核心输出数据说明

如果需要查阅最终的研究数据，请直接访问 `output/` 目录下的相关文件夹。它们主要分为两类：

### 1. 单用户全量数据与全局表格
**路径**：`output/final_results/Final_Run_20260101_145115/`
* **生成来源**：由 `batch_run_experiment.py` 批处理主程序运行生成。
* **包含内容**：
  * **001 ~ 040 文件夹**：包含了 40 位志愿者每人独立的详细可视化分析图片。
  * **数据报表 (`Global_Analysis_Result_Batch.xlsx`)**：汇总了所有图片中识别出的物体面积、SAM 置信度 (Score A) 和 注意力密度 (Score B)。
  * **✨ GIS 友好**：表格中包含 `Geometry_JSON` 字段，已将图像的 Mask 轮廓转换为 **WKT (Well-Known Text)** 格式多边形，可直接导入 ArcGIS / QGIS 或其它空间分析软件进行后续处理。

### 2. 多用户叠合终极结果
**路径**：`output/analysis/20260109_003500/`
* **生成来源**：由 `generate_analysis_maps.py` 聚合出图脚本生成。
* **包含内容**：这是本研究的最终核心产出，包含了 **100个场景图** 的完整分析结果。每张场景图上都**叠合了 40 个用户**的眼动与语义识别数据，是对空间节点视觉注意力分布的最全面总结。
```
