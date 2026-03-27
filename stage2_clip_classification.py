# stage2_clip_classification.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import matplotlib
import datetime
import math
import gc
import pickle
import os
import sys
import json
import pandas as pd  # 🔥 新增
from shapely.geometry import Polygon, MultiPolygon # 🔥 新增，用于生成WKT
from shapely import wkt # 🔥 新增
import openpyxl  # 确保 openpyxl 已导入

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 🔥 [字体设置]
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

from src.clip_adapter import CLIPWrapper
from src.QINCHUAN_LABELS_MAP import QINCHUAN_LABELS_MAP
from src.new_data_structures import HeatmapCluster, SamplingPoint, ProposedObject, ValidatedObject, \
    StreetViewAnalysisResult
from src.heatmap_extractor import HeatmapRedZoneExtractor

CLIP_CANDIDATE_LABELS = list(QINCHUAN_LABELS_MAP.keys())
# 获取所有唯一的中文类别列表 (10类)
UNIQUE_CATEGORIES = sorted(list(set(QINCHUAN_LABELS_MAP.values())))


# ==============================================================================
# 🔥 新增辅助函数：IoU 计算与 Top-K 聚类筛选
# ==============================================================================
def mask_to_wkt(mask):
    """
    将二值 Mask 转换为 WKT (Well-Known Text) 格式的多边形，
    便于后续存入 Excel 并被 GIS 软件读取。
    """
    if mask is None: return "EMPTY"

    # 提取轮廓
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        if len(cnt) >= 3:  # 至少3个点构成多边形
            # 转换为 (x, y) 列表
            points = cnt.squeeze().tolist()
            if len(points) >= 3:
                polys.append(Polygon(points))

    if not polys:
        return "EMPTY"

    if len(polys) == 1:
        return polys[0].wkt
    else:
        return MultiPolygon(polys).wkt


def calculate_iou(mask1, mask2):
    """计算两个布尔 Mask 的 IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


# ==============================================================================
# 🔥 新增核心算法：基于像素投票与父级扩张的 Top-K 选择
# ==============================================================================

def calculate_containment(container_mask, core_mask):
    """
    计算 core_mask 有多少比例在 container_mask 内部
    Containment = Area(Intersection) / Area(Core)
    """
    core_area = np.sum(core_mask)
    if core_area == 0: return 0.0

    intersection = np.logical_and(container_mask, core_mask).sum()
    return intersection / core_area


def select_topk_objects(candidates, k=2, iou_threshold=0.85):
    """
    [优化版 v2] 基于像素投票 + 最大父级轮廓搜索
    逻辑：
    1. 叠加所有 Mask 形成频率图。
    2. 提取高频重叠区域作为 'Core Seeds'。
    3. 针对每个 Seed，寻找能包含它的 'Largest Parent Object' (最大轮廓)。
    4. 对找到的 Parents 进行去重和 Top-K 排序。
    """
    if not candidates:
        return []

    # -------------------------------------------------------
    # 1. 像素投票 (Pixel Voting)
    # -------------------------------------------------------
    # 获取 mask 尺寸 (假设所有 mask 尺寸一致，取第一个)
    h, w = candidates[0].mask.shape
    frequency_map = np.zeros((h, w), dtype=np.float32)

    # 简单的加权累加：Score高的Mask权重稍微大一点，防止低分Mask干扰核心
    # 或者直接 +1 (硬投票)。这里采用 硬投票 + 弱Score加权
    for cand in candidates:
        weight = 1.0
        if cand.score_a > 0.9: weight = 1.2
        frequency_map += (cand.mask.astype(np.float32) * weight)

    max_freq = np.max(frequency_map)
    if max_freq == 0: return []

    # -------------------------------------------------------
    # 2. 提取核心 (Core Extraction)
    # -------------------------------------------------------
    # 阈值：只有由于一半以上的 Mask (或权重) 覆盖的区域才算核心
    # 动态阈值：max_freq * 0.4 (稍微放宽一点，保证能找到核心)
    core_thresh_val = max(1.0, max_freq * 0.4)
    core_binary = (frequency_map >= core_thresh_val).astype(np.uint8)

    # 连通域分析，找出独立的 Core Seeds
    num_labels, labels_im = cv2.connectedComponents(core_binary, connectivity=8)

    # -------------------------------------------------------
    # 3. 父级扩张 (Parent Search)
    # -------------------------------------------------------
    potential_winners = []

    # 跳过 label 0 (背景)
    for label_id in range(1, num_labels + 1):
        seed_mask = (labels_im == label_id)
        seed_area = np.sum(seed_mask)
        if seed_area < 10: continue  # 忽略极噪点

        # 寻找该 Seed 的最佳容器
        valid_parents = []
        for cand in candidates:
            # 检查包含率：候选物体是否包裹了这个核心？
            containment = calculate_containment(cand.mask, seed_mask)

            # 核心必须有 85% 以上被该物体覆盖
            if containment > 0.85:
                valid_parents.append(cand)

        if not valid_parents:
            continue

        # 🔥 [关键逻辑] 如何在所有父级中选择？
        # 用户需求：找到"最大轮廓"。
        # 策略：按面积降序排列，但过滤掉 Score 太低的物体（防止选到巨大的背景噪声）
        # 也可以结合 Score * Area。
        # 这里直接按面积降序，但要求 Score_A > 0.6 (SAM的基本置信度)

        # 过滤低分
        valid_parents = [p for p in valid_parents if p.score_a > 0.6]
        if not valid_parents: continue

        # 按面积从大到小排序
        valid_parents.sort(key=lambda x: np.sum(x.mask), reverse=True)

        # 取最大的那个作为该 Seed 的代表
        best_parent = valid_parents[0]

        # 构造 Group 结构以适配原有接口
        # 这里的 count 我们用覆盖该 Seed 的 mask 数量来近似
        group = {
            'representative': best_parent,
            'members': valid_parents,  # 包含所有能覆盖核的物体
            'count': len(valid_parents),
            'final_score': best_parent.score_a * (1 + 0.1 * len(valid_parents)),  # 简单加分
            'seed_id': label_id
        }
        potential_winners.append(group)

    # -------------------------------------------------------
    # 4. 全局去重 (Global NMS)
    # -------------------------------------------------------
    # 不同 Seed 可能会指向同一个大的 Parent（例如 Seed1是左窗，Seed2是右窗，Parent是整栋楼）
    # 或者两个 Parent 高度重叠。

    # 按 final_score 降序 (或者按面积降序，看你偏好。这里按置信度+热度降序)
    potential_winners.sort(key=lambda x: x['final_score'], reverse=True)

    final_groups = []
    for group in potential_winners:
        if len(final_groups) >= k:
            break

        is_distinct = True
        for selected in final_groups:
            # 计算两个代表物体的 IoU
            iou = calculate_iou(group['representative'].mask, selected['representative'].mask)
            if iou > 0.7:  # 如果重叠超过 70%，认为是同一个物体
                is_distinct = False
                # 如果当前这个比已选的更大，是否要替换？
                # 当前逻辑：先入为主（分数高的先入）。通常分数高且大的比较好。
                break

        if is_distinct:
            final_groups.append(group)

    # 如果没有找到任何 Seed (极端情况)，回退到简单的 Score 排序
    if not final_groups and candidates:
        candidates.sort(key=lambda x: x.score_a, reverse=True)
        best = candidates[0]
        return [{
            'representative': best,
            'members': [best],
            'count': 1,
            'final_score': best.score_a
        }]

    return final_groups


# ==============================================================================
# 🚀 类定义
# ==============================================================================

class Stage2CLIPProcessor:
    def __init__(self, clip_model_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 [Stage 2] Initializing CLIP on {self.device}...")
        try:
            self.clip_tool = CLIPWrapper(model_path_or_name=clip_model_path, device=self.device)
            print("✅ CLIP Model Loaded.")
        except Exception as e:
            print(f"❌ CLIP Init Failed: {e}")
            sys.exit(1)

    def _classify_mask_region(self, full_image_bgr, mask):
        label_text_en, score = self.clip_tool.classify_object(
            image_bgr=full_image_bgr,
            mask=mask,
            text_labels=CLIP_CANDIDATE_LABELS,
            threshold=0.20
        )
        if label_text_en in QINCHUAN_LABELS_MAP:
            chinese_label = QINCHUAN_LABELS_MAP[label_text_en]
        else:
            chinese_label = "其他"
        return chinese_label, score

    def save_analysis_to_excel(self, result_obj, original_img_path, heatmap_path, excel_save_path):
        """
        🔥 [修改版]
        1. 修复路径问题
        2. 字段重构：每个类别下列出所有物体的 Area/ScoreA/ScoreB (逗号分隔)
        """
        # 1. 解析基础元数据
        volunteer_id = os.path.basename(os.path.dirname(heatmap_path))
        image_name = os.path.basename(original_img_path)

        print(f"   📊 Aggregating Data for User: {volunteer_id}, Image: {image_name}...")

        # 2. 修复 CSV 路径查找逻辑
        project_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(project_root, "data", "experiment_data", "csv", f"{volunteer_id}.csv")

        serial_num = "N/A"
        likert_scale = "N/A"

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                target_row = df[df['pics'].astype(str).str.contains(image_name, case=False)]
                if not target_row.empty:
                    serial_num = target_row.iloc[0]['Serial number']
                    likert_scale = target_row.iloc[0]['likert scale']
            except Exception as e:
                print(f"      ⚠️ CSV Read Error: {e}")
        else:
            print(f"      ⚠️ CSV not found: {csv_path}")

        # 3. 初始化行数据
        row_data = {
            "Volunteer_ID": volunteer_id,
            "Image_Name": image_name,
            "Serial_Number": serial_num,
            "Likert_Scale": likert_scale,
            "Total_Objects": len(result_obj.final_objects)
        }

        # 4. 按类别收集数据列表
        # 结构: { "1.传统屋顶": {"areas": [100, 200], "score_a": [0.9, 0.8], "score_b": [0.1, 0.2]}, ... }
        category_data = {cat: {"areas": [], "score_a": [], "score_b": []} for cat in UNIQUE_CATEGORIES}
        geometry_list = []

        # 5. 遍历物体
        for obj in result_obj.final_objects:
            cat = obj.category_name
            if cat in category_data:
                # 收集原始数值
                area_val = float(np.sum(obj.mask))
                sa_val = float(obj.score_a)
                sb_val = float(obj.score_b)

                category_data[cat]["areas"].append(area_val)
                category_data[cat]["score_a"].append(sa_val)
                category_data[cat]["score_b"].append(sb_val)

            # 生成几何元数据
            wkt_str = mask_to_wkt(obj.mask)
            geo_info = {
                "obj_id": int(obj.obj_id),
                "category": cat,
                "score_a": float(obj.score_a),
                "score_b": float(obj.score_b),
                "wkt_geometry": wkt_str
            }
            geometry_list.append(geo_info)

        # 6. 填入统计数据 (逗号分隔字符串)
        # 字段格式: Label_Name, Label_Count, Label_Area_List, Label_ScoreA_List, Label_ScoreB_List
        for i, cat in enumerate(UNIQUE_CATEGORIES):
            data = category_data[cat]
            count = len(data["areas"])

            # 格式化列表为字符串 "100.0, 200.0"
            area_str = ", ".join([f"{x:.1f}" for x in data["areas"]])
            sa_str = ", ".join([f"{x:.3f}" for x in data["score_a"]])
            sb_str = ", ".join([f"{x:.4f}" for x in data["score_b"]])

            # 为了Excel列名简洁，可以用 Label1, Label2... 或者直接用中文名作为前缀
            # 这里按照你的要求，用 LabelX_ 前缀，但为了可读性，保留类别名在 Name 字段
            prefix = f"Label{i + 1}"

            row_data[f"{prefix}_Name"] = cat
            row_data[f"{prefix}_Count"] = count
            row_data[f"{prefix}_Area"] = area_str
            row_data[f"{prefix}_ScoreA"] = sa_str
            row_data[f"{prefix}_ScoreB"] = sb_str

        # 7. 添加 JSON
        row_data["Geometry_JSON"] = json.dumps(geometry_list, ensure_ascii=False)

        # 8. 写入 Excel (单文件模式，每次保存到当前的 Final_Run 文件夹)
        # 🔥 [关键修改] Excel 追加逻辑
        df_new_row = pd.DataFrame([row_data])

        # 确保目录存在
        os.makedirs(os.path.dirname(excel_save_path), exist_ok=True)

        try:
            if os.path.exists(excel_save_path):
                # 如果文件存在，先读取，再拼接，再保存
                # 注意：对于非常大的文件，这可能会变慢，但在几千行级别是安全的
                df_existing = pd.read_excel(excel_save_path)
                df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
                df_combined.to_excel(excel_save_path, index=False)
                # print(f"   ✅ Excel Appended: {excel_save_path} (Total Rows: {len(df_combined)})")
            else:
                # 文件不存在，直接写入
                df_new_row.to_excel(excel_save_path, index=False)
                print(f"   ✅ New Excel Created: {excel_save_path}")
        except Exception as e:
            print(f"   ❌ Excel Save Error: {e}")

    def run_classification_and_viz(self, pkl_path, original_img_path, heatmap_path, output_root):
        """
        读取 .pkl -> 跑 CLIP -> 生成最终可视化
        """
        if not os.path.exists(pkl_path):
            print(f"❌ Pickle file not found: {pkl_path}")
            return

        print(f"🔹 Processing Stage 2 for {os.path.basename(pkl_path)}...")

        # 1. 加载数据
        with open(pkl_path, 'rb') as f:
            result_obj = pickle.load(f)  # result_obj 是 StreetViewAnalysisResult

        # 2. 读取原图 (CLIP需要)
        # 注意：pickle 里没存图，需要重新读
        streetview_bgr = cv2.imread(original_img_path)
        if streetview_bgr is None:
            print("❌ Original Image load failed.")
            return

        # 重新生成用于计算 Score B 的热力灰度图
        # 这一步很快，不影响性能
        extractor = HeatmapRedZoneExtractor(original_img_path, heatmap_path)
        # 只需要 extract 过程产生的 aligned_heatmap
        extractor.extract_red_zones(min_area=50, grid_spacing=30)
        _, aligned_heatmap_bgr = extractor.overlay_img, extractor.aligned_heatmap
        heatmap_gray_for_scoring = cv2.cvtColor(aligned_heatmap_bgr, cv2.COLOR_BGR2GRAY)

        # 准备输出目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = result_obj.image_name.split('.')[0]
        final_output_dir = os.path.join(output_root, f"Final_Run_{timestamp}_{image_id}")
        os.makedirs(final_output_dir, exist_ok=True)
        step1_dir = os.path.join(final_output_dir, "step1_details")
        os.makedirs(step1_dir, exist_ok=True)
        step1_5_dir = os.path.join(final_output_dir, "step1_5_cluster_consensus")
        os.makedirs(step1_5_dir, exist_ok=True)

        # 3. 遍历所有 Proposals 进行 CLIP 分类
        print("   - Running CLIP Classification...")
        base_viz_bg = cv2.cvtColor(extractor.overlay_img, cv2.COLOR_BGR2RGB)  # RGB底图

        # 🔥 [新增] 在此处调用 Step 0 可视化
        self._viz_step0_clusters_points(base_viz_bg, result_obj, final_output_dir)

        for cluster in result_obj.clusters:
            if not cluster.candidates: continue

            for proposal in cluster.candidates:
                # 🔥 这里是核心：此时 proposal.category_name 还是 "Pending"
                # 我们现在调用 CLIP 填充它
                cn_label, _ = self._classify_mask_region(streetview_bgr, proposal.mask)
                proposal.category_name = cn_label  # 更新标签

                # 重新生成 Step 1 可视化 (带正确的标签)
                # 找到对应的 sampling point (通过ID)
                target_pt = next((p for p in cluster.sample_points if p.point_id == proposal.source_point_id), None)
                if target_pt:
                    self._viz_step1_single_proposal(base_viz_bg, cluster, target_pt, proposal, result_obj.clusters,
                                                    step1_dir)

        # 4. 执行 Top-K 筛选 (逻辑与原 Step 3 & 4 一致)
        print("   - Running Consensus & Filtering...")
        SCORE_A_THRESH = 0.60
        TOP_K = 2

        # 清空 final_objects (防止重复运行累加)
        result_obj.final_objects = []

        for cluster in result_obj.clusters:
            if not cluster.candidates: continue

            # 筛选
            top_groups = select_topk_objects(cluster.candidates, k=TOP_K, iou_threshold=0.85)

            # Step 1.5 可视化
            self._viz_step1_5_cluster_consensus(base_viz_bg, cluster, top_groups, step1_5_dir)

            # 生成 Final Object
            rank_in_cluster = 1
            for group in top_groups:
                winner = group['representative']
                if winner.score_a < SCORE_A_THRESH: continue

                # 计算 Score B
                area_pixels = np.sum(winner.mask)
                heatmap_norm = heatmap_gray_for_scoring.astype(float) / 255.0
                score_b = 0.0
                if area_pixels > 0:
                    score_b = np.sum(heatmap_norm[winner.mask]) / area_pixels

                # ==================== 🔥 修复开始 ====================
                # 1. 构造对象时不传 mask 参数
                final_obj = ValidatedObject(
                    obj_id=len(result_obj.final_objects) + 1,
                    original_cluster_id=cluster.cluster_id,
                    score_a=winner.score_a,
                    score_b=score_b,
                    # mask=winner.mask,  <--- ❌ 删掉这行，这里不能传 mask
                    category_name=winner.category_name,
                    color=np.random.rand(3),
                    consensus_count=group['count'],
                    consensus_score=group['final_score']
                )

                # 2. 构造完成后，单独赋值 mask，触发自动压缩 setter
                final_obj.mask = winner.mask
                # ==================== 🔥 修复结束 ====================

                result_obj.final_objects.append(final_obj)
                rank_in_cluster += 1

        # 5. Step 2 最终汇总可视化
        self._viz_step2_final_summary(base_viz_bg, result_obj, final_output_dir)

        # 🔥 6. [修改] 导出 Excel 到当前 Final_Run 文件夹
        # 路径改为: final_output_dir/Analysis_Result.xlsx
        excel_path = os.path.join(final_output_dir, "Analysis_Result.xlsx")
        self.save_analysis_to_excel(result_obj, original_img_path, heatmap_path, excel_path)

        print(f"✅ Stage 2 Completed. Results saved to {final_output_dir}")

    # ==========================================================================
    # 🔥 [修改] 批处理专用入口 (增加可视化控制参数)
    # ==========================================================================
    def batch_run_classification_and_viz(self, pkl_path, original_img_path, heatmap_path, output_dir,
                                         global_excel_path,
                                         viz_step0=False,
                                         viz_step1=False,
                                         viz_step1_5=False,
                                         viz_step2=True):
        """
        批量运行专用，增加可视化开关
        """
        if not os.path.exists(pkl_path): return
        with open(pkl_path, 'rb') as f:
            result_obj = pickle.load(f)

        streetview_bgr = cv2.imread(original_img_path)
        if streetview_bgr is None: return

        # 重建热力图 (仅用于计算)
        extractor = HeatmapRedZoneExtractor(original_img_path, heatmap_path)
        extractor.extract_red_zones(min_area=50, grid_spacing=30)
        _, aligned_heatmap_bgr = extractor.overlay_img, extractor.aligned_heatmap
        heatmap_gray_for_scoring = cv2.cvtColor(aligned_heatmap_bgr, cv2.COLOR_BGR2GRAY)
        base_viz_bg = cv2.cvtColor(extractor.overlay_img, cv2.COLOR_BGR2RGB)

        # 🔥 可视化 Step 0: 簇与采样点
        if viz_step0:
            self._viz_step0_clusters_points(base_viz_bg, result_obj, output_dir)

        # 1. 快速分类
        # 如果需要 Step 1 可视化，需要提前建立文件夹
        step1_dir = os.path.join(output_dir, "step1_details")
        if viz_step1:
            os.makedirs(step1_dir, exist_ok=True)

        for cluster in result_obj.clusters:
            if not cluster.candidates: continue
            for proposal in cluster.candidates:
                if proposal.category_name == "Pending":  # 避免重复计算
                    cn_label, _ = self._classify_mask_region(streetview_bgr, proposal.mask)
                    proposal.category_name = cn_label

                # 🔥 可视化 Step 1: 单个 Proposal
                if viz_step1:
                    target_pt = next((p for p in cluster.sample_points if p.point_id == proposal.source_point_id), None)
                    if target_pt:
                        self._viz_step1_single_proposal(base_viz_bg, cluster, target_pt, proposal, result_obj.clusters,
                                                        step1_dir)

        # 2. 筛选汇总
        SCORE_A_THRESH = 0.60
        TOP_K = 2
        result_obj.final_objects = []

        for cluster in result_obj.clusters:
            if not cluster.candidates: continue
            top_groups = select_topk_objects(cluster.candidates, k=TOP_K, iou_threshold=0.85)

            # 🔥 可视化 Step 1.5: 簇内共识
            if viz_step1_5:
                step1_5_dir = os.path.join(output_dir, "step1_5_cluster_consensus")
                os.makedirs(step1_5_dir, exist_ok=True)
                self._viz_step1_5_cluster_consensus(base_viz_bg, cluster, top_groups, step1_5_dir)

            for group in top_groups:
                winner = group['representative']
                if winner.score_a < SCORE_A_THRESH: continue
                area_pixels = np.sum(winner.mask)
                heatmap_norm = heatmap_gray_for_scoring.astype(float) / 255.0
                score_b = np.sum(heatmap_norm[winner.mask]) / area_pixels if area_pixels > 0 else 0

                # ==========================================
                # 🔥 修复点：ValidObject 初始化逻辑
                # ==========================================
                final_obj = ValidatedObject(
                    obj_id=len(result_obj.final_objects) + 1,
                    original_cluster_id=cluster.cluster_id,
                    score_a=winner.score_a,
                    score_b=score_b,
                    # mask=winner.mask,  <-- 删除这里
                    category_name=winner.category_name,
                    color=np.random.rand(3),
                    consensus_count=group['count'],
                    consensus_score=group['final_score']
                )
                # 后置赋值，触发 Setter 压缩
                final_obj.mask = winner.mask

                result_obj.final_objects.append(final_obj)

        # 3. 🔥 可视化 Step 2: 最终汇总 (由开关控制)
        if viz_step2:
            image_basename = os.path.splitext(os.path.basename(original_img_path))[0]
            self._viz_step2_final_summary(base_viz_bg, result_obj, output_dir,
                                          filename_override=f"{image_basename}_Summary.png")

        # 4. 写入全局 Excel
        self.save_analysis_to_excel(result_obj, original_img_path, heatmap_path, global_excel_path)

        # =========================================================
        # 🔥 [修改] 强制内存回收 (防止 OpenCV/Matplotlib 内存泄漏)
        # =========================================================
        # 1. 显式删除大对象
        del streetview_bgr
        del result_obj
        del base_viz_bg

        if 'heatmap_gray_for_scoring' in locals(): del heatmap_gray_for_scoring
        if 'extractor' in locals(): del extractor

        # 2. 关闭所有 Matplotlib 图表 (双重保险)
        plt.close('all')

        # 3. 强制 Python 进行垃圾回收
        gc.collect()
        # =========================================================

    # ==================================================================================
    #  可视化函数群 (直接复用你已修正好的版本，含 resize 优化)
    # ==================================================================================
    # 请直接将你之前那个包含 resize 和中文标签逻辑的 _viz_step1, _viz_step1_5, _viz_step2 函数复制到这里
    # 为了简洁，我这里只写函数名占位，实际内容与你上一轮修改好的一模一样。

    def _draw_contour(self, ax, cnt, color, **kwargs):
        """绘制 OpenCV 轮廓"""
        coords = cnt.squeeze()
        if len(coords.shape) == 1: coords = coords[np.newaxis, :]
        coords = np.vstack([coords, coords[0]])
        ax.plot(coords[:, 0], coords[:, 1], color=color, **kwargs)

    def _draw_mask_contour(self, ax, mask, color, **kwargs):
        """绘制 Binary Mask 轮廓"""
        m_uint8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            self._draw_contour(ax, c, color, **kwargs)

    def _viz_step0_clusters_points(self, bg_img, result, output_dir):
        """Step 0: 簇与采样点 (全分辨率，单图无内存风险)"""
        fig, ax = plt.subplots(figsize=(16, 10))  # 保持高分辨率
        ax.imshow(bg_img)  # 底图已经是 街景+热力

        total_points = 0
        for c in result.clusters:
            # 簇轮廓
            coords = c.contour_points.squeeze()
            if len(coords.shape) == 1: coords = coords[np.newaxis, :]
            coords = np.vstack([coords, coords[0]])
            ax.plot(coords[:, 0], coords[:, 1], color='cyan', linewidth=1.0, linestyle='--')

            # 簇ID
            # ax.text(c.centroid[0], c.centroid[1], f"C{c.cluster_id}", color='cyan', fontsize=3, fontweight='bold')

            # 采样点：模仿 heatmap_extractor 样式
            for p in c.sample_points:
                # 绘制点：白底红芯
                ax.scatter(p.coords[0], p.coords[1], c='white', s=1, marker='o', zorder=5)  # 白底
                ax.scatter(p.coords[0], p.coords[1], c='red', s=0.5, marker='o', zorder=6)  # 红芯

                # ID文字 (极小)
                # ax.text(p.coords[0] + 6, p.coords[1], str(p.point_id), color='yellow', fontsize=2, zorder=10)
                total_points += 1

        ax.set_title(f"STEP 0: Clusters & Points (Total: {total_points})", fontsize=10)
        ax.axis('off')

        save_path = os.path.join(output_dir, "STEP0_Clusters_and_Points.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)  # 提高DPI保证小字清晰
        plt.close()
        print(f"✅ Saved: {save_path}")

    def _viz_step1_single_proposal(self, bg_img, target_cluster, target_point, proposal, all_clusters, output_dir):
        """Step 1: 单点单图 (最关键的修改)"""
        # 🔥 [内存优化] 降采样倍率 (例如缩小4倍，从 5000x3000 -> 1250x750)
        # 这对于 Debug 预览图完全足够，且能解决 MemoryError
        scale_factor = 1

        # 缩放背景图
        small_bg = cv2.resize(bg_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        h, w = small_bg.shape[:2]

        # 创建画布 (尺寸也相应减小)
        fig, ax = plt.subplots(figsize=(12, 8))  # 稍微调小一点尺寸
        ax.imshow(small_bg)

        # 1. 绘制背景上下文 (其他簇和点变淡)
        for c in all_clusters:
            is_target_cluster = (c.cluster_id == target_cluster.cluster_id)

            # 簇轮廓
            alpha = 1.0 if is_target_cluster else 0.3
            color = 'cyan' if is_target_cluster else 'white'
            ls = '--' if is_target_cluster else ':'
            self._draw_contour(ax, c.contour_points, color, linewidth=1.0, linestyle=ls, alpha=alpha)

            # 簇内采样点
            if is_target_cluster:
                # 如果是当前簇，只高亮当前点，其他点半透明
                for p in c.sample_points:
                    if p.point_id == target_point.point_id:
                        # 当前激活点：大红叉
                        ax.scatter(p.coords[0], p.coords[1], c='red', s=150, marker='x', linewidth=3, zorder=10)
                        # 标注点ID
                        ax.text(p.coords[0] + 10, p.coords[1], f"P{p.point_id}", color='yellow', fontsize=12,
                                fontweight='bold')
                    else:
                        # 簇内其他点：淡白色
                        ax.scatter(p.coords[0], p.coords[1], c='white', s=10, marker='o', alpha=0.5, zorder=5)
            else:
                # 其他簇的点：完全忽略或极淡
                pass

        # 2. 绘制 Mask (需要缩放 Mask)
        # 原始 mask 是 bool 类型 (H_orig, W_orig)
        mask_uint8 = (proposal.mask * 255).astype(np.uint8)
        small_mask = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
        small_mask_bool = small_mask > 127

        if np.any(small_mask_bool):
            # 绘制填充
            fill_mask = np.zeros((h, w, 4))
            fill_mask[small_mask_bool] = [0, 1, 0, 0.4]  # 绿色半透明
            ax.imshow(fill_mask)

            # 绘制轮廓
            self._draw_mask_contour(ax, small_mask_bool, 'lime', linewidth=1.5)

            # 3. 🔥 [修改] 标签显示在 Mask 中心
            M = cv2.moments(small_mask)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 构造标签文本
                label_txt = (f"【{proposal.category_name}】\n"
                             f"Sc:{proposal.score_a:.2f}")

                ax.text(cx, cy, label_txt, color='white', fontsize=6.5,
                        # fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='lime', pad=0.3))

        # 标题简化
        ratio = (np.sum(proposal.mask) / (bg_img.shape[0] * bg_img.shape[1])) * 100
        title = (f"STEP 1 | C{target_cluster.cluster_id} | P{target_point.point_id} | Area: {ratio:.2f}%")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # 保存
        filename = f"S1_C{target_cluster.cluster_id}_P{target_point.point_id}_Sc{proposal.score_a:.2f}.jpg"  # 用jpg省空间
        save_path = os.path.join(output_dir, filename)

        # 降低DPI，因为是海量小图
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)  # 🔥 关键：每一张画完立刻释放内存
        print(f"Saved: {filename}")  # 太多了就不打印了，刷屏

    # 🔥 新增 Step 1.5 可视化
    def _viz_step1_5_cluster_consensus(self, bg_img, cluster, top_groups, output_dir):
        """
        Step 1.5: 单簇汇总图
        功能：显示该簇内所有散点的轮廓（淡色），并高亮 Top-K 轮廓，解释为什么选中它们。
        """
        # 使用高分辨率大图
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.imshow(bg_img)

        # 1. 绘制该簇的热力轮廓 (虚线)
        self._draw_contour(ax, cluster.contour_points, 'cyan', linewidth=1.0, linestyle='--')

        # 2. 绘制所有候选 Proposal 的轮廓 (Ghost Mode)
        # 用极淡的白色细线，展示"探索过程"
        # for candidate in cluster.candidates:
        #     self._draw_mask_contour(ax, candidate.mask, 'white', linewidth=0.5, alpha=0.2)

        # 3. 高亮 Top-K Winners
        colors = ['lime', 'magenta', 'orange']  # Rank 1, 2, 3 的颜色

        for i, group in enumerate(top_groups):
            winner = group['representative']
            color = colors[i % len(colors)]

            # 填充半透明色
            fill_mask = np.zeros((*bg_img.shape[:2], 4))
            fill_mask[winner.mask] = np.concatenate([matplotlib.colors.to_rgb(color), [0.4]])
            ax.imshow(fill_mask)

            # 绘制加粗轮廓
            self._draw_mask_contour(ax, winner.mask, color, linewidth=2.5)

            # 标注核心指标 (解释为什么选中它)
            M = cv2.moments(winner.mask.astype(np.uint8))
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 🔥 [修改] 标签内容 (中文)
                # winner.category_name 已经是中文
                label_txt = (f"RANK {i + 1}\n"
                             f"[{winner.category_name}]\n"
                             f"共识点数: {group['count']}\n"
                             f"得分: {group['final_score']:.2f}")

                # ax.text(cx, cy, label_txt, color='white', fontsize=5, fontweight='bold',  # 字号加大
                #         ha='center', va='center',
                #         bbox=dict(facecolor=color, alpha=0.8, edgecolor='white', boxstyle='round,pad=0.3'))

        ax.set_title(
            f"STEP 1.5: Cluster {cluster.cluster_id} Consensus Analysis (Candidates: {len(cluster.candidates)})",
            fontsize=16)
        ax.axis('off')

        save_path = os.path.join(output_dir, f"C{cluster.cluster_id}_Consensus.png")
        # 使用高 DPI 保存
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close(fig)

    # 🔥 修改后的 Step 2 可视化
    def _viz_step2_final_summary(self, bg_img, result, output_dir, filename_override=None):
        scale = 0.5
        """
        Step 2: 最终汇总 (高清版)
        要求：标注 Cluster ID, Overlap Area, Score B, 字号极小不挡画面
        """
        # 极高分辨率
        fig, ax = plt.subplots(figsize=(24, 16))
        ax.imshow(bg_img)

        # 1. 背景簇轮廓 (淡色虚线)
        for c in result.clusters:
            self._draw_contour(ax, c.contour_points, 'white', linewidth=0.8, linestyle=':', alpha=0.6)
            # 标注簇 ID (仅在簇中心点一个小字)
            # ax.text(c.centroid[0], c.centroid[1], f"C{c.cluster_id}", color='cyan', fontsize=6, fontweight='bold',
            #         alpha=0.8)

        # -----------------------------------------------------------
        # 🔥 [修改逻辑 1] 过滤面积最小的物体
        # -----------------------------------------------------------
        all_objects = result.final_objects
        objects_to_draw = []

        if len(all_objects) > 0:
            # 按 Mask 面积从小到大排序
            # np.sum(obj.mask) 计算 True 的像素数量，即面积
            sorted_objects = sorted(all_objects, key=lambda obj: np.sum(obj.mask))

            # [1:] 表示去掉第一个（即最小的），保留剩下的
            # 如果只有一个物体，切片后为空，符合逻辑
            objects_to_draw = sorted_objects[1:]

        num_objs = len(objects_to_draw)

        # -----------------------------------------------------------
        # 🔥 [修改逻辑 2] 准备颜色映射 (Colormap)
        # -----------------------------------------------------------
        # 使用 'tab20' (20种高对比色) 或 'hsv' (彩虹色)，确保颜色区分度高
        # 如果物体特别多，建议用 'hsv'；一般数量用 'tab20' 视觉效果更好
        cmap = plt.get_cmap('tab20')

        # 2. 绘制最终物体
        for i, obj in enumerate(objects_to_draw):
            # 动态获取颜色 (返回的是 [R, G, B, A] 0-1范围)
            # 避免除以0错误 (虽然上面过滤了逻辑，但为了稳健)
            idx_color = cmap(i / num_objs) if num_objs > 1 else cmap(0)
            rgb_color = idx_color[:3]  # 取 RGB 部分

            # Mask 填充 (淡色)
            fill_mask = np.zeros((*bg_img.shape[:2], 4))
            # 使用动态生成的 rgb_color
            fill_mask[obj.mask] = np.concatenate([rgb_color, [0.4]])  # 透明度 0.4
            ax.imshow(fill_mask)

            # Mask 轮廓 (清晰)
            # 传入动态生成的 rgb_color
            self._draw_mask_contour(ax, obj.mask, rgb_color, linewidth=3)

            # 3. 详细标签 (Small Labels)
            M = cv2.moments(obj.mask.astype(np.uint8))
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 标签格式
                label = (f"【{obj.category_name}】\n"
                         f"ID:{obj.obj_id}\n"
                         f"ScA:{obj.score_a:.2f}\n"
                         f"ScB:{obj.score_b:.2f}")

                # (可选) 如果你想把文字框颜色也改成对应颜色，可以将 edgecolor=obj.color 改为 edgecolor=rgb_color
                # ax.text(cx, cy, label, color='white', fontsize=6, fontweight='bold',
                #         ha='center', va='center',
                #         bbox=dict(facecolor='black', alpha=0.6, edgecolor=rgb_color, linewidth=0.5, pad=0.3))

        ax.set_title(f"STEP 2: Final Summary (N={len(result.final_objects)})", fontsize=12)
        ax.axis('off')

        fname = filename_override if filename_override else "STEP2_Final_Summary.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=400, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    # 配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "data/weights/clip")

    # 示例路径 (需要与 Stage 1 对应)
    TEST_IMG = os.path.join(PROJECT_ROOT, "data/input_streetview/QINCHUAN-62.JPG")
    TEST_HEATMAP = os.path.join(PROJECT_ROOT,"data/experiment_data/gaze_heatmap/001/001_62_eyetrack_heatmap_20250929_190147.png")

    # Stage 1 产生的缓存文件
    CACHE_PKL = os.path.join(PROJECT_ROOT, "output/intermediate_cache/QINCHUAN-62_stage1.pkl")
    FINAL_OUT = os.path.join(PROJECT_ROOT, "output/final_results")

    if not os.path.exists(CLIP_MODEL_PATH):
        print(f"❌ CLIP Model not found.")
        sys.exit(1)

    processor = Stage2CLIPProcessor(CLIP_MODEL_PATH)
    processor.run_classification_and_viz(CACHE_PKL, TEST_IMG, TEST_HEATMAP, FINAL_OUT)