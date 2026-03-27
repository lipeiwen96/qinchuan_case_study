# main_process_heatmap_to_object_scores.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import matplotlib
import datetime
import math
import gc  # 引入垃圾回收
import random  # 引入随机库用于模拟分类结果

matplotlib.use('Agg')  # 避免服务器端弹窗错误
import matplotlib.pyplot as plt
import os
import sys

# 🔥 [修改] 设置中文字体，防止显示乱码 (根据你的系统调整，SimHei 是黑体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from scr.sam2_adapter import SAM2Wrapper
from scr.clip_adapter import CLIPWrapper
from scr.QINCHUAN_LABELS_MAP import QINCHUAN_LABELS_MAP

# 引入数据结构
from scr.new_data_structures import HeatmapCluster, SamplingPoint, ProposedObject, ValidatedObject, StreetViewAnalysisResult
from scr.image_processing import overlay_heatmap_on_streetview
from scr.heatmap_extractor import HeatmapRedZoneExtractor


# 提取用于送给 CLIP 的纯文本列表
CLIP_CANDIDATE_LABELS = list(QINCHUAN_LABELS_MAP.keys())


# ==============================================================================
# 🔥 新增辅助函数：IoU 计算与 Top-K 聚类筛选
# ==============================================================================

def calculate_iou(mask1, mask2):
    """计算两个布尔 Mask 的 IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def select_topk_objects(candidates, k=2, iou_threshold=0.85):
    """
    输入: 一个 Cluster 内所有的候选 ProposedObject 列表
    输出: 筛选出的 Top-K 个最具代表性的 ValidatedObject (数据结构需转换)
    逻辑: 基于 Mask IoU 进行聚类，寻找"共识"最强的物体
    """
    if not candidates:
        return []

    # 1. 按 SAM 分数降序排列，作为贪婪聚类的基准
    sorted_candidates = sorted(candidates, key=lambda x: x.score_a, reverse=True)

    # 2. 聚类 (Groups)
    # 结构: [ {'representative': obj, 'members': [obj, obj...], 'count': 2, 'avg_score': 0.9} ]
    object_groups = []

    assigned_indices = set()

    for i, base_obj in enumerate(sorted_candidates):
        if i in assigned_indices:
            continue

        # 创建新组
        current_group = {
            'representative': base_obj,  # 默认最高分的作为代表
            'members': [base_obj],
            'sum_score': base_obj.score_a
        }
        assigned_indices.add(i)

        # 寻找同类
        for j in range(i + 1, len(sorted_candidates)):
            if j in assigned_indices:
                continue

            compare_obj = sorted_candidates[j]
            iou = calculate_iou(base_obj.mask, compare_obj.mask)

            if iou > iou_threshold:
                # 判定为同一个物体
                current_group['members'].append(compare_obj)
                current_group['sum_score'] += compare_obj.score_a
                assigned_indices.add(j)

        object_groups.append(current_group)

    # 3. 计算共识得分 (Consensus Score)
    # 评分公式: Average_SAM_Score * (1 + 0.5 * log2(Count))
    # 解释: 基础是SAM分数，但出现的次数越多，权重越高。用log是为了防止数量碾压质量。
    for group in object_groups:
        count = len(group['members'])
        avg_score = group['sum_score'] / count
        # 权重因子: 1个点=1.0, 2个点=1.5, 4个点=2.0...
        frequency_boost = 1.0 + 0.5 * math.log2(count)
        group['final_score'] = avg_score * frequency_boost
        group['avg_score'] = avg_score
        group['count'] = count

    # 4. 排序并取 Top-K
    # 按共识得分降序
    sorted_groups = sorted(object_groups, key=lambda x: x['final_score'], reverse=True)

    # 取前 K 个，但要防止 Top 2 之间依然高度重叠 (Inter-Group NMS)
    # 虽然聚类处理了大部分，但以防万一
    final_groups = []
    for group in sorted_groups:
        if len(final_groups) >= k:
            break

        # 检查与已选组的重叠度
        is_distinct = True
        for selected in final_groups:
            iou = calculate_iou(group['representative'].mask, selected['representative'].mask)
            if iou > 0.5:  # 组间 NMS 阈值，如果两个组代表物体重叠超过 50%，丢弃较弱的
                is_distinct = False
                break

        if is_distinct:
            final_groups.append(group)

    return final_groups


# ==============================================================================
# 🚀 类定义
# ==============================================================================

class SaliencySAMProcessor:
    def __init__(self, sam_checkpoint, clip_model_path, device=None):
        """
        初始化 SAM2 和 CLIP
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⏳ Loading Models on {self.device}...")

        # 1. Load SAM 2
        try:
            self.sam_tool = SAM2Wrapper(checkpoint_path=sam_checkpoint, device=self.device)
            print("✅ SAM 2 Model Loaded.")
        except Exception as e:
            print(f"❌ SAM 2 Init Failed: {e}")
            sys.exit(1)

        # 2. 🔥 [NEW] Load CLIP
        try:
            self.clip_tool = CLIPWrapper(model_path_or_name=clip_model_path, device=self.device)
            print("✅ CLIP Model Loaded.")
        except Exception as e:
            print(f"❌ CLIP Init Failed: {e}")
            sys.exit(1)

    def _classify_mask_region(self, full_image_bgr, mask):
        """
        调用 CLIP 识别 Mask 区域，并返回 (中文标签, 置信度)
        """
        # 调用 CLIP Adapter 的核心功能
        # CLIP_CANDIDATE_LABELS 是英文 Prompts 列表
        label_text_en, score = self.clip_tool.classify_object(
            image_bgr=full_image_bgr,
            mask=mask,
            text_labels=CLIP_CANDIDATE_LABELS,
            threshold=0.20  # 稍微降低门槛，确保尽可能分出来
        )

        # 🔥 [修改] 直接映射到 10个中文类别
        if label_text_en in QINCHUAN_LABELS_MAP:
            chinese_label = QINCHUAN_LABELS_MAP[label_text_en]
        else:
            chinese_label = "其他"

        return chinese_label, score

    def run_pipeline(self, image_path, heatmap_path, output_dir_root=None):
        """
        执行完整的 Saliency-to-Object 流程
        """
        image_basename = os.path.basename(image_path)
        image_id = image_basename.split('-')[1].split('.')[0] if '-' in image_basename else "unknown"

        # 1. 设置输出目录
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_output_dir = None
        if output_dir_root:
            current_output_dir = os.path.join(output_dir_root, f"run_{run_timestamp}_{image_id}")
            os.makedirs(current_output_dir, exist_ok=True)
            print(f"📂 Output directory created: {current_output_dir}")

        # 2. 图像加载
        # 🔥 SAM2 Wrapper 接受 BGR (cv2默认读取) 或 RGB，这里保留 BGR 传给 Wrapper
        streetview_bgr = cv2.imread(image_path)
        if streetview_bgr is None:
            print(f"❌ Error: Cannot read image at {image_path}")
            return None, None

        # 用于可视化的 RGB 图像
        streetview_rgb = cv2.cvtColor(streetview_bgr, cv2.COLOR_BGR2RGB)
        analysis_result = StreetViewAnalysisResult(image_name=image_basename)

        # 🔥 注意：SAM2Wrapper 通常在 predict 时传入图像，不需要像 SamPredictor 那样先 set_image
        # 如果你的 Adapter 有 set_image 逻辑，可以在这里调用，但标准用法是在 predict 中传入

        # ==========================================
        # Step 1: 提取红区与采样点
        # ==========================================
        print("🔹 Step 1: Extracting Red Zones...")
        extractor = HeatmapRedZoneExtractor(image_path, heatmap_path)
        contours, all_sample_points = extractor.extract_red_zones(min_area=50, grid_spacing=50)

        # 获取叠加好热力图的RGB图像，作为后续所有可视化的通用底图
        # 注意：overlay_img 是 BGR，需转 RGB
        base_viz_bg = cv2.cvtColor(extractor.overlay_img, cv2.COLOR_BGR2RGB)

        # 获取用于计算的灰度热力图
        _, aligned_heatmap_bgr = extractor.overlay_img, extractor.aligned_heatmap
        heatmap_gray_for_scoring = cv2.cvtColor(aligned_heatmap_bgr, cv2.COLOR_BGR2GRAY)

        # 初始化簇
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cluster = HeatmapCluster(cluster_id=idx, area=int(area), centroid=(cx, cy), contour_points=cnt)
            analysis_result.clusters.append(cluster)

        # 分配点
        global_point_id = 0
        for px, py in all_sample_points:
            heatmap_val = heatmap_gray_for_scoring[py, px]
            for cluster in analysis_result.clusters:
                if cv2.pointPolygonTest(cluster.contour_points, (px, py), False) >= 0:
                    pt = SamplingPoint(
                        point_id=global_point_id,
                        coords=(px, py),
                        intensity=int(heatmap_val),
                        rank=len(cluster.sample_points)
                    )
                    cluster.sample_points.append(pt)
                    global_point_id += 1
                    break

        # 🎨 可视化 Step 0: 概览图
        if current_output_dir:
            self._viz_step0_clusters_points(base_viz_bg, analysis_result, current_output_dir)

        # ==========================================
        # Step 2: 逐点推理与独立可视化 (Split Output)
        # ==========================================
        print(f"🔹 Step 2: Running SAM inference & Saving individual plots...")

        # 为 Step 1 的结果创建子文件夹，避免文件太乱
        step1_dir = os.path.join(current_output_dir, "step1_details")
        os.makedirs(step1_dir, exist_ok=True)

        for cluster in analysis_result.clusters:
            if not cluster.sample_points: continue

            for pt in cluster.sample_points:
                # 🔥 SAM 2 推理核心修改
                try:
                    # Wrapper 接收 [point] 列表
                    # 传入 streetview_bgr，Wrapper 内部处理 RGB 转换
                    best_mask, best_score = self.sam_tool.predict(streetview_bgr, [pt.coords])
                except Exception as e:
                    print(f"⚠️ Inference failed at point {pt.point_id}: {e}")
                    continue

                # 🔥 数据清洗 (关键步骤)
                # 1. 转为布尔型
                mask_bool = best_mask.astype(bool)
                # 2. 降维 (1, H, W) -> (H, W)
                if mask_bool.ndim > 2:
                    mask_bool = mask_bool.squeeze()

                # 🔥 [修改] 调用分类，获取中文标签
                chinese_label, clip_score = self._classify_mask_region(streetview_bgr, mask_bool)

                # 创建 Proposal
                proposal = ProposedObject(
                    proposal_id=f"C{cluster.cluster_id}_P{pt.point_id}",
                    score_a=best_score,
                    source_point_id=pt.point_id,
                    mask=mask_bool,
                    category_name=chinese_label  # 🔥 这里直接存中文标签
                )
                cluster.candidates.append(proposal)

                # 🎨 可视化 Step 1 (单张独立输出)
                if current_output_dir:
                    self._viz_step1_single_proposal(
                        base_viz_bg,
                        cluster,
                        pt,
                        proposal,
                        analysis_result.clusters,
                        step1_dir
                    )

        # ==============================================================================
        # 🔥 Step 3 & 4: 基于聚类共识的 Top-K 筛选 (Logic Update)
        # ==============================================================================
        print("🔹 Step 3 & 4: Clustering & Top-K Consensus Selection...")

        # 创建 Step 1.5 的输出文件夹
        step1_5_dir = os.path.join(current_output_dir, "step1_5_cluster_consensus")
        os.makedirs(step1_5_dir, exist_ok=True)

        # 参数设置
        TOP_K = 2  # 每个热力斑块最多保留几个物体
        SCORE_A_THRESH = 0.80  # 基础门槛

        for cluster in analysis_result.clusters:
            if not cluster.candidates:
                continue

            # 1. 筛选 Top-K 组
            top_groups = select_topk_objects(cluster.candidates, k=TOP_K, iou_threshold=0.85)

            # 🔥 2. 生成 Step 1.5: 单簇汇总图 (Cluster Consensus Summary)
            if current_output_dir:
                self._viz_step1_5_cluster_consensus(
                    base_viz_bg,
                    cluster,
                    top_groups,
                    step1_5_dir
                )

            # 3. 将筛选出的组转换为 ValidatedObject 存入结果
            rank_in_cluster = 1
            for group in top_groups:
                winner_proposal = group['representative']

                # 重新获取大类 (因为 Proposal 里只存了详细标签)
                main_cat = QINCHUAN_LABELS_MAP.get(winner_proposal.category_name, "Unknown")

                # 基础分门槛过滤
                if winner_proposal.score_a < SCORE_A_THRESH:
                    continue

                # Score B 计算
                mask_bool = winner_proposal.mask
                heatmap_norm = heatmap_gray_for_scoring.astype(float) / 255.0
                area_pixels = np.sum(mask_bool)
                score_b = 0.0
                if area_pixels > 0:
                    score_b = np.sum(heatmap_norm[mask_bool]) / area_pixels

                # 创建最终对象
                final_obj = ValidatedObject(
                    obj_id=len(analysis_result.final_objects) + 1,
                    original_cluster_id=cluster.cluster_id,
                    score_a=winner_proposal.score_a,
                    score_b=score_b,
                    mask=winner_proposal.mask,

                    # 🔥 [修改] 填入中文标签
                    category_name=winner_proposal.category_name,  # 直接使用 Proposal 里的中文

                    color=np.random.rand(3),
                    consensus_count=group['count'],
                    consensus_score=group['final_score']
                )
                analysis_result.final_objects.append(final_obj)
                rank_in_cluster += 1

        # 🎨 可视化 Step 2: 最终汇总 (修改后的高分辨率版本)
        if current_output_dir:
            self._viz_step2_final_summary(base_viz_bg, analysis_result, current_output_dir)

        gc.collect()
        return analysis_result, current_output_dir

    # ==================================================================================
    #  可视化辅助函数群 (内存优化版)
    # ==================================================================================

    def _resize_for_viz(self, img, target_width=600):
        """辅助函数：缩放图像以节省可视化内存"""
        h, w = img.shape[:2]
        scale = target_width / float(w)
        dim = (target_width, int(h * scale))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _draw_mask_on_ax(self, ax, img_rgb, mask, color, label_text, title=None):
        """在给定的ax上绘制Mask"""
        ax.imshow(img_rgb)

        if mask is not None:
            # Mask 轮廓
            m_uint8 = (mask * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 使用极细的线绘制轮廓
            for cc in cnts:
                cc = cc.squeeze()
                if len(cc.shape) > 1:
                    cc = np.vstack([cc, cc[0]])
                    ax.plot(cc[:, 0], cc[:, 1], color=color, linewidth=0.8)  # 线宽改小

            # 半透明填充
            fill_mask = np.zeros((*img_rgb.shape[:2], 4))
            fill_mask[mask] = np.concatenate([color, [0.3]])  # 透明度0.3
            ax.imshow(fill_mask)

            # 标签 (极小字体)
            if label_text:
                M = cv2.moments(m_uint8)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    ax.text(cx, cy, label_text, color='white', fontsize=5, fontweight='normal',  # 字号改小
                            ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0.5))

        if title:
            ax.set_title(title, fontsize=6)  # 标题字号改小
        ax.axis('off')

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
            ax.text(c.centroid[0], c.centroid[1], f"C{c.cluster_id}", color='cyan', fontsize=3, fontweight='bold')

            # 采样点：模仿 heatmap_extractor 样式
            for p in c.sample_points:
                # 绘制点：白底红芯
                ax.scatter(p.coords[0], p.coords[1], c='white', s=1, marker='o', zorder=5)  # 白底
                ax.scatter(p.coords[0], p.coords[1], c='red', s=0.5, marker='o', zorder=6)  # 红芯

                # ID文字 (极小)
                ax.text(p.coords[0] + 6, p.coords[1], str(p.point_id), color='yellow', fontsize=2, zorder=10)
                total_points += 1

        ax.set_title(f"STEP 0: Clusters & Points (Total: {total_points})", fontsize=10)
        ax.axis('off')

        save_path = os.path.join(output_dir, "STEP0_Clusters_and_Points.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)  # 提高DPI保证小字清晰
        plt.close()
        print(f"✅ Saved: {save_path}")

    def _viz_step1_single_proposal(self, bg_img, target_cluster, target_point, proposal, all_clusters, output_dir):
        """Step 1: 单点单图 (最关键的修改)"""
        # 🔥 [内存优化] 降采样倍率 (例如缩小4倍，从 5000x3000 -> 1250x750)
        # 这对于 Debug 预览图完全足够，且能解决 MemoryError
        scale_factor = 0.8

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
            ls = '-' if is_target_cluster else ':'
            self._draw_contour(ax, c.contour_points, color, linewidth=1.5, linestyle=ls, alpha=alpha)

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
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)  # 🔥 关键：每一张画完立刻释放内存
        print(f"Saved: {filename}") # 太多了就不打印了，刷屏

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
        self._draw_contour(ax, cluster.contour_points, 'cyan', linewidth=1.5, linestyle='--')

        # 2. 绘制所有候选 Proposal 的轮廓 (Ghost Mode)
        # 用极淡的白色细线，展示"探索过程"
        for candidate in cluster.candidates:
            self._draw_mask_contour(ax, candidate.mask, 'white', linewidth=0.5, alpha=0.2)

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

                ax.text(cx, cy, label_txt, color='white', fontsize=12, fontweight='bold',  # 字号加大
                        ha='center', va='center',
                        bbox=dict(facecolor=color, alpha=0.8, edgecolor='white', boxstyle='round,pad=0.3'))

        ax.set_title(
            f"STEP 1.5: Cluster {cluster.cluster_id} Consensus Analysis (Candidates: {len(cluster.candidates)})",
            fontsize=16)
        ax.axis('off')

        save_path = os.path.join(output_dir, f"C{cluster.cluster_id}_Consensus.png")
        # 使用高 DPI 保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 🔥 修改后的 Step 2 可视化
    def _viz_step2_final_summary(self, bg_img, result, output_dir):
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
            ax.text(c.centroid[0], c.centroid[1], f"C{c.cluster_id}", color='cyan', fontsize=6, fontweight='bold',
                    alpha=0.8)

        # 2. 最终物体
        for obj in result.final_objects:
            # Mask 填充 (淡色)
            fill_mask = np.zeros((*bg_img.shape[:2], 4))
            fill_mask[obj.mask] = np.concatenate([obj.color, [0.2]])  # 透明度降低，不遮挡热力
            ax.imshow(fill_mask)

            # Mask 轮廓 (清晰)
            self._draw_mask_contour(ax, obj.mask, obj.color, linewidth=1.5)

            # 3. 详细标签 (Small Labels)
            M = cv2.moments(obj.mask.astype(np.uint8))
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 🔥 [修改] 标签显示逻辑
                label = (f"【{obj.category_name}】\n"  # 中文类别
                         f"ID:{obj.obj_id} ScB:{obj.score_b:.2f}")

                ax.text(cx, cy, label, color='white', fontsize=10, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor=obj.color, linewidth=0.5, pad=0.3))

        ax.set_title(f"STEP 2: Final Summary (N={len(result.final_objects)})", fontsize=12)
        ax.axis('off')

        save_path = os.path.join(output_dir, "STEP2_Final_Summary.png")
        # 300 DPI 保证放大后字迹清晰
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved High-Res Summary: {save_path}")

    # --- 绘图辅助 ---
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


if __name__ == "__main__":
    # 配置路径
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 改为自动获取当前目录作为根
    # 🔥 这里的路径请确保指向了正确的 SAM2 Checkpoint
    SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam2_repo", "checkpoints", "sam2.1_hiera_large.pt")

    TEST_IMG = os.path.join(PROJECT_ROOT, "data/input_streetview/QINCHUAN-62.jpg")
    TEST_HEATMAP = os.path.join(PROJECT_ROOT,
                                "data/experiment_data/gaze_heatmap/001/001_62_eyetrack_heatmap_20250929_190147.png")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output/test_output")
    # 🔥 配置 CLIP 模型路径
    CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "data/weights/clip")  # 确保这里是你手动下载的路径

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"❌ Weights not found: {SAM_CHECKPOINT}")
        sys.exit(1)

    # 初始化传入两个模型路径
    processor = SaliencySAMProcessor(
        sam_checkpoint=SAM_CHECKPOINT,
        clip_model_path=CLIP_MODEL_PATH
    )

    print(f"🚀 Processing {os.path.basename(TEST_IMG)}...")
    result, final_output_dir = processor.run_pipeline(TEST_IMG, TEST_HEATMAP, OUTPUT_ROOT)

    print(f"\n✨ Pipeline completed. All detailed visualizations saved to: {final_output_dir}")
    print("\n=== Final Analysis Summary ===")
    print(result.summary())# main_process_heatmap_to_object_scores.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import matplotlib
import datetime
import math
import gc  # 引入垃圾回收
import random  # 引入随机库用于模拟分类结果

matplotlib.use('Agg')  # 避免服务器端弹窗错误
import matplotlib.pyplot as plt
import os
import sys

# 🔥 [修改] 设置中文字体，防止显示乱码 (根据你的系统调整，SimHei 是黑体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from scr.sam2_adapter import SAM2Wrapper
from scr.clip_adapter import CLIPWrapper
from scr.QINCHUAN_LABELS_MAP import QINCHUAN_LABELS_MAP

# 引入数据结构
from scr.new_data_structures import HeatmapCluster, SamplingPoint, ProposedObject, ValidatedObject, StreetViewAnalysisResult
from scr.image_processing import overlay_heatmap_on_streetview
from scr.heatmap_extractor import HeatmapRedZoneExtractor


# 提取用于送给 CLIP 的纯文本列表
CLIP_CANDIDATE_LABELS = list(QINCHUAN_LABELS_MAP.keys())


# ==============================================================================
# 🔥 新增辅助函数：IoU 计算与 Top-K 聚类筛选
# ==============================================================================

def calculate_iou(mask1, mask2):
    """计算两个布尔 Mask 的 IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def select_topk_objects(candidates, k=2, iou_threshold=0.85):
    """
    输入: 一个 Cluster 内所有的候选 ProposedObject 列表
    输出: 筛选出的 Top-K 个最具代表性的 ValidatedObject (数据结构需转换)
    逻辑: 基于 Mask IoU 进行聚类，寻找"共识"最强的物体
    """
    if not candidates:
        return []

    # 1. 按 SAM 分数降序排列，作为贪婪聚类的基准
    sorted_candidates = sorted(candidates, key=lambda x: x.score_a, reverse=True)

    # 2. 聚类 (Groups)
    # 结构: [ {'representative': obj, 'members': [obj, obj...], 'count': 2, 'avg_score': 0.9} ]
    object_groups = []

    assigned_indices = set()

    for i, base_obj in enumerate(sorted_candidates):
        if i in assigned_indices:
            continue

        # 创建新组
        current_group = {
            'representative': base_obj,  # 默认最高分的作为代表
            'members': [base_obj],
            'sum_score': base_obj.score_a
        }
        assigned_indices.add(i)

        # 寻找同类
        for j in range(i + 1, len(sorted_candidates)):
            if j in assigned_indices:
                continue

            compare_obj = sorted_candidates[j]
            iou = calculate_iou(base_obj.mask, compare_obj.mask)

            if iou > iou_threshold:
                # 判定为同一个物体
                current_group['members'].append(compare_obj)
                current_group['sum_score'] += compare_obj.score_a
                assigned_indices.add(j)

        object_groups.append(current_group)

    # 3. 计算共识得分 (Consensus Score)
    # 评分公式: Average_SAM_Score * (1 + 0.5 * log2(Count))
    # 解释: 基础是SAM分数，但出现的次数越多，权重越高。用log是为了防止数量碾压质量。
    for group in object_groups:
        count = len(group['members'])
        avg_score = group['sum_score'] / count
        # 权重因子: 1个点=1.0, 2个点=1.5, 4个点=2.0...
        frequency_boost = 1.0 + 0.5 * math.log2(count)
        group['final_score'] = avg_score * frequency_boost
        group['avg_score'] = avg_score
        group['count'] = count

    # 4. 排序并取 Top-K
    # 按共识得分降序
    sorted_groups = sorted(object_groups, key=lambda x: x['final_score'], reverse=True)

    # 取前 K 个，但要防止 Top 2 之间依然高度重叠 (Inter-Group NMS)
    # 虽然聚类处理了大部分，但以防万一
    final_groups = []
    for group in sorted_groups:
        if len(final_groups) >= k:
            break

        # 检查与已选组的重叠度
        is_distinct = True
        for selected in final_groups:
            iou = calculate_iou(group['representative'].mask, selected['representative'].mask)
            if iou > 0.5:  # 组间 NMS 阈值，如果两个组代表物体重叠超过 50%，丢弃较弱的
                is_distinct = False
                break

        if is_distinct:
            final_groups.append(group)

    return final_groups


# ==============================================================================
# 🚀 类定义
# ==============================================================================

class SaliencySAMProcessor:
    def __init__(self, sam_checkpoint, clip_model_path, device=None):
        """
        初始化 SAM2 和 CLIP
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⏳ Loading Models on {self.device}...")

        # 1. Load SAM 2
        try:
            self.sam_tool = SAM2Wrapper(checkpoint_path=sam_checkpoint, device=self.device)
            print("✅ SAM 2 Model Loaded.")
        except Exception as e:
            print(f"❌ SAM 2 Init Failed: {e}")
            sys.exit(1)

        # 2. 🔥 [NEW] Load CLIP
        try:
            self.clip_tool = CLIPWrapper(model_path_or_name=clip_model_path, device=self.device)
            print("✅ CLIP Model Loaded.")
        except Exception as e:
            print(f"❌ CLIP Init Failed: {e}")
            sys.exit(1)

    def _classify_mask_region(self, full_image_bgr, mask):
        """
        调用 CLIP 识别 Mask 区域，并返回 (中文标签, 置信度)
        """
        # 调用 CLIP Adapter 的核心功能
        # CLIP_CANDIDATE_LABELS 是英文 Prompts 列表
        label_text_en, score = self.clip_tool.classify_object(
            image_bgr=full_image_bgr,
            mask=mask,
            text_labels=CLIP_CANDIDATE_LABELS,
            threshold=0.20  # 稍微降低门槛，确保尽可能分出来
        )

        # 🔥 [修改] 直接映射到 10个中文类别
        if label_text_en in QINCHUAN_LABELS_MAP:
            chinese_label = QINCHUAN_LABELS_MAP[label_text_en]
        else:
            chinese_label = "其他"

        return chinese_label, score

    def run_pipeline(self, image_path, heatmap_path, output_dir_root=None):
        """
        执行完整的 Saliency-to-Object 流程
        """
        image_basename = os.path.basename(image_path)
        image_id = image_basename.split('-')[1].split('.')[0] if '-' in image_basename else "unknown"

        # 1. 设置输出目录
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_output_dir = None
        if output_dir_root:
            current_output_dir = os.path.join(output_dir_root, f"run_{run_timestamp}_{image_id}")
            os.makedirs(current_output_dir, exist_ok=True)
            print(f"📂 Output directory created: {current_output_dir}")

        # 2. 图像加载
        # 🔥 SAM2 Wrapper 接受 BGR (cv2默认读取) 或 RGB，这里保留 BGR 传给 Wrapper
        streetview_bgr = cv2.imread(image_path)
        if streetview_bgr is None:
            print(f"❌ Error: Cannot read image at {image_path}")
            return None, None

        # 用于可视化的 RGB 图像
        streetview_rgb = cv2.cvtColor(streetview_bgr, cv2.COLOR_BGR2RGB)
        analysis_result = StreetViewAnalysisResult(image_name=image_basename)

        # 🔥 注意：SAM2Wrapper 通常在 predict 时传入图像，不需要像 SamPredictor 那样先 set_image
        # 如果你的 Adapter 有 set_image 逻辑，可以在这里调用，但标准用法是在 predict 中传入

        # ==========================================
        # Step 1: 提取红区与采样点
        # ==========================================
        print("🔹 Step 1: Extracting Red Zones...")
        extractor = HeatmapRedZoneExtractor(image_path, heatmap_path)
        contours, all_sample_points = extractor.extract_red_zones(min_area=50, grid_spacing=50)

        # 获取叠加好热力图的RGB图像，作为后续所有可视化的通用底图
        # 注意：overlay_img 是 BGR，需转 RGB
        base_viz_bg = cv2.cvtColor(extractor.overlay_img, cv2.COLOR_BGR2RGB)

        # 获取用于计算的灰度热力图
        _, aligned_heatmap_bgr = extractor.overlay_img, extractor.aligned_heatmap
        heatmap_gray_for_scoring = cv2.cvtColor(aligned_heatmap_bgr, cv2.COLOR_BGR2GRAY)

        # 初始化簇
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cluster = HeatmapCluster(cluster_id=idx, area=int(area), centroid=(cx, cy), contour_points=cnt)
            analysis_result.clusters.append(cluster)

        # 分配点
        global_point_id = 0
        for px, py in all_sample_points:
            heatmap_val = heatmap_gray_for_scoring[py, px]
            for cluster in analysis_result.clusters:
                if cv2.pointPolygonTest(cluster.contour_points, (px, py), False) >= 0:
                    pt = SamplingPoint(
                        point_id=global_point_id,
                        coords=(px, py),
                        intensity=int(heatmap_val),
                        rank=len(cluster.sample_points)
                    )
                    cluster.sample_points.append(pt)
                    global_point_id += 1
                    break

        # 🎨 可视化 Step 0: 概览图
        if current_output_dir:
            self._viz_step0_clusters_points(base_viz_bg, analysis_result, current_output_dir)

        # ==========================================
        # Step 2: 逐点推理与独立可视化 (Split Output)
        # ==========================================
        print(f"🔹 Step 2: Running SAM inference & Saving individual plots...")

        # 为 Step 1 的结果创建子文件夹，避免文件太乱
        step1_dir = os.path.join(current_output_dir, "step1_details")
        os.makedirs(step1_dir, exist_ok=True)

        for cluster in analysis_result.clusters:
            if not cluster.sample_points: continue

            for pt in cluster.sample_points:
                # 🔥 SAM 2 推理核心修改
                try:
                    # Wrapper 接收 [point] 列表
                    # 传入 streetview_bgr，Wrapper 内部处理 RGB 转换
                    best_mask, best_score = self.sam_tool.predict(streetview_bgr, [pt.coords])
                except Exception as e:
                    print(f"⚠️ Inference failed at point {pt.point_id}: {e}")
                    continue

                # 🔥 数据清洗 (关键步骤)
                # 1. 转为布尔型
                mask_bool = best_mask.astype(bool)
                # 2. 降维 (1, H, W) -> (H, W)
                if mask_bool.ndim > 2:
                    mask_bool = mask_bool.squeeze()

                # 🔥 [修改] 调用分类，获取中文标签
                chinese_label, clip_score = self._classify_mask_region(streetview_bgr, mask_bool)

                # 创建 Proposal
                proposal = ProposedObject(
                    proposal_id=f"C{cluster.cluster_id}_P{pt.point_id}",
                    score_a=best_score,
                    source_point_id=pt.point_id,
                    mask=mask_bool,
                    category_name=chinese_label  # 🔥 这里直接存中文标签
                )
                cluster.candidates.append(proposal)

                # 🎨 可视化 Step 1 (单张独立输出)
                if current_output_dir:
                    self._viz_step1_single_proposal(
                        base_viz_bg,
                        cluster,
                        pt,
                        proposal,
                        analysis_result.clusters,
                        step1_dir
                    )

        # ==============================================================================
        # 🔥 Step 3 & 4: 基于聚类共识的 Top-K 筛选 (Logic Update)
        # ==============================================================================
        print("🔹 Step 3 & 4: Clustering & Top-K Consensus Selection...")

        # 创建 Step 1.5 的输出文件夹
        step1_5_dir = os.path.join(current_output_dir, "step1_5_cluster_consensus")
        os.makedirs(step1_5_dir, exist_ok=True)

        # 参数设置
        TOP_K = 2  # 每个热力斑块最多保留几个物体
        SCORE_A_THRESH = 0.80  # 基础门槛

        for cluster in analysis_result.clusters:
            if not cluster.candidates:
                continue

            # 1. 筛选 Top-K 组
            top_groups = select_topk_objects(cluster.candidates, k=TOP_K, iou_threshold=0.85)

            # 🔥 2. 生成 Step 1.5: 单簇汇总图 (Cluster Consensus Summary)
            if current_output_dir:
                self._viz_step1_5_cluster_consensus(
                    base_viz_bg,
                    cluster,
                    top_groups,
                    step1_5_dir
                )

            # 3. 将筛选出的组转换为 ValidatedObject 存入结果
            rank_in_cluster = 1
            for group in top_groups:
                winner_proposal = group['representative']

                # 重新获取大类 (因为 Proposal 里只存了详细标签)
                main_cat = QINCHUAN_LABELS_MAP.get(winner_proposal.category_name, "Unknown")

                # 基础分门槛过滤
                if winner_proposal.score_a < SCORE_A_THRESH:
                    continue

                # Score B 计算
                mask_bool = winner_proposal.mask
                heatmap_norm = heatmap_gray_for_scoring.astype(float) / 255.0
                area_pixels = np.sum(mask_bool)
                score_b = 0.0
                if area_pixels > 0:
                    score_b = np.sum(heatmap_norm[mask_bool]) / area_pixels

                # 创建最终对象
                final_obj = ValidatedObject(
                    obj_id=len(analysis_result.final_objects) + 1,
                    original_cluster_id=cluster.cluster_id,
                    score_a=winner_proposal.score_a,
                    score_b=score_b,
                    mask=winner_proposal.mask,

                    # 🔥 [修改] 填入中文标签
                    category_name=winner_proposal.category_name,  # 直接使用 Proposal 里的中文

                    color=np.random.rand(3),
                    consensus_count=group['count'],
                    consensus_score=group['final_score']
                )
                analysis_result.final_objects.append(final_obj)
                rank_in_cluster += 1

        # 🎨 可视化 Step 2: 最终汇总 (修改后的高分辨率版本)
        if current_output_dir:
            self._viz_step2_final_summary(base_viz_bg, analysis_result, current_output_dir)

        gc.collect()
        return analysis_result, current_output_dir

    # ==================================================================================
    #  可视化辅助函数群 (内存优化版)
    # ==================================================================================

    def _resize_for_viz(self, img, target_width=600):
        """辅助函数：缩放图像以节省可视化内存"""
        h, w = img.shape[:2]
        scale = target_width / float(w)
        dim = (target_width, int(h * scale))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized, scale

    def _draw_mask_on_ax(self, ax, img_rgb, mask, color, label_text, title=None):
        """在给定的ax上绘制Mask"""
        ax.imshow(img_rgb)

        if mask is not None:
            # Mask 轮廓
            m_uint8 = (mask * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 使用极细的线绘制轮廓
            for cc in cnts:
                cc = cc.squeeze()
                if len(cc.shape) > 1:
                    cc = np.vstack([cc, cc[0]])
                    ax.plot(cc[:, 0], cc[:, 1], color=color, linewidth=0.8)  # 线宽改小

            # 半透明填充
            fill_mask = np.zeros((*img_rgb.shape[:2], 4))
            fill_mask[mask] = np.concatenate([color, [0.3]])  # 透明度0.3
            ax.imshow(fill_mask)

            # 标签 (极小字体)
            if label_text:
                M = cv2.moments(m_uint8)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    ax.text(cx, cy, label_text, color='white', fontsize=5, fontweight='normal',  # 字号改小
                            ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0.5))

        if title:
            ax.set_title(title, fontsize=6)  # 标题字号改小
        ax.axis('off')

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
            ax.text(c.centroid[0], c.centroid[1], f"C{c.cluster_id}", color='cyan', fontsize=3, fontweight='bold')

            # 采样点：模仿 heatmap_extractor 样式
            for p in c.sample_points:
                # 绘制点：白底红芯
                ax.scatter(p.coords[0], p.coords[1], c='white', s=1, marker='o', zorder=5)  # 白底
                ax.scatter(p.coords[0], p.coords[1], c='red', s=0.5, marker='o', zorder=6)  # 红芯

                # ID文字 (极小)
                ax.text(p.coords[0] + 6, p.coords[1], str(p.point_id), color='yellow', fontsize=2, zorder=10)
                total_points += 1

        ax.set_title(f"STEP 0: Clusters & Points (Total: {total_points})", fontsize=10)
        ax.axis('off')

        save_path = os.path.join(output_dir, "STEP0_Clusters_and_Points.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)  # 提高DPI保证小字清晰
        plt.close()
        print(f"✅ Saved: {save_path}")

    def _viz_step1_single_proposal(self, bg_img, target_cluster, target_point, proposal, all_clusters, output_dir):
        """Step 1: 单点单图 (最关键的修改)"""
        # 🔥 [内存优化] 降采样倍率 (例如缩小4倍，从 5000x3000 -> 1250x750)
        # 这对于 Debug 预览图完全足够，且能解决 MemoryError
        scale_factor = 0.8

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
            ls = '-' if is_target_cluster else ':'
            self._draw_contour(ax, c.contour_points, color, linewidth=1.5, linestyle=ls, alpha=alpha)

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
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)  # 🔥 关键：每一张画完立刻释放内存
        print(f"Saved: {filename}") # 太多了就不打印了，刷屏

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
        self._draw_contour(ax, cluster.contour_points, 'cyan', linewidth=1.5, linestyle='--')

        # 2. 绘制所有候选 Proposal 的轮廓 (Ghost Mode)
        # 用极淡的白色细线，展示"探索过程"
        for candidate in cluster.candidates:
            self._draw_mask_contour(ax, candidate.mask, 'white', linewidth=0.5, alpha=0.2)

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

                ax.text(cx, cy, label_txt, color='white', fontsize=12, fontweight='bold',  # 字号加大
                        ha='center', va='center',
                        bbox=dict(facecolor=color, alpha=0.8, edgecolor='white', boxstyle='round,pad=0.3'))

        ax.set_title(
            f"STEP 1.5: Cluster {cluster.cluster_id} Consensus Analysis (Candidates: {len(cluster.candidates)})",
            fontsize=16)
        ax.axis('off')

        save_path = os.path.join(output_dir, f"C{cluster.cluster_id}_Consensus.png")
        # 使用高 DPI 保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 🔥 修改后的 Step 2 可视化
    def _viz_step2_final_summary(self, bg_img, result, output_dir):
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
            ax.text(c.centroid[0], c.centroid[1], f"C{c.cluster_id}", color='cyan', fontsize=6, fontweight='bold',
                    alpha=0.8)

        # 2. 最终物体
        for obj in result.final_objects:
            # Mask 填充 (淡色)
            fill_mask = np.zeros((*bg_img.shape[:2], 4))
            fill_mask[obj.mask] = np.concatenate([obj.color, [0.2]])  # 透明度降低，不遮挡热力
            ax.imshow(fill_mask)

            # Mask 轮廓 (清晰)
            self._draw_mask_contour(ax, obj.mask, obj.color, linewidth=1.5)

            # 3. 详细标签 (Small Labels)
            M = cv2.moments(obj.mask.astype(np.uint8))
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 🔥 [修改] 标签显示逻辑
                label = (f"【{obj.category_name}】\n"  # 中文类别
                         f"ID:{obj.obj_id} ScB:{obj.score_b:.2f}")

                ax.text(cx, cy, label, color='white', fontsize=10, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor=obj.color, linewidth=0.5, pad=0.3))

        ax.set_title(f"STEP 2: Final Summary (N={len(result.final_objects)})", fontsize=12)
        ax.axis('off')

        save_path = os.path.join(output_dir, "STEP2_Final_Summary.png")
        # 300 DPI 保证放大后字迹清晰
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved High-Res Summary: {save_path}")

    # --- 绘图辅助 ---
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


if __name__ == "__main__":
    # 配置路径
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 改为自动获取当前目录作为根
    # 🔥 这里的路径请确保指向了正确的 SAM2 Checkpoint
    SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam2_repo", "checkpoints", "sam2.1_hiera_large.pt")

    TEST_IMG = os.path.join(PROJECT_ROOT, "data/input_streetview/QINCHUAN-62.jpg")
    TEST_HEATMAP = os.path.join(PROJECT_ROOT,
                                "data/experiment_data/gaze_heatmap/001/001_62_eyetrack_heatmap_20250929_190147.png")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output/test_output")
    # 🔥 配置 CLIP 模型路径
    CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "data/weights/clip")  # 确保这里是你手动下载的路径

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"❌ Weights not found: {SAM_CHECKPOINT}")
        sys.exit(1)

    # 初始化传入两个模型路径
    processor = SaliencySAMProcessor(
        sam_checkpoint=SAM_CHECKPOINT,
        clip_model_path=CLIP_MODEL_PATH
    )

    print(f"🚀 Processing {os.path.basename(TEST_IMG)}...")
    result, final_output_dir = processor.run_pipeline(TEST_IMG, TEST_HEATMAP, OUTPUT_ROOT)

    print(f"\n✨ Pipeline completed. All detailed visualizations saved to: {final_output_dir}")
    print("\n=== Final Analysis Summary ===")
    print(result.summary())