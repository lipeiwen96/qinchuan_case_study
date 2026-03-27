# stage1_sam_segmentation.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import datetime
import os
import sys
import pickle  # 🔥 引入 pickle 用于序列化
import gc

# 引入 SAM2
from src.sam2_adapter import SAM2Wrapper
from src.new_data_structures import HeatmapCluster, SamplingPoint, ProposedObject, ValidatedObject, \
    StreetViewAnalysisResult
from src.heatmap_extractor import HeatmapRedZoneExtractor


class Stage1SAMProcessor:
    def __init__(self, sam_checkpoint, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 [Stage 1] Initializing SAM 2 on {self.device}...")
        try:
            self.sam_tool = SAM2Wrapper(checkpoint_path=sam_checkpoint, device=self.device)
            print("✅ SAM 2 Model Loaded.")
        except Exception as e:
            print(f"❌ SAM 2 Init Failed: {e}")
            sys.exit(1)

    def run_segmentation(self, image_path, heatmap_path, cache_dir):
        """
        运行第一阶段：热力提取 + SAM2 分割
        结果将保存为 .pkl 文件
        """
        image_basename = os.path.basename(image_path)
        print(f"🔹 Processing {image_basename}...")

        # 1. 读取图像
        streetview_bgr = cv2.imread(image_path)
        if streetview_bgr is None:
            print(f"❌ Error: Cannot read image at {image_path}")
            return None

        # 2. 初始化结果容器
        analysis_result = StreetViewAnalysisResult(image_name=image_basename)

        # 3. 提取热力红区与采样点
        print("   - Extracting Red Zones...")
        extractor = HeatmapRedZoneExtractor(image_path, heatmap_path)
        contours, all_sample_points = extractor.extract_red_zones(min_area=50, grid_spacing=55)  # 采样间隔

        # 构建 Clusters
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cluster = HeatmapCluster(cluster_id=idx, area=int(area), centroid=(cx, cy), contour_points=cnt)
            analysis_result.clusters.append(cluster)

        # 分配采样点
        # 为了后续 CLIP 分类，我们需要保存热力图数据，用于计算 score_b
        # 这里我们将 heatmap 灰度图也存入 result 对象（虽然数据结构里没定义，但 python 动态特性允许临时挂载）
        # 或者在 Stage 2 重新生成 heatmap，这里选择后者以减小 pickle 体积。

        _, aligned_heatmap_bgr = extractor.overlay_img, extractor.aligned_heatmap
        heatmap_gray = cv2.cvtColor(aligned_heatmap_bgr, cv2.COLOR_BGR2GRAY)

        global_point_id = 0
        for px, py in all_sample_points:
            heatmap_val = heatmap_gray[py, px]
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

        # 4. 运行 SAM2 分割
        print("   - Running SAM2 Inference...")
        for cluster in analysis_result.clusters:
            if not cluster.sample_points: continue

            for pt in cluster.sample_points:
                try:
                    best_mask, best_score = self.sam_tool.predict(streetview_bgr, [pt.coords])
                except Exception as e:
                    print(f"⚠️ Inference failed at point {pt.point_id}: {e}")
                    continue

                # 数据清洗
                mask_bool = best_mask.astype(bool)
                if mask_bool.ndim > 2:
                    mask_bool = mask_bool.squeeze()

                # 先构造对象，不传 mask
                proposal = ProposedObject(
                    proposal_id=f"C{cluster.cluster_id}_P{pt.point_id}",
                    score_a=best_score,
                    source_point_id=pt.point_id,
                    category_name="Pending"
                )
                # 再赋值，触发自动压缩
                proposal.mask = mask_bool
                cluster.candidates.append(proposal)

        # 5. 保存缓存文件 (Pickle)
        # 我们只保存 analysis_result 对象，不保存图像本身以节省空间，图像路径在 analysis_result.image_name 里有记录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_filename = f"{os.path.splitext(image_basename)[0]}_stage1.pkl"
        save_path = os.path.join(cache_dir, pkl_filename)

        with open(save_path, 'wb') as f:
            pickle.dump(analysis_result, f)

        print(f"✅ Stage 1 Cache Saved: {save_path}")

        # 显式清理
        del streetview_bgr
        del extractor
        gc.collect()

        return save_path


if __name__ == "__main__":
    # 配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam2_repo", "checkpoints", "sam2.1_hiera_large.pt")

    # 示例输入
    TEST_IMG = os.path.join(PROJECT_ROOT, "data/input_streetview/QINCHUAN-62.JPG")
    TEST_HEATMAP = os.path.join(PROJECT_ROOT,
                                "data/experiment_data/gaze_heatmap/001/001_62_eyetrack_heatmap_20250929_190147.png")

    # 缓存目录
    CACHE_DIR = os.path.join(PROJECT_ROOT, "output/intermediate_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"❌ Checkpoint not found: {SAM_CHECKPOINT}")
        sys.exit(1)

    processor = Stage1SAMProcessor(SAM_CHECKPOINT)
    processor.run_segmentation(TEST_IMG, TEST_HEATMAP, CACHE_DIR)