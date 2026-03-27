# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # 🔥 核心修复：强制使用非交互式后端
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import os

# ===========================
# 1. 配置与加载模型
# ===========================
# 请确保模型文件路径正确
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM model ({MODEL_TYPE})...")
# 检查模型文件是否存在
if not os.path.exists(SAM_CHECKPOINT):
    print(f"❌ Error: Model checkpoint '{SAM_CHECKPOINT}' not found.")
    exit(1)

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

# ===========================
# 2. 模拟/读取数据
# ===========================
# 请确保以下路径指向您的实际文件
image_path = "data/input_streetview/QINCHUAN-65.jpg"
heatmap_path = "data/experiment_data/gaze_heatmap/001/001_65_eyetrack_heatmap_20250929_190147.png"

if not os.path.exists(image_path) or not os.path.exists(heatmap_path):
    print("❌ Error: Image or heatmap file not found. Please check paths.")
    exit(1)

# 读取原始图像 (SAM 需要 RGB)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 读取热力图 (灰度，用于计算)
heatmap_gray = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

# 对齐尺寸 (非常重要，必须与原图一致)
if image.shape[:2] != heatmap_gray.shape[:2]:
    heatmap_gray = cv2.resize(heatmap_gray, (image.shape[1], image.shape[0]))

# 🔥 新增：生成彩色热力图用于可视化叠加
heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)


# ===========================
# 3. 核心算法：从热力图提取提示点 (Prompts)
# ===========================
def get_attention_points(heatmap, threshold=200, max_points=5):
    # 阈值过滤
    _, thresh_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    input_points = []
    input_labels = []

    # 按面积排序，只取前N个最大的关注点
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]

    for cnt in contours:
        # 计算重心 (Centroid)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            input_points.append([cx, cy])
            input_labels.append(1)  # 1 表示前景点

    return np.array(input_points), np.array(input_labels)


# 获取提示点
points, labels = get_attention_points(heatmap_gray, threshold=180)

if len(points) == 0:
    print("未检测到高关注区域，请降低阈值。")
else:
    print(f"提取到 {len(points)} 个视觉关注焦点。")

    # ===========================
    # 4. SAM 推理与增强可视化
    # ===========================
    predictor.set_image(image)

    # 设置画布大小
    plt.figure(figsize=(len(points) * 5, 5))

    for i, (point, label) in enumerate(zip(points, labels)):
        # SAM 推理
        input_point = np.array([point])
        input_label = np.array([label])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        # ===========================
        # 🔥 可视化增强部分 🔥
        # ===========================
        viz_img = image.copy()

        # 1. [需求实现] 叠加彩色热力图背景
        # 创建一个遮罩，只在有热力值的地方叠加
        heat_mask_bool = heatmap_gray > 20 # 过滤掉过暗的背景噪音
        # 使用 addWeighted 进行半透明叠加 (原图 60% + 热力图 40%)
        viz_img[heat_mask_bool] = cv2.addWeighted(
            viz_img[heat_mask_bool], 0.6,
            heatmap_color[heat_mask_bool], 0.4, 0
        )

        # 2. [需求实现] 让物体更加明显 (高亮填充)
        # 生成一个鲜艳的随机颜色
        obj_color = np.random.randint(50, 255, (3)).tolist()
        # 提高填充颜色的不透明度 (原图 30% + 填充色 70%)
        viz_img[best_mask] = (
            np.array(viz_img[best_mask]) * 0.3 +
            np.array(obj_color) * 0.7
        ).astype(np.uint8)

        # 3. [需求实现] 画一个清晰的轮廓
        # 将 bool mask 转为 uint8
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        # 查找 mask 的轮廓
        contours_sam, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓：白色 (255,255,255)，线宽 3
        cv2.drawContours(viz_img, contours_sam, -1, (255, 255, 255), 3)

        # 4. 画出注视点 (加一个白边让它在热力图上更突出)
        cv2.circle(viz_img, tuple(point), 12, (255, 255, 255), -1) # 白底
        cv2.circle(viz_img, tuple(point), 8, (255, 0, 0), -1)   # 红心

        # 显示结果
        plt.subplot(1, len(points), i + 1)
        plt.imshow(viz_img)
        plt.title(f"Attention Object {i + 1}\nScore: {scores[best_idx]:.2f}")
        plt.axis('off')

    plt.tight_layout()
    save_path = "demo_enhanced.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✨ 增强版可视化结果已保存至: {save_path}")