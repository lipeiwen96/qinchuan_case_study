# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')  # 强制非交互后端
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys

# ===========================
# 🔥 1. 导入封装好的 SAM2 模块
# ===========================
# 假设 sam2_adapter.py 在 src 文件夹下
try:
    from src.sam2_adapter import SAM2Wrapper
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保 'src' 文件夹下有 '__init__.py' 和 'sam2_adapter.py'")
    sys.exit(1)

# ===========================
# 2. 配置路径
# ===========================
# 自动获取项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 权重路径 (根据你的实际情况修改)
SAM_CHECKPOINT = os.path.join(ROOT_DIR, "sam2_repo", "checkpoints", "sam2.1_hiera_large.pt")

# 数据路径
image_path = os.path.join(ROOT_DIR, "data/input_streetview/QINCHUAN-62.jpg")
heatmap_path = os.path.join(ROOT_DIR,
                            "data/experiment_data/gaze_heatmap/001/001_62_eyetrack_heatmap_20250929_190147.png")

# ===========================
# 3. 初始化模型
# ===========================
try:
    # SAM2Wrapper 会自动找 Config，只需要传 Checkpoint
    sam_tool = SAM2Wrapper(checkpoint_path=SAM_CHECKPOINT)
except Exception as e:
    print(f"初始化失败: {e}")
    sys.exit(1)

# ===========================
# 4. 数据预处理
# ===========================
if not os.path.exists(image_path) or not os.path.exists(heatmap_path):
    print("❌ Error: Image or heatmap file not found.")
    sys.exit(1)

# 读取图像
image = cv2.imread(image_path)
# 注意：Wrapper 内部会自动转 RGB，所以这里传 BGR 给 wrapper 也没问题，
# 但为了可视化方便，我们这里还是转一下，因为 matplotlib 显示需要 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 读取热力图
heatmap_gray = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

# 对齐尺寸
if image.shape[:2] != heatmap_gray.shape[:2]:
    heatmap_gray = cv2.resize(heatmap_gray, (image.shape[1], image.shape[0]))

# 生成彩色热力图
heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)


# ===========================
# 5. 提取关注点
# ===========================
def get_attention_points(heatmap, threshold=200, max_points=5):
    _, thresh_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    input_points = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            input_points.append([cx, cy])

    return input_points


points = get_attention_points(heatmap_gray, threshold=180)

if not points:
    print("未检测到高关注区域。")
    sys.exit(0)

print(f"提取到 {len(points)} 个视觉关注焦点。")

# ===========================
# 6. 推理与可视化
# ===========================
plt.figure(figsize=(len(points) * 5, 5))

# 为了可视化，使用 RGB 图像副本
viz_base = image_rgb.copy()

for i, point in enumerate(points):
    # 🔥 调用模块进行推理
    # 注意：wrapper.predict 接收的是 list of points
    # 这里我们在循环里每次只传一个点
    try:
        best_mask, best_score = sam_tool.predict(image, [point])
    except Exception as e:
        print(f"推理点 {point} 时出错: {e}")
        continue

    # ===========================
    # 🔥 核心修复：处理 Mask 格式
    # ===========================
    # 1. 确保 mask 是布尔类型 (True/False)
    mask_bool = best_mask.astype(bool)

    # 2. 如果 mask 有多余的维度 (例如 1, H, W)，降维成 (H, W)
    if mask_bool.ndim > 2:
        mask_bool = mask_bool.squeeze()

    # 3. 创建可视化图像
    viz_img = viz_base.copy()

    # --- A. 叠加热力图 ---
    heat_mask_bool = heatmap_gray > 20
    # 注意：matplotlib 显示用 RGB，OpenCV 处理颜色转换要注意
    # heatmap_color 是 BGR，转 RGB 方便显示
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    viz_img[heat_mask_bool] = cv2.addWeighted(
        viz_img[heat_mask_bool], 0.6,
        heatmap_color_rgb[heat_mask_bool], 0.4, 0
    )

    # --- B. 高亮填充物体 ---
    # 生成随机颜色 (RGB)
    obj_color = np.random.randint(50, 255, (3)).tolist()

    # 使用 bool 索引进行赋值
    # 🔥 这里是之前报错的地方，现在应该修好了
    viz_img[mask_bool] = (
            viz_img[mask_bool] * 0.3 +
            np.array(obj_color) * 0.7
    ).astype(np.uint8)

    # --- C. 绘制轮廓 ---
    mask_uint8 = (mask_bool * 255).astype(np.uint8)
    contours_sam, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(viz_img, contours_sam, -1, (255, 255, 255), 3)

    # --- D. 绘制注视点 ---
    cv2.circle(viz_img, tuple(point), 12, (255, 255, 255), -1)
    cv2.circle(viz_img, tuple(point), 8, (255, 0, 0), -1)

    # 显示结果
    plt.subplot(1, len(points), i + 1)
    plt.imshow(viz_img)
    plt.title(f"Object {i + 1}\nScore: {best_score:.2f}")
    plt.axis('off')

plt.tight_layout()
save_path = "demo_enhanced_sam2_result.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✨ 增强版可视化结果已保存至: {save_path}")