# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import torch
import warnings

warnings.filterwarnings('ignore')

# 引入你现有的 SAM2 模型
from src.sam2_adapter import SAM2Wrapper

# ==========================================
# 配置与全局变量
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam2_repo", "checkpoints", "sam2.1_hiera_large.pt")

# 全局状态，用于存储当前图片的所有标注记录
# 结构: [{"mask": np.array, "score": int, "simulated_points": [(x,y)...], "center": (x,y)}]
ANNOTATIONS = []
CURRENT_IMG_BGR = None
CURRENT_IMG_RGB = None

# 初始化 SAM2 (全局单例)
print("🚀 正在加载 SAM2 模型，请稍候...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_tool = SAM2Wrapper(checkpoint_path=SAM_CHECKPOINT, device=device)
print("✅ SAM2 加载完成！")


# ==========================================
# 核心逻辑与算法
# ==========================================

def create_sci_sentiment_cmap():
    """1~9 分的红黄蓝配色 (复用给 MapB 和 前端预览图)"""
    colors = [(0.0, "#313695"), (0.5, "#FFFFBF"), (1.0, "#A50026")]
    return mcolors.LinearSegmentedColormap.from_list("sci_sentiment", colors)


def get_color_for_score(score):
    """根据分数返回对应的 RGB 颜色元组，用于前端预览"""
    custom_cmap = create_sci_sentiment_cmap()
    norm_score = plt.Normalize(vmin=1.0, vmax=9.0)
    rgba = custom_cmap(norm_score(score))
    # 转换为 OpenCV 使用的 0-255 RGB格式
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]


def simulate_user_gaze(mask, center_x, center_y, num_users=40, spread=80):
    """基于中心点和 Mask，模拟生成 40 个用户的视线落点（带毛刺扩散）"""
    h, w = mask.shape
    points = []
    xs = np.random.normal(center_x, spread, num_users * 3)
    ys = np.random.normal(center_y, spread, num_users * 3)

    for x, y in zip(xs, ys):
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            if mask[iy, ix] or np.random.rand() > 0.8:
                points.append((ix, iy))
        if len(points) >= num_users:
            break
    return points


def redraw_preview():
    """根据全局 ANNOTATIONS 重新绘制带颜色的预览图"""
    global CURRENT_IMG_RGB, ANNOTATIONS
    if CURRENT_IMG_RGB is None:
        return None

    preview_img = CURRENT_IMG_RGB.copy()
    overlay = np.zeros_like(preview_img, dtype=np.uint8)
    alpha_mask = np.zeros(preview_img.shape[:2], dtype=bool)

    # 绘制各个色块
    for ann in ANNOTATIONS:
        m = ann["mask"]
        color = get_color_for_score(ann["score"])
        overlay[m] = color
        alpha_mask = alpha_mask | m

    # 半透明叠合
    preview_img[alpha_mask] = cv2.addWeighted(
        preview_img[alpha_mask], 0.4, overlay[alpha_mask], 0.6, 0
    )

    # 绘制点击中心点和边框
    for ann in ANNOTATIONS:
        cx, cy = ann["center"]
        color = get_color_for_score(ann["score"])
        cv2.circle(preview_img, (cx, cy), 6, (255, 255, 255), -1)  # 白底边框
        cv2.circle(preview_img, (cx, cy), 4, color, -1)  # 内部颜色

    return preview_img


def on_image_upload(image):
    """处理用户上传/改变原图的事件"""
    global CURRENT_IMG_BGR, CURRENT_IMG_RGB, ANNOTATIONS
    if image is not None:
        CURRENT_IMG_RGB = image.copy()
        CURRENT_IMG_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ANNOTATIONS.clear()
        return image, "✅ 新底图已就绪，请点击标注。"
    return None, "⚠️ 图片已清空。"


def on_image_click(image, evt: gr.SelectData, score_val):
    """当用户点击图片时触发：运行 SAM2 -> 生成模拟数据 -> 保存状态 -> 返回预览图"""
    global ANNOTATIONS, CURRENT_IMG_BGR, CURRENT_IMG_RGB

    if CURRENT_IMG_RGB is None:
        return image, "请先上传或选择一张街景图"

    click_x, click_y = evt.index
    print(f"📍 点击坐标: ({click_x}, {click_y}), 分数: {score_val}")

    # 1. 运行 SAM2 获取 Mask
    try:
        mask, sam_score = sam_tool.predict(CURRENT_IMG_BGR, [[click_x, click_y]])
        mask_bool = mask.squeeze().astype(bool)
    except Exception as e:
        return image, f"SAM2 分割失败: {e}"

    # 2. 模拟 40 人视线点
    sim_points = simulate_user_gaze(mask_bool, click_x, click_y, num_users=40, spread=60)

    # 3. 记录到全局状态
    ANNOTATIONS.append({
        "mask": mask_bool,
        "score": score_val,
        "simulated_points": sim_points,
        "center": (click_x, click_y)
    })

    # 4. 重新绘制即时预览图
    preview_img = redraw_preview()

    return preview_img, f"✅ 成功添加区域！当前共标注了 {len(ANNOTATIONS)} 个片区。"


def undo_last_annotation():
    """撤回最新的一次标注"""
    global ANNOTATIONS, CURRENT_IMG_RGB
    if not ANNOTATIONS:
        # 如果已经空了，直接返回原图
        return CURRENT_IMG_RGB, "⚠️ 已经没有可以撤回的标注了。"

    ANNOTATIONS.pop()  # 移除最后一个

    if not ANNOTATIONS:
        return CURRENT_IMG_RGB, "↩️ 已撤回，当前图片无标注。"

    preview_img = redraw_preview()
    return preview_img, f"↩️ 已撤回上一步！当前剩余 {len(ANNOTATIONS)} 个片区。"


def clear_annotations():
    """清空当前图片的所有标注，恢复原图"""
    global ANNOTATIONS, CURRENT_IMG_RGB
    ANNOTATIONS.clear()
    # 注意：这里必须返回干干净净的 CURRENT_IMG_RGB，而不是传进来的已经被画花的 image
    return CURRENT_IMG_RGB, "🗑️ 所有标注已清空，请重新点击。"


# ==========================================
# 绘图逻辑
# ==========================================
def generate_final_maps():
    """
    基于全局标注数据，生成 Map A (热力) 和 Map B (情感)
    """
    global ANNOTATIONS, CURRENT_IMG_RGB
    if not ANNOTATIONS or CURRENT_IMG_RGB is None:
        return None, None, "⚠️ 没有标注数据，请先在图上点击标注。"

    h, w, _ = CURRENT_IMG_RGB.shape
    bg_whitened = cv2.addWeighted(CURRENT_IMG_RGB, 0.4, np.full_like(CURRENT_IMG_RGB, 255), 0.6, 0)

    # --- 1. 绘制 Map A: 模拟注意力热力图 ---
    heatmap_acc = np.zeros((h, w), dtype=np.float32)
    for ann in ANNOTATIONS:
        for px, py in ann["simulated_points"]:
            y_grid, x_grid = np.ogrid[-20:21, -20:21]
            kernel = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * 10 ** 2))

            y1, y2 = max(0, py - 20), min(h, py + 21)
            x1, x2 = max(0, px - 20), min(w, px + 21)
            ky1, ky2 = 20 - (py - y1), 20 + (y2 - py)
            kx1, kx2 = 20 - (px - x1), 20 + (x2 - px)

            heatmap_acc[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    fig_a, ax_a = plt.subplots(figsize=(10, 6))
    ax_a.imshow(bg_whitened)
    if np.max(heatmap_acc) > 0:
        norm_heat = heatmap_acc / np.max(heatmap_acc)
        colored_heat = plt.cm.jet(norm_heat)
        colored_heat[..., 3] = np.power(norm_heat, 0.5) * 0.9
        ax_a.imshow(colored_heat)
    ax_a.set_title("Map A: Simulated Expert Attention Heatmap (N=40)", fontsize=12)
    ax_a.axis('off')

    # 【修复Bug】：使用更现代安全的 buffer_rgba 提取图像矩阵
    fig_a.canvas.draw()
    img_a = np.asarray(fig_a.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig_a)

    # --- 2. 绘制 Map B: 专家情感分布图 ---
    sentiment_grid = np.zeros((h, w), dtype=np.float32)
    mask_acc = np.zeros((h, w), dtype=bool)

    for ann in ANNOTATIONS:
        m = ann["mask"]
        score = ann["score"]
        sentiment_grid[m] = score
        mask_acc = mask_acc | m

    fig_b, ax_b = plt.subplots(figsize=(10, 6))
    ax_b.imshow(bg_whitened)

    custom_cmap = create_sci_sentiment_cmap()
    norm_score = plt.Normalize(vmin=1.0, vmax=9.0)
    sentiment_colored = custom_cmap(norm_score(sentiment_grid))
    sentiment_colored[~mask_acc, 3] = 0.0
    sentiment_colored[mask_acc, 3] = 0.8

    ax_b.imshow(sentiment_colored)

    for ann in ANNOTATIONS:
        m = ann["mask"].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 100:
                coords = cnt.squeeze()
                if len(coords.shape) == 1: coords = coords[np.newaxis, :]
                coords = np.vstack([coords, coords[0]])
                ax_b.plot(coords[:, 0], coords[:, 1], color='gray', linewidth=1)

                cx, cy = ann["center"]
                ax_b.text(cx, cy, str(ann["score"]), color='white', fontsize=10,
                          fontweight='bold', ha='center', va='center',
                          path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax_b.set_title("Map B: Expert Guided Semantic Sentiment Map", fontsize=12)
    ax_b.axis('off')

    # 【修复Bug】：使用更现代安全的 buffer_rgba 提取图像矩阵
    fig_b.canvas.draw()
    img_b = np.asarray(fig_b.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig_b)

    return img_a, img_b, "🎉 图表生成完成！右键可保存图片。"


# ==========================================
# Gradio 界面搭建
# ==========================================
with gr.Blocks(title="Qinchuan 专家交互标注系统") as demo:
    gr.Markdown("## 🏙️ Qinchuan 专家交互热力与情感标注系统")
    gr.Markdown(
        "1. 上传或选择街景底图。 2. 调节上方的情感滑块(1-9分)。 3. **在图片上点击你想标注的物体中心**。 4. 点击【生成分析图】即可得到 40 人叠加视效。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            score_slider = gr.Slider(minimum=1, maximum=9, value=5, step=1, label="当前选择的分数 (1=消极, 9=积极)")
            input_image = gr.Image(type="numpy", label="交互底图 (直接在此处点击)")

            with gr.Row():
                undo_btn = gr.Button("↩️ 撤回上一步")
                clear_btn = gr.Button("🗑️ 清空所有标注")

            with gr.Row():
                generate_btn = gr.Button("🚀 生成分析叠合图", variant="primary")

            status_text = gr.Textbox(label="状态栏", interactive=False)

        with gr.Column(scale=1):
            output_map_a = gr.Image(type="numpy", label="Map A: 模拟 40 人注意力热力图")
            output_map_b = gr.Image(type="numpy", label="Map B: 情感分布区")

    # === 事件绑定 ===

    # 1. 监听图片的上传与改变 (确保底层原图不被弄脏)
    input_image.upload(
        fn=on_image_upload,
        inputs=[input_image],
        outputs=[input_image, status_text]
    )

    # 2. 当用户点击图片时：
    input_image.select(
        fn=on_image_click,
        inputs=[input_image, score_slider],
        outputs=[input_image, status_text]
    )

    # 3. 撤回按钮
    undo_btn.click(
        fn=undo_last_annotation,
        inputs=[],
        outputs=[input_image, status_text]
    )

    # 4. 清空按钮 (不再传入污染了的 input_image，直接由后端用干净的原图覆盖)
    clear_btn.click(
        fn=clear_annotations,
        inputs=[],
        outputs=[input_image, status_text]
    )

    # 5. 生成终图按钮
    generate_btn.click(
        fn=generate_final_maps,
        inputs=[],
        outputs=[output_map_a, output_map_b, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)