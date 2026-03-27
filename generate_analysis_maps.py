# generate_analysis_maps.py
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm
import datetime

# ==========================================
# 配置区域
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 输入文件路径
EXCEL_PATH = os.path.join(PROJECT_ROOT, "data/results/Global_Analysis_Result_Batch.xlsx")
IMG_DIR = os.path.join(PROJECT_ROOT, "data/input_streetview")

# 输出文件路径 (增加时间戳到秒)
CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "analysis", CURRENT_TIMESTAMP)

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 情感评分设置
LIKERT_NEUTRAL = 3.5
LIKERT_MIN = 1.0
LIKERT_MAX = 8.0

# ----------------------------
# 🎨 可视化参数微调区
# ----------------------------

# 1. 背景白化程度 (0.0=原图, 1.0=纯白)
BG_WHITEN_FACTOR = 0.6

# 2. 图 A (热力图) 设置
HEATMAP_ALPHA_MAX = 0.98
# 热力图轮廓线阈值 (多少人看过才画轮廓)
HEATMAP_CONTOUR_THRESH = 3.0

# 3. 图 B (情感图) SCI 配色设置
COLOR_NEGATIVE = "#313695"  # 深蓝
COLOR_NEUTRAL = "#FFFFBF"  # 浅黄
COLOR_POSITIVE = "#A50026"  # 深红
SENTIMENT_ALPHA_MAX = 0.98

# 4. 标签与轮廓设置
# 是否显示轮廓中心的分值标签开关
SHOW_CONTOUR_LABELS = True
# [新增] 最小标注面积阈值 (像素)
# 只有当轮廓面积 > 此值时，才会在中心写数字。防止细碎斑点上有文字堆叠。
MIN_AREA_FOR_LABELS = 400
# 标签字体大小
LABEL_FONT_SIZE = 6
# [图B专用] 情感显著性阈值
SENTIMENT_SIGNIFICANCE_THRESHOLD = 1.0


# ==========================================
# 辅助函数
# ==========================================

def wkt_to_cv2_contours(wkt_str, img_h, img_w):
    """将 WKT 字符串转换为 OpenCV 可用的轮廓点集"""
    try:
        geom = wkt.loads(wkt_str)
    except Exception:
        return []

    contours = []

    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = geom.geoms
    else:
        return []

    for poly in polys:
        if poly.exterior:
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            contours.append(coords)

    return contours


def create_sci_sentiment_cmap():
    """创建符合 SCI 审美的情感颜色映射"""
    norm_neutral = (LIKERT_NEUTRAL - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)
    colors = [
        (0.0, COLOR_NEGATIVE),
        (norm_neutral, COLOR_NEUTRAL),
        (1.0, COLOR_POSITIVE)
    ]
    return mcolors.LinearSegmentedColormap.from_list("sci_sentiment_cmap", colors)


def create_whitened_background(img_rgb):
    """创建一个变白且变淡的背景图"""
    h, w, c = img_rgb.shape
    white_layer = np.full((h, w, c), 255, dtype=np.uint8)
    whitened = cv2.addWeighted(img_rgb, (1 - BG_WHITEN_FACTOR), white_layer, BG_WHITEN_FACTOR, 0)
    return whitened


def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None


def draw_center_label(ax, x, y, text, color='white'):
    ax.text(x, y, text, color=color,
            fontsize=LABEL_FONT_SIZE, ha='center', va='center', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])


# ==========================================
# 绘图核心逻辑
# ==========================================

def process_single_image(img_name, group_df, output_dir):
    """处理单张街景图的所有数据"""

    # 1. 读取底图
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        base, _ = os.path.splitext(img_name)
        for ext in ['.jpg', '.JPG', '.png', '.PNG']:
            temp_path = os.path.join(IMG_DIR, base + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        else:
            print(f"⚠️ Image not found: {img_name}, skipping.")
            return

    bg_img_bgr = cv2.imread(img_path)
    if bg_img_bgr is None:
        return
    bg_img_rgb = cv2.cvtColor(bg_img_bgr, cv2.COLOR_BGR2RGB)
    h, w = bg_img_rgb.shape[:2]

    # 预处理背景图
    bg_whitened = create_whitened_background(bg_img_rgb)

    # 初始化累加器
    heatmap_accumulator = np.zeros((h, w), dtype=np.float32)
    sentiment_sum_grid = np.zeros((h, w), dtype=np.float32)
    sentiment_count_grid = np.zeros((h, w), dtype=np.float32)

    # 2. 遍历数据
    valid_objects_count = 0

    for _, row in group_df.iterrows():
        likert = row['Likert_Scale']
        geo_json_str = row['Geometry_JSON']

        try:
            likert = float(likert)
        except:
            continue

        try:
            geo_list = json.loads(geo_json_str)
        except:
            continue

        if not geo_list:
            continue

        for obj in geo_list:
            wkt_str = obj.get('wkt_geometry', '')
            if wkt_str == 'EMPTY' or not wkt_str:
                continue

            contours = wkt_to_cv2_contours(wkt_str, h, w)
            if not contours:
                continue

            # 绘制 Mask
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(obj_mask, contours, 1)

            # 更新累加器
            heatmap_accumulator += obj_mask

            mask_bool = obj_mask.astype(bool)
            sentiment_sum_grid[mask_bool] += likert
            sentiment_count_grid[mask_bool] += 1

            valid_objects_count += 1

    if valid_objects_count == 0:
        print(f"⚠️ No valid geometry for {img_name}")
        return

    # ==========================================
    # 绘制 Map A: 关注度热力图 (Attention Heatmap)
    # ==========================================
    fig_a, ax_a = plt.subplots(figsize=(12, 8))

    # 底图
    ax_a.imshow(bg_whitened)

    # 热力层
    max_val = np.max(heatmap_accumulator)
    if max_val > 0:
        heatmap_norm = heatmap_accumulator / max_val

        # 颜色: jet
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap_norm)

        # 透明度控制
        alphas = np.power(heatmap_norm, 0.5) * HEATMAP_ALPHA_MAX
        heatmap_colored[..., 3] = alphas

        ax_a.imshow(heatmap_colored)

        # 轮廓线与标签
        if max_val >= HEATMAP_CONTOUR_THRESH:
            ret, thresh_mask = cv2.threshold(heatmap_accumulator, HEATMAP_CONTOUR_THRESH, 255, cv2.THRESH_BINARY)
            thresh_mask = thresh_mask.astype(np.uint8)
            contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if len(cnt) < 3: continue
                coords = cnt.squeeze()
                if len(coords.shape) == 1: coords = coords[np.newaxis, :]
                coords = np.vstack([coords, coords[0]])
                # 亮白色加粗虚线
                ax_a.plot(coords[:, 0], coords[:, 1], color='white', linewidth=1.2, linestyle='--', alpha=0.9)

                # [新增] 面积过滤与标签绘制
                area = cv2.contourArea(cnt)
                if SHOW_CONTOUR_LABELS and area >= MIN_AREA_FOR_LABELS:
                    center = get_contour_center(cnt)
                    if center:
                        cx, cy = center
                        # 计算最大值
                        single_cnt_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(single_cnt_mask, [cnt], -1, 255, -1)
                        try:
                            min_val, max_val_zone, min_loc, max_loc = cv2.minMaxLoc(heatmap_accumulator,
                                                                                    mask=single_cnt_mask)
                            label_text = f"{int(max_val_zone)}"
                            draw_center_label(ax_a, cx, cy, label_text)
                        except:
                            pass

    ax_a.set_title(f"Map A: Gaze Attention Heatmap - {img_name}\n(N={len(group_df)} Volunteers)", fontsize=14)
    ax_a.axis('off')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, max_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_a, fraction=0.03, pad=0.04)
    cbar.set_label('Gaze Frequency (Overlay Count)')

    save_path_a = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_MapA_Attention.png")
    plt.savefig(save_path_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)

    # ==========================================
    # 绘制 Map B: 情感倾向图 (Sentiment Map)
    # ==========================================
    fig_b, ax_b = plt.subplots(figsize=(12, 8))

    # 底图
    ax_b.imshow(bg_whitened)

    # 计算平均分
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_sentiment_grid = sentiment_sum_grid / sentiment_count_grid

    avg_sentiment_grid[sentiment_count_grid == 0] = LIKERT_NEUTRAL

    # 1. 颜色映射 (SCI Style)
    custom_cmap = create_sci_sentiment_cmap()
    norm = plt.Normalize(vmin=LIKERT_MIN, vmax=LIKERT_MAX)
    sentiment_colored = custom_cmap(norm(avg_sentiment_grid))

    # 2. 透明度映射
    diff = np.abs(avg_sentiment_grid - LIKERT_NEUTRAL)
    alpha_curve = np.power(diff / 4.5, 0.7)
    alpha_channel = 0.2 + alpha_curve * (SENTIMENT_ALPHA_MAX - 0.2)
    alpha_channel = np.clip(alpha_channel, 0, SENTIMENT_ALPHA_MAX)
    alpha_channel[sentiment_count_grid == 0] = 0.0
    sentiment_colored[..., 3] = alpha_channel

    ax_b.imshow(sentiment_colored)

    # 轮廓与标签
    if SHOW_CONTOUR_LABELS:
        # 显著性掩膜
        significance_mask = (diff >= SENTIMENT_SIGNIFICANCE_THRESHOLD) & (sentiment_count_grid > 0)
        significance_mask = significance_mask.astype(np.uint8) * 255

        contours_b, _ = cv2.findContours(significance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_b:
            area = cv2.contourArea(cnt)
            if area < 50: continue  # 过滤极小噪点 (绘图过滤)

            # 绘制轮廓线
            coords = cnt.squeeze()
            if len(coords.shape) == 1: coords = coords[np.newaxis, :]
            coords = np.vstack([coords, coords[0]])
            ax_b.plot(coords[:, 0], coords[:, 1], color='gray', linewidth=0.8, linestyle='-', alpha=0.7)

            # [新增] 面积过滤与标签绘制
            if area >= MIN_AREA_FOR_LABELS:
                center = get_contour_center(cnt)
                if center:
                    cx, cy = center
                    single_cnt_mask_b = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(single_cnt_mask_b, [cnt], -1, 255, -1)
                    mean_val = cv2.mean(avg_sentiment_grid, mask=single_cnt_mask_b)[0]

                    label_text = f"{mean_val:.1f}"
                    draw_center_label(ax_b, cx, cy, label_text)

    ax_b.set_title(f"Map B: Semantic Sentiment Map - {img_name}\n(Blue=Negative, Red=Positive)",
                   fontsize=14)
    ax_b.axis('off')

    # 图例
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE, edgecolor='k', label='Negative (1.0)'),
        Patch(facecolor=COLOR_NEUTRAL, edgecolor='k', label='Neutral (3.5)'),
        Patch(facecolor=COLOR_POSITIVE, edgecolor='k', label='Positive (8.0)'),
    ]
    ax_b.legend(handles=legend_elements, loc='upper right')

    # Colorbar
    sm_b = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm_b.set_array([])
    cbar_b = plt.colorbar(sm_b, ax=ax_b, fraction=0.03, pad=0.04)
    cbar_b.set_label('Avg Likert Scale')

    save_path_b = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_MapB_Sentiment.png")
    plt.savefig(save_path_b, dpi=300, bbox_inches='tight')
    plt.close(fig_b)


# ==========================================
# 主程序
# ==========================================

def main():
    print("🚀 Starting Visualization Analysis...")
    print(f"📂 Output Directory: {OUTPUT_ROOT}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if not os.path.exists(EXCEL_PATH):
        print(f"❌ Excel file not found: {EXCEL_PATH}")
        return

    # 1. 读取 Excel
    print("   Reading Excel data...")
    df = pd.read_excel(EXCEL_PATH)

    # 2. 按街景图分组
    if 'Image_Name' not in df.columns:
        print("❌ Column 'Image_Name' not found in Excel!")
        return

    grouped = df.groupby('Image_Name')
    print(f"   Found {len(grouped)} unique street view images.")

    # 3. 循环处理
    for img_name, group_df in tqdm(grouped, desc="Generating Maps"):
        try:
            process_single_image(img_name, group_df, OUTPUT_ROOT)
        except Exception as e:
            print(f"   ❌ Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()

    print("🎉 All tasks completed!")


if __name__ == "__main__":
    main()