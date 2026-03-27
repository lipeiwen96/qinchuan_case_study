# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # 🔥 核心修复：强制使用非交互式后端，必须写在 import pyplot 之前
import matplotlib.pyplot as plt

import os
import sys
import glob
import re
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from typing import Dict, List

# ✅ TF 环境配置
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
from scr.data_structures import Volunteer, VolunteerInfo, SingleTrialData
from scr.semantic_engine import SemanticModelEngine
from scr.image_processing import overlay_heatmap_on_streetview, align_heatmap_to_streetview

# ===========================
# 🎨 1. 强制颜色定义 (COCO_META)
# ===========================
from scr.COCO_META import COCO_META

# ⚡️ 预处理颜色字典 (全局调用)
# 确保所有图表、所有步骤使用完全一致的颜色
NAME_TO_COLOR_BGR = {}  # OpenCV 使用 (0-255)
NAME_TO_COLOR_RGB = {}  # Matplotlib 使用 (0-1)

# 定义一个默认颜色 (灰色)，防止查不到字典时报错
DEFAULT_COLOR_BGR = (128, 128, 128)
DEFAULT_COLOR_RGB = (0.5, 0.5, 0.5)

for item in COCO_META:
    name = item['name']
    rgb = item['color']  # List [R, G, B] like [220, 20, 60]

    # BGR for OpenCV (Opencv reads as Blue, Green, Red)
    NAME_TO_COLOR_BGR[name] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    # RGB Normalized for Matplotlib
    NAME_TO_COLOR_RGB[name] = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

# ===========================
# ⚙️ 全局配置
# ===========================
# DIFFERENTIAL: Low(20-80), Medium(80-180), High(180-255)
# CUMULATIVE:   Low(20-255), Medium(80-255), High(180-255)
CALCULATION_MODE = 'CUMULATIVE'

DATA_ROOT = 'data'
INPUT_STREETVIEW_DIR = os.path.join(DATA_ROOT, 'input_streetview')
EXP_DATA_DIR = os.path.join(DATA_ROOT, 'experiment_data')
VOL_INFO_PATH = os.path.join(EXP_DATA_DIR, '志愿者信息.xlsx')
CSV_DIR = os.path.join(EXP_DATA_DIR, 'csv')
GAZE_DIR = os.path.join(EXP_DATA_DIR, 'gaze_heatmap')

OUTPUT_ROOT = 'output/version2_20251221'
CACHE_DIR = os.path.join("output", 'segmentation_cache')
INDIVIDUAL_DATA_DIR = os.path.join(OUTPUT_ROOT, 'individual_data_json')
INDIVIDUAL_VIZ_DIR = os.path.join(OUTPUT_ROOT, 'individual_charts')
CHECK_IMG_ROOT = os.path.join(OUTPUT_ROOT, 'visual_checks')
AGGREGATED_VIZ_DIR = os.path.join(OUTPUT_ROOT, 'aggregated_analysis')

MODEL_NAME = 'resnet50_kmax_deeplab_coco_train'


# ===========================
# 🎨 颜色获取函数 (已修正)
# ===========================
def get_class_color(class_name):
    """
    根据类名返回固定的 BGR 颜色。
    严格对应 COCO_META，不再使用 Hash 随机生成。
    """
    return NAME_TO_COLOR_BGR.get(class_name, DEFAULT_COLOR_BGR)


def log(msg):
    print(f"🔹 {msg}")


# ===========================
# 🧮 核心计算函数
# ===========================
def analyze_heatmap_intersection(heatmap_path: str, seg_mask: np.ndarray, id_map: dict) -> Dict[str, Dict[str, float]]:
    heatmap_src = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    if heatmap_src is None: return {}

    h_base, w_base = seg_mask.shape

    # === 新的对齐逻辑 (Height-based scaling + Center Cropping) ===
    # 调用对齐函数
    final_heatmap = align_heatmap_to_streetview(heatmap_src, h_base, w_base)

    # 2. 定义阈值
    if CALCULATION_MODE == 'DIFFERENTIAL':
        thresholds = {
            'High Attention': (180, 255),
            'Medium Attention': (80, 180),
            'Low Attention': (20, 80)
        }
    else:
        # CUMULATIVE
        thresholds = {
            'High Attention': (180, 255),
            'Medium Attention': (80, 255),
            'Low Attention': (20, 255)
        }

    intersection_results = {}

    for level_name, (low_thr, high_thr) in thresholds.items():
        heat_mask = cv2.inRange(final_heatmap, low_thr, high_thr)
        total_heat_pixels = np.sum(heat_mask > 0)

        level_stats = {}
        if total_heat_pixels > 0:
            present_ids = np.unique(seg_mask)
            for cid in present_ids:
                if cid == 0 or cid not in id_map: continue

                # 语义掩码
                semantic_mask = (seg_mask == cid).astype(np.uint8) * 255
                # 交集
                inter = cv2.bitwise_and(heat_mask, semantic_mask)
                inter_px = np.sum(inter > 0)

                ratio = inter_px / total_heat_pixels
                if ratio > 0.005:  # 0.5%
                    level_stats[id_map[cid]] = ratio

        intersection_results[level_name] = level_stats

    return intersection_results


# ===========================
# 🎨 可视化生成函数
# ===========================
def generate_advanced_dashboard(vol_id: str, serial_num: int, img_name: str,
                                img_path: str, heatmap_path: str, seg_mask: np.ndarray,
                                intersection_data: dict, id_map: dict, output_dir: str):
    base_img = cv2.imread(img_path)
    raw_heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    if base_img is None or raw_heatmap is None: return

    h, w = base_img.shape[:2]

    # 调用对齐函数
    full_heatmap = align_heatmap_to_streetview(raw_heatmap, h, w)  # 这里 raw_heatmap 是灰度图

    def put_outlined_text(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=2):
        x, y = pos
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def draw_floating_legend(canvas, stats):
        if not stats: return canvas
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]

        box_w = 260
        box_h = 30 + len(sorted_stats) * 35
        margin = 20
        start_x = w - box_w - margin
        start_y = margin

        overlay = canvas.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + box_w, start_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        put_outlined_text(canvas, "Top Segments:", (start_x + 10, start_y + 25), 0.7, (255, 255, 255), 2)

        y_cursor = start_y + 60
        for name, ratio in sorted_stats:
            # ✅ 这里使用修正后的 get_class_color，颜色将与 COCO_META 一致
            color = get_class_color(name)

            cv2.rectangle(canvas, (start_x + 10, y_cursor - 15), (start_x + 40, y_cursor + 5), color, -1)
            cv2.rectangle(canvas, (start_x + 10, y_cursor - 15), (start_x + 40, y_cursor + 5), (255, 255, 255), 1)

            text = f"{name}: {ratio:.1%}"
            put_outlined_text(canvas, text, (start_x + 50, y_cursor), 0.6, (255, 255, 255), 1)
            y_cursor += 35
        return canvas

    def draw_heatmap_overlay():
        heatmap_color = cv2.applyColorMap(full_heatmap, cv2.COLORMAP_JET)
        mask = full_heatmap > 20
        overlay = base_img.copy()
        overlay[mask] = cv2.addWeighted(base_img[mask], 0.6, heatmap_color[mask], 0.4, 0)
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        info_txt = f"Original Heatmap (Vol:{vol_id} SN:{serial_num})"
        put_outlined_text(overlay, info_txt, (20, 35), 1.0, (255, 255, 255), 2)
        return overlay

    def draw_level_analysis(level_name, color_theme, low_thr, high_thr):
        canvas = base_img.copy()
        canvas = (canvas * 0.35).astype(np.uint8)

        heat_mask = cv2.inRange(full_heatmap, low_thr, high_thr)
        mode_str = "(Incl.)" if CALCULATION_MODE == 'CUMULATIVE' else "(Diff.)"
        title = f"{level_name} {mode_str}"
        cv2.rectangle(canvas, (0, 0), (450, 60), (0, 0, 0), -1)
        put_outlined_text(canvas, title, (10, 40), 1.0, color_theme, 2)

        if np.sum(heat_mask) == 0:
            put_outlined_text(canvas, "No Gaze Data in Range", (w // 2 - 200, h // 2), 1.5, (200, 200, 200), 3)
            return canvas

        contours, _ = cv2.findContours(heat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color_theme, 3)

        stats = intersection_data.get(level_name, {})
        semantic_overlay = canvas.copy()
        labels_to_draw = []

        present_ids = np.unique(seg_mask)
        for cid in present_ids:
            if cid == 0 or cid not in id_map: continue
            c_name = id_map[cid]
            if c_name not in stats: continue

            c_mask = (seg_mask == cid).astype(np.uint8)
            inter = cv2.bitwise_and(c_mask, c_mask, mask=heat_mask)

            if np.sum(inter) > 0:
                # ✅ 语义填充颜色修正
                color = get_class_color(c_name)
                semantic_overlay[inter > 0] = color

                M = cv2.moments(inter)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if stats[c_name] > 0.03:
                        labels_to_draw.append((cX, cY, f"{c_name}\n{stats[c_name]:.1%}"))

        cv2.addWeighted(semantic_overlay, 0.7, canvas, 0.3, 0, canvas)

        for lx, ly, ltext in labels_to_draw:
            lines = ltext.split('\n')
            for i, line in enumerate(lines):
                y_pos = ly + (i * 25) - 10
                put_outlined_text(canvas, line, (lx - 40, y_pos), 0.6, (255, 255, 0), 2)

        canvas = draw_floating_legend(canvas, stats)
        return canvas

    img_tl = draw_heatmap_overlay()
    if CALCULATION_MODE == 'DIFFERENTIAL':
        img_tr = draw_level_analysis('High Attention', (0, 0, 255), 180, 255)
        img_bl = draw_level_analysis('Medium Attention', (0, 255, 255), 80, 180)
        img_br = draw_level_analysis('Low Attention', (255, 100, 0), 20, 80)
    else:
        img_tr = draw_level_analysis('High Attention', (0, 0, 255), 180, 255)
        img_bl = draw_level_analysis('Medium Attention', (0, 255, 255), 80, 255)
        img_br = draw_level_analysis('Low Attention', (255, 100, 0), 20, 255)

    top_row = np.hstack((img_tl, img_tr))
    bot_row = np.hstack((img_bl, img_br))
    final_grid = np.vstack((top_row, bot_row))

    h_g, w_g = final_grid.shape[:2]
    final_small = cv2.resize(final_grid, (w_g // 2, h_g // 2))

    save_name = f"Check_VOL{vol_id}_SN{serial_num:03d}_{img_name}"
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, final_small)


# ===========================
# 📊 Matplotlib 绘图 (单人)
# ===========================
def plot_individual_stats(vol_id, stats_list, output_dir):
    """
    修改点：强制使用 COCO_META 定义的颜色
    """
    if not stats_list: return
    df = pd.DataFrame(stats_list)
    pivot = df.pivot_table(index='HeatLevel', columns='SemanticClass', values='Ratio', aggfunc='mean').fillna(0)

    # 强制顺序
    order = ['High Attention', 'Medium Attention', 'Low Attention']
    pivot = pivot.reindex(order)

    # 筛选 Top Columns
    top_cols = pivot.sum().nlargest(12).index
    pivot = pivot[top_cols]
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # ✅ 构造颜色列表，一一对应 columns 的顺序
    chart_colors = [NAME_TO_COLOR_RGB.get(col, DEFAULT_COLOR_RGB) for col in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 使用自定义颜色列表
    pivot.plot(kind='barh', stacked=True, color=chart_colors, ax=ax)

    mode_txt = "Cumulative" if CALCULATION_MODE == 'CUMULATIVE' else "Differential"
    ax.set_title(f"Volunteer {vol_id} - Semantic Distribution ({mode_txt})")
    ax.set_xlabel("Percentage (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Analysis_Vol_{vol_id}.png"), dpi=150)
    plt.close()


# ===========================
# 📊 [新增] 聚合绘图 (所有人)
# ===========================
def generate_aggregated_analysis(csv_path, output_dir):
    """
    生成所有受访者的综合语义叠加图。
    逻辑：读取最终的CSV -> 按 Attention Level 分组 -> 计算所有人的平均占比 -> 绘图
    """
    log("[Step 6] Generating Aggregated Semantic Analysis Chart...")

    if not os.path.exists(csv_path):
        print("⚠️ CSV file not found, skipping aggregation chart.")
        return

    df = pd.read_csv(csv_path)

    # 计算每个注意力等级下，各语义类别的平均占比
    # 这里的平均是 "所有图、所有人" 的平均
    agg_pivot = df.pivot_table(index='HeatLevel', columns='SemanticClass', values='Ratio', aggfunc='mean').fillna(0)

    # 排序：High -> Medium -> Low
    order = ['High Attention', 'Medium Attention', 'Low Attention']
    agg_pivot = agg_pivot.reindex(order)

    # 筛选主要的语义类别 (比如前 15 个，防止图例太乱)
    top_cols = agg_pivot.sum().nlargest(15).index
    agg_pivot = agg_pivot[top_cols]

    # 归一化为 100%
    agg_pivot = agg_pivot.div(agg_pivot.sum(axis=1), axis=0) * 100

    # ✅ 强制颜色绑定
    chart_colors = [NAME_TO_COLOR_RGB.get(col, DEFAULT_COLOR_RGB) for col in agg_pivot.columns]

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 7))
    agg_pivot.plot(kind='barh', stacked=True, color=chart_colors, ax=ax, edgecolor='white', linewidth=0.5)

    mode_txt = "Cumulative Mode" if CALCULATION_MODE == 'CUMULATIVE' else "Differential Mode"
    ax.set_title(f"Aggregated Semantic Attention Distribution (All Volunteers) - {mode_txt}", fontsize=14)
    ax.set_xlabel("Mean Composition Ratio (%)", fontsize=12)
    ax.set_ylabel("Attention Level", fontsize=12)

    # 图例放外侧
    plt.legend(title="Semantic Class", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"Aggregated_Analysis_{CALCULATION_MODE}.png")
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Aggregated Chart Saved: {save_path}")
    plt.close()


# ===========================
# 🚀 主流程
# ===========================
def main():
    print("=" * 60)
    print(f"🚀 AI Gaze & Semantic Analysis Pipeline Started")
    print(f"⚙️  Current Mode: [{CALCULATION_MODE}]")
    print("=" * 60)

    for d in [CACHE_DIR, INDIVIDUAL_DATA_DIR, INDIVIDUAL_VIZ_DIR, CHECK_IMG_ROOT, AGGREGATED_VIZ_DIR]:
        os.makedirs(d, exist_ok=True)

    log("[Step 1] Loading Volunteers & Heatmaps...")
    volunteers_map = load_all_volunteers()
    link_heatmaps(volunteers_map)

    log("[Step 2] Checking Segmentation Cache...")
    all_imgs = set()
    for v in volunteers_map.values():
        all_imgs.update(v.trials.keys())

    streetview_files = []
    for img_name in all_imgs:
        p = os.path.join(INPUT_STREETVIEW_DIR, img_name)
        if os.path.exists(p):
            streetview_files.append(p)
        else:
            p_alt = os.path.join(INPUT_STREETVIEW_DIR, img_name.replace('.JPG', '.jpg'))
            if os.path.exists(p_alt): streetview_files.append(p_alt)

    engine = SemanticModelEngine(MODEL_NAME)
    preprocess_segmentation(streetview_files, engine)
    id_map = engine.id_to_name_map
    del engine
    tf.keras.backend.clear_session()

    log("[Step 3] Pre-loading Masks to RAM...")
    mask_cache = {}
    mask_files = glob.glob(os.path.join(CACHE_DIR, 'masks', '*.png'))
    for mp in tqdm(mask_files, desc="📥 Loading Masks"):
        name_base = os.path.splitext(os.path.basename(mp))[0]
        mask_cache[name_base] = np.array(Image.open(mp).convert('L'))

    log("[Step 4] Processing Volunteers Individually...")
    sorted_vol_ids = sorted(volunteers_map.keys())

    for vol_id in tqdm(sorted_vol_ids, desc="👤 Volunteers"):
        volunteer = volunteers_map[vol_id]
        vol_check_dir = os.path.join(CHECK_IMG_ROOT, vol_id)
        os.makedirs(vol_check_dir, exist_ok=True)
        vol_stats_list = []

        sorted_trials = sorted(volunteer.trials.values(), key=lambda x: x.serial_number)

        for trial in sorted_trials:
            if not trial.heatmap_path: continue

            img_name = trial.streetview_name
            name_key = os.path.splitext(img_name)[0]
            seg_mask = mask_cache.get(name_key)
            if seg_mask is None: continue

            intersection_res = analyze_heatmap_intersection(
                trial.heatmap_path, seg_mask, id_map
            )

            img_path = os.path.join(INPUT_STREETVIEW_DIR, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(INPUT_STREETVIEW_DIR, img_name.replace('.JPG', '.jpg'))

            generate_advanced_dashboard(
                vol_id=vol_id,
                serial_num=trial.serial_number,
                img_name=img_name,
                img_path=img_path,
                heatmap_path=trial.heatmap_path,
                seg_mask=seg_mask,
                intersection_data=intersection_res,
                id_map=id_map,
                output_dir=vol_check_dir
            )

            for level, items in intersection_res.items():
                for cls_name, ratio in items.items():
                    vol_stats_list.append({
                        'VolunteerID': vol_id,
                        'SerialNum': trial.serial_number,
                        'Image': img_name,
                        'HeatLevel': level,
                        'SemanticClass': cls_name,
                        'Ratio': ratio,
                        'Mode': CALCULATION_MODE
                    })

        if vol_stats_list:
            json_path = os.path.join(INDIVIDUAL_DATA_DIR, f"Vol_{vol_id}_data.json")
            with open(json_path, 'w') as f:
                json.dump(vol_stats_list, f, indent=2)
            # 个人图表也使用修正后的颜色
            plot_individual_stats(vol_id, vol_stats_list, INDIVIDUAL_VIZ_DIR)

        del vol_stats_list

    log("[Step 5] Aggregating All Data...")
    all_data = []
    json_files = glob.glob(os.path.join(INDIVIDUAL_DATA_DIR, "*.json"))
    for jf in json_files:
        with open(jf, 'r') as f:
            all_data.extend(json.load(f))

    if all_data:
        df_all = pd.DataFrame(all_data)
        csv_name = f"Final_Report_{CALCULATION_MODE}.csv"
        csv_path = os.path.join(OUTPUT_ROOT, csv_name)
        df_all.to_csv(csv_path, index=False)
        print(f"📄 Report saved: {csv_path}")

        # ✅ 调用新增的聚合绘图函数
        generate_aggregated_analysis(csv_path, AGGREGATED_VIZ_DIR)
    else:
        print("⚠️ No data aggregated.")


# ===========================
# 数据加载与预处理流程 (保持不变)
# ===========================
def load_all_volunteers() -> Dict[str, Volunteer]:
    print("📑 正在读取志愿者基础信息...")
    try:
        df_info = pd.read_excel(VOL_INFO_PATH)
        df_info['志愿者编号'] = df_info['志愿者编号'].astype(str).str.zfill(3)
    except Exception as e:
        print(f"❌ 读取 {VOL_INFO_PATH} 失败: {e}")
        sys.exit(1)

    volunteers_map = {}
    for _, row in df_info.iterrows():
        vol_id = row['志愿者编号']
        info = VolunteerInfo(vol_id=vol_id, age=row['年龄'], gender=row['性别'])
        volunteers_map[vol_id] = Volunteer(info=info)

    print("📑 正在读取CSV评分数据并链接...")
    csv_files = glob.glob(os.path.join(CSV_DIR, '*.csv'))
    for csv_path in csv_files:
        vol_id = os.path.basename(csv_path).split('.')[0]
        if vol_id not in volunteers_map: continue

        try:
            df_resp = pd.read_csv(csv_path)
            df_resp['Serial number'] = pd.to_numeric(df_resp['Serial number'], errors='coerce')
            df_resp = df_resp.dropna(subset=['Serial number'])
            df_resp['Serial number'] = df_resp['Serial number'].astype(int)

            for _, row in df_resp.iterrows():
                serial_num = row['Serial number']
                score = row['likert scale']
                streetview_name = f"QINCHUAN-{serial_num}.JPG"
                trial = SingleTrialData(
                    streetview_name=streetview_name,
                    serial_number=serial_num,
                    likert_scale=score
                )
                volunteers_map[vol_id].add_trial(trial)
        except Exception as e:
            print(f"⚠️ 读取 CSV {csv_path} 失败: {e}")

    return volunteers_map


def link_heatmaps(volunteers_map: Dict[str, Volunteer]):
    print("🔥 正在链接眼动热力图数据...")
    for vol_id, volunteer in volunteers_map.items():
        vol_gaze_dir = os.path.join(GAZE_DIR, vol_id)
        if not os.path.exists(vol_gaze_dir): continue

        heatmap_files = glob.glob(os.path.join(vol_gaze_dir, '*heatmap*.png'))
        for h_path in heatmap_files:
            filename = os.path.basename(h_path)
            match = re.match(r'\d+_(\d+)_eyetrack', filename)
            if match:
                serial_num = int(match.group(1))
                streetview_name = f"QINCHUAN-{serial_num}.JPG"
                if streetview_name in volunteer.trials:
                    volunteer.trials[streetview_name].heatmap_path = h_path


def preprocess_segmentation(image_files: List[str], engine: SemanticModelEngine):
    print("\n🚀 [预处理] 开始语义分割检查与缓存...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    mask_dir = os.path.join(CACHE_DIR, 'masks')
    meta_dir = os.path.join(CACHE_DIR, 'metadata')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    skipped_count = 0
    pbar = tqdm(image_files, desc="Checking Cache & Segmenting")

    for img_path in pbar:
        img_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(img_name)[0]
        mask_path = os.path.join(mask_dir, f"{name_no_ext}.png")
        json_path = os.path.join(meta_dir, f"{name_no_ext}.json")

        if os.path.exists(mask_path) and os.path.exists(json_path):
            skipped_count += 1
            pbar.set_postfix(skipped=skipped_count, last_processed="Cached")
            continue

        try:
            original_image = Image.open(img_path).convert('RGB')
            mask_np = engine.predict_mask(original_image)
            Image.fromarray(mask_np).save(mask_path)
            metadata = engine.generate_metadata(
                image_name=img_name,
                original_size=original_image.size,
                mask=mask_np,
                mask_filename=os.path.basename(mask_path)
            )
            metadata.save_json(json_path)
            pbar.set_postfix(skipped=skipped_count, last_processed=img_name)
        except Exception as e:
            print(f"❌ 处理 {img_name} 失败: {e}")

    print(f"✨ [预处理] 完成。共跳过 {skipped_count} 个已有缓存的文件。")


if __name__ == "__main__":
    main()