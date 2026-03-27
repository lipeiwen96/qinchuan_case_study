"""
main_segment_raw_streetscape_results.py
处理场地的原始街景数据，从而获得语义分割的结果，导出成excel文件
"""
# -*- coding: utf-8 -*-

# ===================================================================
# Part 0: 环境配置与系统信息
# ===================================================================
USE_GPU = True

import os
import logging
import sys
import warnings

# 过滤警告
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

try:
    import psutil
    import cv2
except ImportError as e:
    print(f"❌ 错误：缺少必要模块。请运行 'pip install psutil opencv-python' 安装。")
    sys.exit(1)

import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import csv

# ===================================================================
# Part 1: COCO 数据集元信息
# ===================================================================
# 为了确保代码独立运行，这里保留完整列表
from scr.COCO_META import COCO_META

# ===================================================================
# Part 2: 数据结构与功能函数
# ===================================================================
from dataclasses import dataclass, field


@dataclass
class SemanticFeature:
    name: str = field(default="unnamed")
    width: int = field(default=0)
    height: int = field(default=0)
    counts_by_id: dict = field(default_factory=dict)

    @property
    def whole_area(self) -> int: return self.width * self.height if self.width > 0 else 1

    def get_percent(self, count):
        return count / self.whole_area

    def to_dict(self):
        data = {"Image Name": self.name}
        for cid, count in self.counts_by_id.items():
            pct = self.get_percent(count)
            data[f"Class_{cid}_Pct"] = pct
        return data


def create_coco_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for i, item in enumerate(COCO_META):
        class_id = i + 1
        colormap[class_id] = item['color']
    return colormap


def create_id_to_name_map():
    id_map = {}
    for i, item in enumerate(COCO_META):
        class_id = i + 1
        id_map[class_id] = {'name': item['name'], 'color': tuple(item['color'])}
    return id_map


def label_to_color_image(label: np.ndarray, colormap: np.ndarray) -> Image.Image:
    return Image.fromarray(colormap[label])


class SemanticModel:
    def __init__(self, model_name, local_path=None):
        self.model_path = None
        if local_path:
            if os.path.exists(local_path):
                if os.path.isdir(local_path) and "saved_model.pb" in os.listdir(local_path):
                    print(f"➡️ [本地加载] 正在加载解压后的模型: {local_path}")
                    self.model_path = local_path
                elif os.path.isfile(local_path) and local_path.endswith(".tar.gz"):
                    print(f"➡️ [本地加载] 正在解压并加载压缩包: {local_path}")
                    import tarfile
                    extract_path = os.path.dirname(local_path)
                    with tarfile.open(local_path) as tar:
                        tar.extractall(path=extract_path)
                    self.model_path = os.path.join(extract_path, model_name)
                else:
                    self.model_path = local_path
            else:
                print(f"⚠️ [本地加载] 警告: 本地路径 '{local_path}' 不存在。")

        if not self.model_path:
            print(f"⬇️ [网络加载] 正在下载模型 '{model_name}'...")
            self.model_url = f'https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/{model_name}.tar.gz'
            downloaded_path = tf.keras.utils.get_file(fname=model_name + ".tar.gz", origin=self.model_url, untar=True)
            self.model_path = os.path.join(os.path.dirname(downloaded_path), model_name)

        try:
            self.loaded_model = tf.saved_model.load(self.model_path)
            self.inference_function = self.loaded_model.signatures['serving_default']
            print(f"✅ 模型加载成功！路径: {self.model_path}")
        except Exception as e:
            print(f"❌ 模型加载严重错误: {e}")
            sys.exit(1)

    def predict(self, image: Image.Image) -> tf.Tensor:
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
        if len(input_tensor.shape) == 4:
            input_tensor = tf.squeeze(input_tensor, axis=0)
        predictions = self.inference_function(input_tensor=input_tensor)
        return tf.squeeze(predictions['semantic_pred'])


# ===================================================================
# Part 3: 可视化核心 (图例侧边栏 + 轮廓线)
# ===================================================================

def draw_text_with_outline(draw, position, text, font, text_color, outline_color='black', outline_width=2):
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text(position, text, font=font, fill=text_color)


def visualize_segmentation_sidebar(original_image, segmentation_map, colormap, id_map, feature):
    """
    1. 混合图像
    2. 绘制轮廓线 (Contours)
    3. 扩展右侧画布制作图例
    4. 绘制中心文字 (修改：标注所有轮廓，字号极小)
    """
    # 1. 基础混合
    color_mask = label_to_color_image(segmentation_map, colormap)
    blended_image = Image.blend(original_image, color_mask, alpha=0.6)

    # 2. 准备绘制轮廓 (需要转为 Numpy 格式供 OpenCV 使用)
    img_np = np.array(blended_image)
    total_pixels = original_image.width * original_image.height
    present_ids = np.unique(segmentation_map)
    legend_data = []

    # --- 字体设置修改 ---
    try:
        font_path = "arial.ttf" if os.name == 'nt' else "DejaVuSans.ttf"

        # [新增] 定义极小的字体用于标注每一个轮廓
        # 使用更大的除数 (如 200) 使字体变小，同时设定一个最小值 (如 8) 防止过于模糊
        tiny_font_size = max(5, int(original_image.width / 200))
        tiny_font = ImageFont.truetype(font_path, tiny_font_size)
        print(f"DEBUG: Tiny font size set to: {tiny_font_size}") # 调试用，可删除

        # 侧边栏图例文字大小 (保持不变，为了清晰)
        legend_font_size = max(11, int(original_image.width / 80))
        legend_font = ImageFont.truetype(font_path, legend_font_size)
    except:
        tiny_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()
    # --------------------

    # 遍历所有类别绘制轮廓
    for class_id in present_ids:
        if class_id == 0 or class_id not in id_map: continue

        # 获取掩码
        mask = np.uint8(segmentation_map == class_id) * 255
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === 绘制轮廓线 (白色，线宽1) ===
        cv2.drawContours(img_np, contours, -1, (255, 255, 255), 1)

        # 收集数据用于后续文字和图例
        class_info = id_map[class_id]
        total_count = np.sum(segmentation_map == class_id)
        total_pct = (total_count / total_pixels) * 100
        legend_data.append({
            'name': class_info['name'],
            'pct': total_pct,
            'color': class_info['color'],
            'id': class_id,
            'contours': contours  # 存下来算中心点
        })

    # 将绘制了轮廓的 Numpy 图转回 PIL
    img_with_contours = Image.fromarray(img_np)

    # 3. 创建扩展画布 (右侧增加侧边栏)
    sidebar_width = max(250, int(original_image.width * 0.25))
    new_width = original_image.width + sidebar_width
    new_height = original_image.height

    # 创建黑色背景大图
    final_canvas = Image.new('RGB', (new_width, new_height), (30, 30, 30))
    # 贴入图片
    final_canvas.paste(img_with_contours, (0, 0))

    draw = ImageDraw.Draw(final_canvas)

    # 4. 绘制中心文字 (在左侧图片区域) - 修改：标注所有轮廓
    for item in legend_data:
        class_name = item['name']
        contours = item['contours']

        for cnt in contours:
            area = cv2.contourArea(cnt)
            blob_pct = (area / total_pixels) * 100

            # --- 修改点：移除面积过滤 ---
            # 过滤太小的区域 (小于 0.2%)
            # if blob_pct < 0.2: continue
            # -------------------------

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 简化的文字：只显示名称和比例
                label_text = f"{class_name}\n{blob_pct:.3f}%"

                # --- 修改点：使用 tiny_font 计算和绘制 ---
                bbox = draw.textbbox((0, 0), label_text, font=tiny_font)
                w_text = bbox[2] - bbox[0]
                h_text = bbox[3] - bbox[1]

                # 确保文字画在图片范围内，使用较细的描边 (outline_width=1)
                draw_text_with_outline(draw, (cX - w_text // 2, cY - h_text // 2),
                                       label_text, tiny_font, (255, 255, 255), outline_width=1)
                # -------------------------------------

    # 5. 绘制右侧图例 (Sidebar) - 保持不变
    legend_data.sort(key=lambda x: x['pct'], reverse=True)

    x_start = original_image.width + 20  # 侧边栏起始 X
    y_start = 20  # 顶部留白
    line_height = legend_font.getbbox("Tg")[3] + 15  # 行高

    # 侧边栏标题
    draw.text((x_start, y_start), "SEMANTIC LEGEND", font=legend_font, fill=(200, 200, 200))
    y_start += 30

    for item in legend_data:
        # 绘制色块
        draw.rectangle([x_start, y_start + 4, x_start + 15, y_start + 19],
                       fill=tuple(item['color']), outline='white', width=1)

        # 绘制文字
        text = f"{item['name']}: {item['pct']:.3f}%"
        draw.text((x_start + 25, y_start), text, font=legend_font, fill='white')

        y_start += line_height

    return final_canvas


# ===================================================================
# Part 4: 主程序入口
# ===================================================================
if __name__ == '__main__':
    INPUT_FOLDER = 'data/input_streetview'
    OUTPUT_FOLDER = 'output/0.raw_streetscape_segmented_results/results'  # 新的输出文件夹
    REPORT_FILE = 'output/0.raw_streetscape_segmented_results/streetscape_results_report.csv'

    MODEL_NAME = 'resnet50_kmax_deeplab_coco_train'
    LOCAL_MODEL_PATH = os.path.join('data/weights', MODEL_NAME)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("🚀 正在初始化语义分割模型...")
    model = SemanticModel(model_name=MODEL_NAME, local_path=LOCAL_MODEL_PATH)

    colormap = create_coco_colormap()
    ID_TO_NAME_MAP = create_id_to_name_map()

    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 错误: 输入文件夹 '{INPUT_FOLDER}' 不存在！")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("❌ 未找到图片。")
        sys.exit(1)

    all_stats = []

    for i, filename in enumerate(image_files):
        print(f"\n[{i + 1}/{len(image_files)}] 处理: {filename}")
        image_path = os.path.join(INPUT_FOLDER, filename)

        try:
            original_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"    -> 图片读取失败: {e}")
            continue

        try:
            segmentation_map = model.predict(original_image).numpy()
        except Exception as e:
            print(f"    -> ❌ 预测出错: {e}")
            continue

        feature = SemanticFeature(name=filename, width=original_image.width, height=original_image.height)
        ids, counts = np.unique(segmentation_map, return_counts=True)
        for cid, count in zip(ids, counts):
            feature.counts_by_id[cid] = count
        all_stats.append(feature.to_dict())

        # 调用新的可视化函数 (带侧边栏)
        result_image = visualize_segmentation_sidebar(
            original_image, segmentation_map, colormap, ID_TO_NAME_MAP, feature
        )

        save_path = os.path.join(OUTPUT_FOLDER, filename)
        result_image.save(save_path)
        print(f"    -> 结果已保存: {save_path}")

    if all_stats:
        # --- 修改 CSV 保存逻辑，增加解释行 ---
        df_data = pd.DataFrame(all_stats)
        headers = df_data.columns.tolist()

        # 1. 构建解释行字典
        desc_row_dict = {}
        for col in headers:
            if col == "Image Name":
                desc_row_dict[col] = "Category Description ->"  # 第一列的提示
            elif col.startswith("Class_") and col.endswith("_Pct"):
                try:
                    # 从列名中提取 Class ID (例如 "Class_1_Pct" -> 1)
                    class_id_str = col.split('_')[1]
                    class_id = int(class_id_str)
                    # 从全局映射表中查找对应的名称
                    name = ID_TO_NAME_MAP.get(class_id, {}).get('name', 'Unknown')
                    desc_row_dict[col] = name
                except:
                    desc_row_dict[col] = ""
            else:
                desc_row_dict[col] = ""

        # 2. 使用 csv 模块手动写入前两行 (表头 + 解释行)
        # 使用 utf-8-sig 编码以便 Excel 正确打开中文或特殊字符
        with open(REPORT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(headers)
            # 根据表头顺序提取解释值并写入第二行
            desc_values = [desc_row_dict[h] for h in headers]
            writer.writerow(desc_values)

        # 3. 使用 Pandas 追加实际数据行 (跳过 Pandas 自己的表头)
        df_data.to_csv(REPORT_FILE, mode='a', index=False, header=False, encoding='utf-8-sig')

        print(f"\n📊 统计报告(含解释行)已生成: {REPORT_FILE}")
        # ------------------------------------

    print("✅ 所有任务完成。")