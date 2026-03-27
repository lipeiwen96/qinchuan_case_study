# src/heatmap_extractor.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys

# 尝试导入同级目录下的 image_processing 模块
try:
    from image_processing import overlay_heatmap_on_streetview
except ImportError:
    # 如果作为脚本独立运行，需要调整路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from image_processing import overlay_heatmap_on_streetview


class HeatmapRedZoneExtractor:
    def __init__(self, streetview_path, heatmap_path):
        self.streetview_path = streetview_path
        self.heatmap_path = heatmap_path

        # 获取叠加底图和对齐后的热力图 (BGR)
        # overlay_img: 用于可视化的底图
        # aligned_heatmap: 用于计算红区的源数据 (保持BGR彩色，因为要转HSV)
        self.overlay_img, self.aligned_heatmap = overlay_heatmap_on_streetview(
            streetview_path, heatmap_path, alpha=0.5
        )

    def extract_red_zones(self, min_area=50, grid_spacing=30):
        """
        核心算法：在 HSV 空间提取真正的红色高关注区域，并进行网格化采样
        :param min_area: 最小连通域面积，小于此值的区域被视为噪点
        :param grid_spacing: 采样点的像素间距（越小越密集）
        """
        # 1. 转为 HSV 色彩空间
        hsv_map = cv2.cvtColor(self.aligned_heatmap, cv2.COLOR_BGR2HSV)

        # 2. 定义红色的范围 (OpenCV中 H: 0-180, S: 0-255, V: 0-255)
        # 红色通常分布在色相环的两端 (0-10 和 170-180)
        # 这里的 S 和 V 阈值决定了红色的"纯度"和"亮度"，越高越接近核心红区

        # 范围1：红色 (原: 0-10 -> 改为: 0-25)
        # 25 左右已经包含了很多橙色，如果是 30 就开始变成黄色了
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([25, 255, 255])  # 🔥 关键修改：把 10 改成 20 或 25

        # 范围2：红色 (原: 170-180 -> 改为: 160-180)
        # 这一端通常对应紫红，热力图里可能较少，但为了保险可以放宽
        lower_red2 = np.array([160, 100, 100])  # 🔥 关键修改：把 170 改成 160
        upper_red2 = np.array([180, 255, 255])

        # 3. 创建掩膜
        mask1 = cv2.inRange(hsv_map, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_map, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 4. 形态学操作修改
        # 只保留去除微小噪点，不再填补空洞，保留原始的"环状"或"中空"结构
        kernel = np.ones((3, 3), np.uint8)
        # 仅做一次开运算去噪 (Open: 先腐蚀后膨胀)，去小白点
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        # 🔥 注释掉闭运算，保留空洞
        # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        # 🔥🔥膨胀操作：人为扩大范围
        # iterations=1 表示膨胀一次，数值越大圈越大
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        # 5. 提取连通域和重心
        # 使用 RETR_EXTERNAL 只找外轮廓，内部空洞自然被排除在掩膜之外
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        all_sample_points = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue

            valid_contours.append(cnt)

            # --- 新增：网格化密集采样逻辑 ---
            # 1. 获取该轮廓的边界框 (Bounding Rect)
            x, y, w, h = cv2.boundingRect(cnt)

            # 2. 在边界框内生成网格点
            # 使用 np.arange 生成从 x 到 x+w 的点，步长为 grid_spacing
            xs = np.arange(x + grid_spacing // 2, x + w, grid_spacing)
            ys = np.arange(y + grid_spacing // 2, y + h, grid_spacing)

            if len(xs) == 0 or len(ys) == 0:
                # 如果区域太小连一个点都放不下，强制取重心作为保底
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    all_sample_points.append((cx, cy))
                continue

            # 3. 遍历网格点，检查是否在多边形内部 (pointPolygonTest)
            # 同时也检查该点在 red_mask 上是否为白色 (255)，双重验证
            for py in ys:
                for px in xs:
                    px, py = int(px), int(py)

                    # 检查点是否在轮廓内 (返回值为正表示在内部，0在边缘，负在外部)
                    # 且检查掩膜值，确保不会采到空洞里
                    if cv2.pointPolygonTest(cnt, (px, py), False) >= 0:
                        # 再次检查 mask 像素值，确保这里真的是红区 (防止复杂轮廓的误判)
                        if red_mask[py, px] > 0:
                            all_sample_points.append((px, py))

        return valid_contours, all_sample_points

    def visualize(self, output_path):
        """
        绘制提取结果
        """
        # 可以调整 grid_spacing 来控制点的稀疏程度，这里设为 30 像素
        contours, points = self.extract_red_zones(grid_spacing=25)

        # 在底图上绘制
        canvas = self.overlay_img.copy()

        # 绘制轮廓 (亮黄色，线宽2)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)

        # 绘制采样点 (红色实心圆 + 白色描边，尺寸改小)
        for i, (px, py) in enumerate(points):
            # 点改小一点：半径 3
            cv2.circle(canvas, (px, py), 4, (255, 255, 255), -1)  # 白底
            cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1)    # 红芯

            # 只有点比较少的时候才显示序号，太多就不显示了，避免乱
            if len(points) < 50:
                cv2.putText(canvas, str(i + 1), (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 保存
        cv2.imwrite(output_path, canvas)
        print(f"✅ Extraction result saved to: {output_path}")
        print(f"📊 Stats: {len(contours)} contours, {len(points)} sample points generated.")


if __name__ == "__main__":
    # =================== 独立调试入口 ===================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 获取输入文件夹中的所有街景和热力图
    streetview_dir = os.path.join(project_root, "data/input_streetview")
    heatmap_dir = os.path.join(project_root, "data/experiment_data/gaze_heatmap/001")
    output_dir = os.path.join(project_root, "output/test_output/test_heatmap_extractor")

    # 获取文件列表
    streetview_files = sorted([f for f in os.listdir(streetview_dir) if f.endswith(".JPG")])
    heatmap_files = sorted([f for f in os.listdir(heatmap_dir) if f.endswith(".png")])
    print(f"Streetview files: {streetview_files}")
    print(f"Heatmap files: {heatmap_files}")

    # 创建字典按ID关联街景和热力图
    streetview_dict = {f.split('-')[1].split('.')[0]: f for f in streetview_files}
    heatmap_dict = {f.split('_')[1]: f for f in heatmap_files}

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个文件，如果ID匹配
    for id, streetview_file in streetview_dict.items():
        if id in heatmap_dict:
            heatmap_file = heatmap_dict[id]
            streetview_path = os.path.join(streetview_dir, streetview_file)
            heatmap_path = os.path.join(heatmap_dir, heatmap_file)

            # 输出结果路径
            output_file = os.path.join(output_dir, f"red_zone_{id}.png")

            # 处理每对图像和热力图
            print(f"🔹 Processing {streetview_file} and {heatmap_file}...")
            extractor = HeatmapRedZoneExtractor(streetview_path, heatmap_path)
            extractor.visualize(output_file)
        else:
            print(f"❌ No matching heatmap for {streetview_file}, skipping.")