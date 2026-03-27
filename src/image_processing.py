# src/image_processing.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


def align_heatmap_to_streetview(heatmap, target_h, target_w):
    """
    输入: heatmap (numpy array), 目标高度, 目标宽度
    输出: 对齐后的 heatmap (numpy array)
    """
    h_heat, w_heat = heatmap.shape[:2]

    # 缩放至等高
    scale_factor = target_h / h_heat
    new_w_heat = int(w_heat * scale_factor)
    resized_heatmap = cv2.resize(heatmap, (new_w_heat, target_h))

    # 居中裁剪/填充
    aligned_heatmap = np.zeros((target_h, target_w), dtype=heatmap.dtype)  # 保持原数据类型(灰度或彩色)

    if new_w_heat >= target_w:
        start_x = (new_w_heat - target_w) // 2
        aligned_heatmap = resized_heatmap[:, start_x: start_x + target_w]
    else:
        start_x = (target_w - new_w_heat) // 2
        aligned_heatmap[:, start_x: start_x + new_w_heat] = resized_heatmap

    return aligned_heatmap


def overlay_heatmap_on_streetview(streetview_path, heatmap_path, alpha=0.6):
    """
    将热力图叠加到街景图上。
    逻辑：
    1. 读取两张图片。
    2. 将热力图缩放至与街景图【等高】。
    3. 如果缩放后的热力图比街景图【宽】，则进行【水平居中裁剪】，裁掉左右多余部分。
    4. 如果缩放后的热力图比街景图【窄】（虽然少见但为了健壮性），则进行【水平居中填充】黑边。
    5. 执行加权叠加。
    """
    # 1. 读取图片
    streetview = cv2.imread(streetview_path)
    heatmap = cv2.imread(heatmap_path)

    if streetview is None:
        raise FileNotFoundError(f"Streetview image not found: {streetview_path}")
    if heatmap is None:
        raise FileNotFoundError(f"Heatmap image not found: {heatmap_path}")

    h_base, w_base = streetview.shape[:2]
    h_heat, w_heat = heatmap.shape[:2]

    # 2. 将热力图缩放至与街景图等高
    scale_factor = h_base / h_heat
    new_w_heat = int(w_heat * scale_factor)
    resized_heatmap = cv2.resize(heatmap, (new_w_heat, h_base))

    # 3. 居中对齐处理
    aligned_heatmap = np.zeros_like(streetview)  # 创建一个和街景图一样大的黑底画布

    if new_w_heat >= w_base:
        # 情况A：热力图更宽 -> 居中裁剪
        start_x = (new_w_heat - w_base) // 2
        # 取中间部分
        aligned_heatmap = resized_heatmap[:, start_x: start_x + w_base]
    else:
        # 情况B：热力图更窄 -> 居中填充（两边补黑）
        start_x = (w_base - new_w_heat) // 2
        aligned_heatmap[:, start_x: start_x + new_w_heat] = resized_heatmap

    # 4. 叠加 (AddWeighted)
    # 确保aligned_heatmap也是3通道
    if len(aligned_heatmap.shape) == 2:
        aligned_heatmap = cv2.cvtColor(aligned_heatmap, cv2.COLOR_GRAY2BGR)

    # 叠加公式: dst = src1*alpha + src2*beta + gamma
    # 这里 beta = 1 - alpha
    overlay = cv2.addWeighted(streetview, 1 - alpha, aligned_heatmap, alpha, 0)

    return overlay, aligned_heatmap


if __name__ == "__main__":
    # ================= 独立测试入口 =================
    # 1. 动态获取路径，防止 "FileNotFoundError"
    # 获取当前脚本(image_processing.py)所在的绝对路径
    current_scr_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (scr的上一级)
    project_root = os.path.dirname(current_scr_dir)

    # 2. 构建输入文件的绝对路径 (请确保文件名与您本地一致)
    TEST_STREET_PATH = os.path.join(project_root, "data", "input_streetview", "QINCHUAN-62.JPG")

    # 注意：请检查您的热力图路径中间是否有 "001" 文件夹，这里根据您之前的描述补全了路径
    TEST_HEAT_PATH = os.path.join(project_root, "data", "experiment_data", "gaze_heatmap", "001",
                                  "001_62_eyetrack_heatmap_20250929_190147.png")

    # 3. 设置输出路径
    OUTPUT_DIR = os.path.join(project_root, "output", "test_output")
    OUTPUT_TEST_PATH = os.path.join(OUTPUT_DIR, "test_overlay_result.png")

    print(f"🔹 Testing overlay module...")
    print(f"📍 Project Root detected: {project_root}")

    # 4. 自动创建输出文件夹 (防止目录不存在报错)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Created output directory: {OUTPUT_DIR}")

    try:
        # 5. 检查输入文件是否存在
        if not os.path.exists(TEST_STREET_PATH):
            print(f"❌ Error: Streetview file missing:\n   {TEST_STREET_PATH}")
        elif not os.path.exists(TEST_HEAT_PATH):
            print(f"❌ Error: Heatmap file missing:\n   {TEST_HEAT_PATH}")
        else:
            # 6. 执行核心叠加函数
            result_img, aligned_heat = overlay_heatmap_on_streetview(TEST_STREET_PATH, TEST_HEAT_PATH)

            # 7. 保存结果
            cv2.imwrite(OUTPUT_TEST_PATH, result_img)
            print("-" * 30)
            print(f"✅ Success! Overlay saved to:\n   {OUTPUT_TEST_PATH}")
            print(f"ℹ️  Aligned Heatmap Shape: {aligned_heat.shape}")
            print("-" * 30)

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        # 打印详细报错堆栈，方便调试
        import traceback

        traceback.print_exc()