# src/clip_adapter.py
# -*- coding: utf-8 -*-
import os
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

# 尝试引入 transformers
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("❌ 未找到 transformers 包。请运行 'pip install transformers' 安装。")


class CLIPWrapper:
    def __init__(self, model_path_or_name: str, device=None):
        """
        初始化 CLIP 模型
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing CLIP on {self.device}...")

        # 路径检查逻辑
        if os.path.exists(model_path_or_name):
            print(f"  - Loading from local path: {os.path.abspath(model_path_or_name)}")
        else:
            print(f"  - Path not found locally, trying to download/load from Hub: {model_path_or_name}")

        try:
            self.processor = CLIPProcessor.from_pretrained(model_path_or_name)
            self.model = CLIPModel.from_pretrained(model_path_or_name).to(self.device)
            self.model.eval()
            print("✅ CLIP Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load CLIP model: {e}")

    def _crop_object_by_mask(self, image_bgr: np.ndarray, mask: np.ndarray) -> Optional[Image.Image]:
        """
        内部辅助函数：根据 Mask 裁剪物体，并转为 PIL 格式
        """
        if mask is None or np.sum(mask) == 0:
            return None

        # 1. 获取 Bounding Box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # 2. 裁剪图像 (Crop)
        # 增加一点点 padding (比如 10px) 让物体稍微完整一点，有助于识别
        pad = 10
        h, w = image_bgr.shape[:2]
        rmin = max(0, rmin - pad)
        rmax = min(h, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(w, cmax + pad)

        crop_bgr = image_bgr[rmin:rmax, cmin:cmax]

        # 3. (可选) Masking: 将物体周围的背景变黑，防止 CLIP 识别到背景
        # 注意：这步取决于 Mask 及其裁剪区域的对应关系，
        # 简单起见，这里只做矩形裁剪 (Crop)，对于大部分场景通常足够且更鲁棒。

        # 转为 PIL RGB
        if crop_bgr.size == 0: return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)

    def classify_object(self, image_bgr: np.ndarray, mask: np.ndarray, text_labels: List[str],
                        threshold: float = 0.2) -> Tuple[str, float]:
        """
        核心功能：输入原图和 SAM2 的 Mask，识别物体类别
        :param image_bgr: 原始街景图 (OpenCV BGR)
        :param mask: SAM2 生成的布尔值 Mask (True/False) 或 0/1 矩阵
        :param text_labels: 候选标签列表
        :param threshold: 置信度门槛，低于此值返回 "Unknown"
        """
        # 1. 裁剪物体
        pil_image = self._crop_object_by_mask(image_bgr, mask)
        if pil_image is None:
            return "Background/Noise", 0.0

        if not text_labels:
            return "Unknown", 0.0

        # 2. 预处理
        try:
            inputs = self.processor(
                text=text_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
        except Exception as e:
            print(f"⚠️ CLIP Preprocessing Error: {e}")
            return "Error", 0.0

        # 3. 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            best_score, best_idx = probs.max(dim=1)

            best_score = best_score.item()
            best_idx = best_idx.item()

        # 4. 阈值判断
        if best_score < threshold:
            return "Unknown", best_score

        return text_labels[best_idx], best_score


# ==============================================================================
# 🧪 论文专用测试入口
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 路径设置 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    local_weights_path = os.path.join(project_root, "data", "weights", "clip")
    hf_model_name = "openai/clip-vit-large-patch14"

    if os.path.exists(local_weights_path):
        MODEL_PATH = local_weights_path
    else:
        print(f"⚠️ Local path '{local_weights_path}' not found. Using HuggingFace Hub.")
        MODEL_PATH = hf_model_name

    # --- 2. 模拟 SAM2 数据 ---
    print("🎨 Creating dummy streetscape image & mask...")
    # 模拟一张 800x600 的图
    dummy_scene = np.zeros((600, 800, 3), dtype=np.uint8)
    dummy_scene[:] = (200, 200, 200)  # 灰色背景

    # 模拟一个“红色的车”在中间 (BGR: 0, 0, 255)
    cv2.rectangle(dummy_scene, (300, 400), (500, 500), (0, 0, 255), -1)

    # 模拟 SAM2 输出的 Mask (对应上面的车)
    dummy_mask = np.zeros((600, 800), dtype=bool)
    dummy_mask[400:500, 300:500] = True

    # --- 3. 定义论文专用的候选标签 (核心步骤) ---
    # 技巧：使用 "category: description" 的格式，方便后续处理，或者直接用描述
    # 这些标签直接对应你论文 Section 3.3.1 的分类
    qinchuan_labels = [
        "traditional tiled roof",  # 传统屋顶
        "aged timber facade wooden wall",  # 木构立面
        "ancestral hall gate tower",  # 门楼/祠堂
        "stone bridge and water stream",  # 水系/桥
        "ancient tree vegetation",  # 古树/植被
        "modern car vehicle",  # 现代车辆 (干扰)
        "pile of trash visual clutter",  # 杂物 (干扰)
        "paved stone path ground",  # 铺地
        "blue sky"  # 天空
    ]
    print(f"📝 Research Categories: {qinchuan_labels}")

    # --- 4. 运行测试 ---
    try:
        clip_tool = CLIPWrapper(model_path_or_name=MODEL_PATH)

        print("🔍 Classifying masked object (The Red Car)...")
        # 传入原图 + Mask
        label, score = clip_tool.classify_object(dummy_scene, dummy_mask, qinchuan_labels)

        print("-" * 30)
        print(f"🏆 Identified Object: '{label}'")
        print(f"📊 Confidence Score: {score:.4f}")
        print("-" * 30)

        # 简单的验证逻辑
        if "car" in label or "vehicle" in label:
            print("✅ Test Passed: Successfully identified the modern vehicle.")
        else:
            print("⚠️ Test Result Unexpected (check dummy image or labels).")

    except Exception as e:
        print(f"❌ Test Failed: {e}")