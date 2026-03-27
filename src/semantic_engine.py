# -*- coding: utf-8 -*-
import os
import sys
import warnings
import numpy as np
from PIL import Image

# ✅ 先设置 TF 的 C++ 日志级别，再导入 TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 0=全部, 1=INFO, 2=WARNING, 3=ERROR

import tensorflow as tf
import logging
from absl import logging as absl_logging

# 关闭 Python 自带 warnings
warnings.filterwarnings('ignore')

# ✅ 进一步关闭 TF / absl 的 Python 日志
tf.get_logger().setLevel('ERROR')              # 只保留 ERROR
absl_logging.set_verbosity(absl_logging.ERROR) # absl 只输出 ERROR
logging.getLogger('absl').setLevel(logging.ERROR)


# 导入你的 COCO_META
try:
    from src.COCO_META import COCO_META
except ImportError:
    # 简化处理，如果找不到文件
    COCO_META = [{'id': i, 'name': f'class_{i}'} for i in range(100)]

from src.data_structures import StreetViewMetadata, SemanticClassMeta

# 配置环境 (同原代码)
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class SemanticModelEngine:
    """专注于模型加载和推理的引擎"""

    def __init__(self, model_name, local_path_root='data/weights'):
        self.model_path = os.path.join(local_path_root, model_name)
        self._load_model()
        self.id_to_name_map = self._create_id_to_name_map()

    def _load_model(self):
        # ... (此处省略原有的详细模型下载和加载代码，假设模型已在本地) ...
        try:
            print(f"➡️ [引擎] 正在加载模型: {self.model_path}")
            self.loaded_model = tf.saved_model.load(self.model_path)
            self.inference_function = self.loaded_model.signatures['serving_default']
            print("✅ [引擎] 模型加载完成。")
        except Exception as e:
            print(f"❌ [引擎] 模型加载失败: {e}")
            sys.exit(1)

    def _create_id_to_name_map(self):
        id_map = {}
        for i, item in enumerate(COCO_META):
            # COCO 输出的 ID 通常是从 0 或 1 开始，需要根据实际模型调整
            # 这里假设模型输出 ID 对应 COCO_META 的索引 + 1 (背景为0)
            class_id = i + 1
            id_map[class_id] = item['name']
        return id_map

    def predict_mask(self, image: Image.Image) -> np.ndarray:
        """仅返回分割掩码 (Numpy数组, uint8)"""
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
        if len(input_tensor.shape) == 4:
            input_tensor = tf.squeeze(input_tensor, axis=0)
        predictions = self.inference_function(input_tensor=input_tensor)
        # 确保返回的是 2D uint8 数组
        mask = tf.squeeze(predictions['semantic_pred']).numpy().astype(np.uint8)
        return mask

    def generate_metadata(self, image_name: str, original_size: tuple, mask: np.ndarray,
                          mask_filename: str) -> StreetViewMetadata:
        """基于分割掩码计算元数据"""
        width, height = original_size
        total_pixels = width * height

        metadata = StreetViewMetadata(
            image_name=image_name,
            width=width,
            height=height,
            mask_filename=mask_filename
        )

        present_ids = np.unique(mask)
        for cid in present_ids:
            # 跳过背景 (ID 0) 或未知类别
            if cid == 0 or cid not in self.id_to_name_map:
                continue

            count = np.sum(mask == cid)
            ratio = count / total_pixels
            class_name = self.id_to_name_map[cid]

            meta_item = SemanticClassMeta(
                class_id=int(cid),  # 确保 JSON 兼容
                name=class_name,
                pixel_count=int(count),
                ratio=float(ratio)
            )
            # 使用字符串 ID 作为键
            metadata.semantic_stats[str(cid)] = meta_item

        return metadata