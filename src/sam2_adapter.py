import torch
import os
import cv2
import numpy as np

# 因为我们已经安装了 sam2，所以这里直接引用，不需要 sys.path.append
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    raise ImportError("❌ 未找到 sam2 包。请确保你已经在 sam2_repo 目录下运行了 'pip install -e .'")


class SAM2Wrapper:
    def __init__(self, checkpoint_path, config_path=None, device=None):
        """
        :param checkpoint_path: 权重文件的绝对路径
        :param config_path: (可选) 配置文件路径。如果不传，自动去 sam2 安装目录找。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing SAM 2 on {self.device}...")

        # 1. 自动处理配置文件路径
        # 如果你没传 config_path，我们尝试自动根据 model type 推断
        if config_path is None:
            # 获取 sam2 包的安装位置
            import sam2
            package_dir = os.path.dirname(sam2.__file__)  # .../sam2_repo/sam2
            # 默认假设使用 large 模型配置
            config_path = os.path.join(package_dir, "configs", "sam2.1", "sam2.1_hiera_l.yaml")

        # 2. 检查文件
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        print(f"  - Checkpoint: {checkpoint_path}")
        print(f"  - Config: {config_path}")

        # 3. 加载模型
        try:
            self.model = build_sam2(config_path, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            print("✅ SAM 2 Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM 2: {e}")

        # 优化设置
        self.inference_context = torch.inference_mode()
        self.autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)

    def predict(self, image, points, labels=None):
        # 确保图片是 RGB
        if image.shape[-1] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        input_points = np.array(points)
        input_labels = np.array(labels) if labels is not None else np.array([1] * len(input_points))

        with self.inference_context, self.autocast_context:
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            # 取最高分 mask
            best_idx = np.argmax(scores)
            return masks[best_idx], scores[best_idx]