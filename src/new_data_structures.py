# src/new_data_structures.py
# -*- coding: utf-8 -*-
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional


# ===========================
# 新增：Saliency-to-Object 过程结构
# ===========================

@dataclass
class SamplingPoint:
    """单个采样点信息"""
    point_id: int
    coords: Tuple[int, int]  # (x, y) 像素坐标
    intensity: int  # 该点的热力值 (0-255)，代表注视强度
    rank: int  # 在簇内的排名 (Top-1, Top-2...)


@dataclass
class ProposedObject:
    """SAM2 生成的候选物体 (未经过滤)"""
    # 1. 必填字段 (无默认值) 放前面
    proposal_id: str  # e.g., "cluster_01_p_01"
    score_a: float  # SAM2 Predicted IoU
    source_point_id: int  # 来源的采样点ID

    # 2. 选填字段 (有默认值) 放后面
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # 二值掩膜
    # 物体类别名称 (例如 "Window", "Tree")
    category_name: str = "Unknown"


@dataclass
class HeatmapCluster:
    """
    热力簇 (Connected Component)
    对应 Step 1: 从'场'到'域'
    """
    # 1. 必填字段
    cluster_id: int
    area: int
    centroid: Tuple[int, int]  # 几何重心 (仅作备用)

    # 2. 选填字段 (有默认值)
    contour_points: Optional[np.ndarray] = field(default=None, repr=False)  # 轮廓坐标

    # 对应 Step 2: 斑块内多点采样
    sample_points: List[SamplingPoint] = field(default_factory=list)

    # 对应 Step 3: 局部推断结果
    candidates: List[ProposedObject] = field(default_factory=list)

    # 对应 Step 3: 内部竞争后的胜出者 (Winner)
    representative_object: Optional[ProposedObject] = None


@dataclass
class ValidatedObject:
    """
    最终验证通过的物体
    对应 Step 4: 全局验证结果
    """
    # 1. 必填字段
    obj_id: int
    original_cluster_id: int
    score_a: float  # Validity (SAM Confidence)
    score_b: float  # Attention Density (Visual Attention Score)

    # 2. 选填字段
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    # 🔥 [修改] 统一字段名为 category_name，用于存储最终的中文标签
    category_name: str = "Unknown"

    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # 可视化颜色
    consensus_count: int = 0
    consensus_score: float = 0.0


@dataclass
class StreetViewAnalysisResult:
    """单张街景图的完整分析结果容器"""
    image_name: str
    clusters: List[HeatmapCluster] = field(default_factory=list)
    final_objects: List[ValidatedObject] = field(default_factory=list)

    def summary(self):
        return f"Image: {self.image_name} | Clusters: {len(self.clusters)} | Valid Objects: {len(self.final_objects)}"

    def save_to_json(self, path):
        """简单的序列化保存 (跳过numpy数组)"""
        # 注意：这里需要处理 numpy 数组不能直接转 json 的问题
        # 实际使用中建议只保存关键元数据，或者用 pickle 保存完整对象
        data = {
            "image_name": self.image_name,
            "summary": self.summary(),
            "objects_count": len(self.final_objects)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)