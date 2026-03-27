# -*- coding: utf-8 -*-
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import numpy as np


# ===========================
# 街景图像元数据结构 (用于缓存)
# ===========================
@dataclass
class SemanticClassMeta:
    """单个语义类别在图像中的统计信息"""
    class_id: int
    name: str
    pixel_count: int
    ratio: float # 占全图比例 (0.0 - 1.0)


@dataclass
class StreetViewMetadata:
    """一张街景图的完整元数据"""
    image_name: str
    width: int
    height: int
    # 存储每个类别的统计信息。key为class_id的字符串形式
    semantic_stats: Dict[str, SemanticClassMeta] = field(default_factory=dict)
    # 分割掩码的文件名 (不存巨大的numpy数组在JSON里，而是存引用)
    mask_filename: str = ""

    def to_dict(self):
        """序列化为字典"""
        data = asdict(self)
        # 将 semantic_stats 中的值也转换为字典
        data['semantic_stats'] = {k: asdict(v) for k, v in self.semantic_stats.items()}
        return data

    @classmethod
    def from_dict(cls, data):
        """从字典反序列化"""
        stats_data = data.get('semantic_stats', {})
        reconstructed_stats = {}
        for k, v in stats_data.items():
            reconstructed_stats[k] = SemanticClassMeta(**v)
        data['semantic_stats'] = reconstructed_stats
        return cls(**data)

    def save_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ===========================
# 志愿者实验数据结构
# ===========================
@dataclass
class VolunteerInfo:
    """志愿者基本信息"""
    vol_id: str  # 格式化后的ID，如 "001"
    age: int
    gender: str


@dataclass
class SingleTrialData:
    """单次实验数据 (一张图的反馈)"""
    streetview_name: str # e.g., "QINCHUAN-62.JPG"
    serial_number: int   # e.g., 62
    likert_scale: int
    heatmap_path: Optional[str] = None # 热力图绝对路径
    # 预留接口
    eeg_data: Optional[np.ndarray] = None
    erp_data: Optional[np.ndarray] = None
    # 分析结果：不同热力等级下的语义占比
    gaze_semantic_intersection: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class Volunteer:
    """包含志愿者所有信息的容器"""
    info: VolunteerInfo
    # key为 streetview_name
    trials: Dict[str, SingleTrialData] = field(default_factory=dict)

    def add_trial(self, trial: SingleTrialData):
        self.trials[trial.streetview_name] = trial