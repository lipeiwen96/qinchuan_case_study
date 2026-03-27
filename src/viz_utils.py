# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合脚本批量出图

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置 SCI 绘图风格
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'  # 英文使用无衬线字体
# 如果需要显示中文，需要另外配置中文字体，例如 SimHei
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

COLOR_PALETTE = sns.color_palette("viridis", as_cmap=False)  # 或 "deep", "husl"


def save_fig_sci(fig, output_path, dpi=300):
    """保存符合发表标准的图片"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 图片已保存: {output_path}")
    plt.close(fig)


def plot_volunteer_demographics(volunteers_df: pd.DataFrame, output_dir):
    """绘制高质量的年龄和性别分布图"""
    # 1. 年龄分布 (Violin Plot + Strip Plot)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x="gender", y="age", data=volunteers_df, palette="muted", inner=None, ax=ax)
    sns.stripplot(x="gender", y="age", data=volunteers_df, color="black", alpha=0.5, jitter=True, ax=ax)

    ax.set_title("Volunteer Age Distribution by Gender", fontweight='bold', fontsize=14)
    ax.set_ylabel("Age (Years)", fontsize=12)
    ax.set_xlabel("Gender", fontsize=12)
    sns.despine(left=True)
    save_fig_sci(fig, os.path.join(output_dir, "demographics_age_gender_violin.png"))

    # 2. 性别比例 (Donut Chart - Matplotlib)
    gender_counts = volunteers_df['gender'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                                      startangle=90, colors=sns.color_palette("pastel"),
                                      wedgeprops=dict(width=0.5))  # width < 1 创建甜甜圈
    ax.set_title("Gender Composition", fontweight='bold')
    plt.setp(autotexts, size=12, weight="bold", color="white")
    save_fig_sci(fig, os.path.join(output_dir, "demographics_gender_donut.png"))


def plot_gaze_semantic_aggregation(aggregated_data: pd.DataFrame, output_dir):
    """
    绘制不同热力等级下的语义占比堆叠条形图
    aggregated_data 结构: Columns=['HeatLevel', 'SemanticClass', 'AverageRatio']
    """
    if aggregated_data.empty:
        print("⚠️ 没有足够的数据绘制聚合图。")
        return

    # 数据透视，行为热力等级，列为语义类别，值为比例
    pivot_df = aggregated_data.pivot(index='HeatLevel', columns='SemanticClass', values='AverageRatio')
    # 填充NaN为0
    pivot_df = pivot_df.fillna(0)
    # 确保热力等级顺序
    heatmap_order = ['High Attention', 'Medium Attention', 'Low Attention']
    pivot_df = pivot_df.reindex(heatmap_order)

    # 选择主要的语义类别进行显示 (避免图例过多)，其他的归为 "Others"
    top_n = 8
    top_classes = pivot_df.sum().nlargest(top_n).index.tolist()

    final_df = pivot_df[top_classes].copy()
    other_classes = pivot_df.columns.difference(top_classes)
    if not other_classes.empty:
        final_df['Others'] = pivot_df[other_classes].sum(axis=1)

    # 归一化到 100%
    final_df = final_df.div(final_df.sum(axis=1), axis=0) * 100

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    # 使用 distinct 的颜色盘
    colors = sns.color_palette("tab20", n_colors=len(final_df.columns))

    final_df.plot(kind='barh', stacked=True, color=colors, ax=ax, edgecolor='white')

    ax.set_title("Semantic Composition Across Gaze Attention Levels", fontweight='bold', fontsize=16)
    ax.set_xlabel("Percentage Composition (%)", fontsize=14)
    ax.set_ylabel("Gaze Heatmap Intensity", fontsize=14)
    plt.legend(title="Semantic Category", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 在条形图上添加数值标签
    for c in ax.containers:
        # 自定义标签文本，只显示大于 3% 的
        labels = [f'{v.get_width():.1f}%' if v.get_width() > 3 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=10, color='white', weight='bold')

    save_fig_sci(fig, os.path.join(output_dir, "aggregated_gaze_semantic_stacked_bar.png"))