import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 数据配置区：您可以直接修改这里的数值
# ==========================================

# 1. 相关性矩阵数据 (Correlation Matrix Data)
# 行: 5个主观感知维度; 列: C1-C10 类别 (需替换为您实际算出的 Pearson r 值)
corr_data = [
    {"Perception": "Satisfaction", "C1": 0.52, "C2": 0.48, "C3": 0.35, "C4": 0.22, "C5": 0.15, "C6": 0.45, "C7": -0.25,
     "C8": -0.42, "C9": -0.15, "C10": 0.05},
    {"Perception": "Authenticity", "C1": 0.61, "C2": 0.42, "C3": 0.55, "C4": 0.38, "C5": 0.08, "C6": 0.31, "C7": -0.41,
     "C8": -0.55, "C9": -0.22, "C10": 0.02},
    {"Perception": "Spatial Coherence", "C1": 0.58, "C2": 0.25, "C3": 0.18, "C4": 0.12, "C5": -0.05, "C6": 0.10,
     "C7": -0.35, "C8": -0.48, "C9": -0.28, "C10": -0.05},
    {"Perception": "Walkability", "C1": 0.15, "C2": 0.45, "C3": 0.12, "C4": 0.51, "C5": 0.18, "C6": 0.22, "C7": -0.12,
     "C8": -0.25, "C9": -0.35, "C10": 0.08},
    {"Perception": "Visual Richness", "C1": 0.42, "C2": 0.58, "C3": 0.65, "C4": 0.35, "C5": 0.25, "C6": 0.38,
     "C7": -0.38, "C8": -0.62, "C9": -0.18, "C10": -0.10}
]

# 2. 注意力差值数据 (Drivers Data: High-rated minus Low-rated scenes)
# 每个感知维度下，10个类别的注意力得分差值
drivers_data = {
    "Satisfaction": [
        {"label": "C1: Trad. Roof", "diff": 7.5}, {"label": "C6: Water/Bridge", "diff": 5.2},
        {"label": "C2: Gate/Entrance", "diff": 4.8}, {"label": "C3: Windows/Details", "diff": 3.5},
        {"label": "C5: Vegetation", "diff": 1.2}, {"label": "C10: Sky/Bg", "diff": -0.5},
        {"label": "C9: Clutter/People", "diff": -1.8}, {"label": "C4: Masonry", "diff": -2.2},
        {"label": "C7: Modern Bldgs", "diff": -4.5}, {"label": "C8: Modern Facil.", "diff": -6.8}
    ],
    "Authenticity": [
        {"label": "C1: Trad. Roof", "diff": 8.5}, {"label": "C3: Windows/Details", "diff": 6.2},
        {"label": "C2: Gate/Entrance", "diff": 4.5}, {"label": "C4: Masonry", "diff": 3.8},
        {"label": "C6: Water/Bridge", "diff": 2.5}, {"label": "C10: Sky/Bg", "diff": 0.2},
        {"label": "C5: Vegetation", "diff": -0.8}, {"label": "C9: Clutter/People", "diff": -2.5},
        {"label": "C7: Modern Bldgs", "diff": -5.2}, {"label": "C8: Modern Facil.", "diff": -7.5}
    ],
    "Spatial Coherence": [
        {"label": "C1: Trad. Roof", "diff": 9.2}, {"label": "C2: Gate/Entrance", "diff": 2.5},
        {"label": "C4: Masonry", "diff": 1.5}, {"label": "C6: Water/Bridge", "diff": 1.0},
        {"label": "C10: Sky/Bg", "diff": -0.5}, {"label": "C5: Vegetation", "diff": -1.2},
        {"label": "C3: Windows/Details", "diff": -1.8}, {"label": "C9: Clutter/People", "diff": -3.5},
        {"label": "C7: Modern Bldgs", "diff": -5.8}, {"label": "C8: Modern Facil.", "diff": -6.5}
    ],
    "Walkability": [
        {"label": "C4: Masonry", "diff": 7.8}, {"label": "C2: Gate/Entrance", "diff": 6.5},
        {"label": "C6: Water/Bridge", "diff": 3.2}, {"label": "C5: Vegetation", "diff": 2.5},
        {"label": "C10: Sky/Bg", "diff": 0.8}, {"label": "C3: Windows/Details", "diff": -1.2},
        {"label": "C9: Clutter/People", "diff": -2.8}, {"label": "C1: Trad. Roof", "diff": -3.5},
        {"label": "C7: Modern Bldgs", "diff": -4.2}, {"label": "C8: Modern Facil.", "diff": -5.5}
    ],
    "Visual Richness": [
        {"label": "C3: Windows/Details", "diff": 8.2}, {"label": "C2: Gate/Entrance", "diff": 6.8},
        {"label": "C1: Trad. Roof", "diff": 5.5}, {"label": "C4: Masonry", "diff": 3.2},
        {"label": "C6: Water/Bridge", "diff": 2.5}, {"label": "C5: Vegetation", "diff": 1.5},
        {"label": "C10: Sky/Bg", "diff": -1.2}, {"label": "C9: Clutter/People", "diff": -3.8},
        {"label": "C7: Modern Bldgs", "diff": -5.5}, {"label": "C8: Modern Facil.", "diff": -8.5}
    ]
}

# ==========================================
# 绘图逻辑区
# ==========================================

# 全局字体和样式设置 (SCI 标准风格)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Integrated Analysis: Correlation & Drivers of Subjective Perception', fontsize=16, y=0.98)

# 创建 2x3 的网格布局
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# --- (A) 绘制热力图 ---
ax_heatmap = fig.add_subplot(gs[0, 0])
df_corr = pd.DataFrame(corr_data).set_index("Perception")
sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-0.8, vmax=0.8, ax=ax_heatmap, cbar_kws={'shrink': 0.8},
            linewidths=.5, linecolor='white')
ax_heatmap.set_title('(A) Correlation Matrix:\nAttention vs. Perception', pad=15)
ax_heatmap.set_ylabel('')
ax_heatmap.tick_params(axis='x', rotation=45)


# --- (B) - (F) 绘制发散条形图的通用函数 ---
def plot_diverging_bar(ax, dimension_name, panel_letter):
    # 提取并排序数据 (按 diff 从大到小)
    df = pd.DataFrame(drivers_data[dimension_name]).sort_values(by="diff", ascending=False)

    # 定义颜色: 绿色(正向驱动), 红色(负向干扰)
    colors = ['#55C667' if val > 0 else '#F05B56' for val in df['diff']]

    bars = ax.barh(df['label'], df['diff'], color=colors, height=0.6)

    ax.axvline(0, color='black', linewidth=0.8)  # 添加 0 刻度线
    ax.set_title(f'({panel_letter}) Drivers of "{dimension_name}"')
    ax.set_xlabel('Attention Diff. (High - Low Rated)')
    ax.invert_yaxis()  # 让正值最大的在最上面

    # 隐藏边框，保留底边
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.2 if width > 0 else width - 0.2
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:+.1f}',
                va='center', ha=ha, fontsize=8, color=bar.get_facecolor())


# 分配各个子图的位置
panels = [
    (gs[0, 1], "Satisfaction", "B"),
    (gs[0, 2], "Authenticity", "C"),
    (gs[1, 0], "Spatial Coherence", "D"),
    (gs[1, 1], "Walkability", "E"),
    (gs[1, 2], "Visual Richness", "F")
]

for pos, dim, letter in panels:
    ax = fig.add_subplot(pos)
    plot_diverging_bar(ax, dim, letter)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig10.png', dpi=600, bbox_inches='tight')