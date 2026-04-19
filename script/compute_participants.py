import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D  # 用于创建自定义图例

# ==========================================
# 1. 数据读取与预处理
# ==========================================
# 读取您的Excel文件（请将文件名替换为您的实际路径）
file_path = '../data/experiment_data/志愿者信息.xlsx'

# 使用 read_excel 替代 read_csv 读取 .xlsx 文件
df = pd.read_excel(file_path)

# 【注意】请根据您的实际 Excel 表头修改以下变量名
col_id = '志愿者编号'          # 对应志愿者编号的列名
col_gender = '性别'            # 对应性别的列名 (例如内容为 'Male', 'Female' 或 '男', '女')
col_age = '年龄'               # 对应年龄的列名

# 剔除无效数据 (例如剔除 ID 为 20 或者是 '020' 的被试)
exclude_id = 20

# 新增一列用于标记数据状态
df['数据状态'] = 'Included'
df.loc[df[col_id] == exclude_id, '数据状态'] = 'Excluded'

# 分离有效数据和剔除数据
df_valid = df[df['数据状态'] == 'Included'].copy()
df_excluded = df[df['数据状态'] == 'Excluded'].copy()

# 获取X轴的固定顺序，确保箱线图和散点图的性别类别对齐
gender_order = df_valid[col_gender].unique()

# ==========================================
# 2. 统计数据计算与输出
# ==========================================
total_recruited = len(df)
valid_total = len(df_valid)

# 计算年龄统计 (基于有效数据)
age_min = df_valid[col_age].min()
age_max = df_valid[col_age].max()
age_mean = df_valid[col_age].mean()
age_sd = df_valid[col_age].std()

# 计算性别统计 (基于有效数据)
gender_counts = df_valid[col_gender].value_counts()

print("========== 计算结果 (用于替换论文文本) ==========")
print(f"招募总人数 (Total recruited): {total_recruited}")
print(f"有效总人数 (Valid datasets): {valid_total}")
print(f"年龄范围 (Age range): {age_min} - {age_max}")
print(f"年龄均值 (Mean): {age_mean:.1f}")
print(f"年龄标准差 (SD): {age_sd:.2f}")
print("性别分布:")
for gender, count in gender_counts.items():
    pct = (count / valid_total) * 100
    print(f" - {gender}: {count} ({pct:.1f}%)")
print("=================================================")

# ==========================================
# 3. 绘制 SCI 级别高质量图表
# ==========================================
# 全局字体和清晰度设置
plt.rcParams['font.family'] = 'Times New Roman'  # 学术常用字体
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300                 # 高分辨率用于预览

# 创建画布
fig, ax = plt.subplots(figsize=(6, 5))

# 定义高雅的学术配色 (蓝色调代表男性，暖色调代表女性)
sci_palette = ['#4A90E2', '#E94E77']

# 1. 绘制箱线图 (仅使用有效数据，避免异常值干扰统计分布)
sns.boxplot(
    x=col_gender, y=col_age, data=df_valid,
    order=gender_order,  # 固定X轴顺序
    width=0.4,
    palette=sci_palette,
    boxprops=dict(alpha=0.7, edgecolor='black', linewidth=1.2),
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    showfliers=False,
    ax=ax
)

# 2. 绘制散点图 - 采纳的数据 (灰色圆点)
sns.stripplot(
    x=col_gender, y=col_age, data=df_valid,
    order=gender_order,
    size=6, color="#404040", linewidth=0.5, edgecolor="white",
    alpha=0.7, jitter=0.15, ax=ax
)

# 3. 绘制散点图 - 剔除的数据 (红色十字 X 星)
if not df_excluded.empty:
    sns.stripplot(
        x=col_gender, y=col_age, data=df_excluded,
        order=gender_order,
        size=9, color="#E74C3C", marker="X", linewidth=0.5, edgecolor="white",
        alpha=0.9, jitter=0.15, ax=ax, zorder=10 # zorder置于顶层，防止被覆盖
    )

# 细节优化
ax.set_ylabel('Age (years)', fontweight='bold')
ax.set_xlabel('Gender', fontweight='bold')

# 移除上方和右方的边框
sns.despine(offset=10, trim=False)

# 4. 创建自定义图例 (Legend)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Included Data',
           markerfacecolor='#404040', markersize=8, markeredgecolor='white', alpha=0.7),
    Line2D([0], [0], marker='X', color='w', label='Excluded Data',
           markerfacecolor='#E74C3C', markersize=10, markeredgecolor='white')
]

# 将图例添加到图表中 (loc='best' 让其自动寻找不遮挡数据的位置，也可以改为 'upper right')
ax.legend(handles=legend_elements, loc='best', frameon=True, edgecolor='black', fontsize=11)

# 保存高分辨率图片
output_filename = 'Fig5_Age_Gender_Distribution.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=600, bbox_inches='tight')

print(f"\n✅ 图片已成功保存为: {output_filename}")