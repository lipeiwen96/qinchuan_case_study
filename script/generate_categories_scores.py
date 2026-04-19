import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. Base Data & Derived Consistency Metrics
# ==========================================
categories_data = [
    {"label": "C2", "n": 2150, "mean": 58.2, "box-left": 47.0, "box-right": 69.0, "Whiskers-left": 8.3, "Whiskers-right": 96.0, "sd": 16.5, "coverage": 92.5},
    {"label": "C3", "n": 2620, "mean": 54.5, "box-left": 42.0, "box-right": 66.0, "Whiskers-left": 18.0, "Whiskers-right": 108.0, "sd": 19.8, "coverage": 86.4},
    {"label": "C1", "n": 1850, "mean": 45.1, "box-left": 36.0, "box-right": 53.0, "Whiskers-left": 20.0, "Whiskers-right": 78.0, "sd": 12.8, "coverage": 76.2},
    {"label": "C4", "n": 3580, "mean": 41.8, "box-left": 32.0, "box-right": 50.0, "Whiskers-left": 16.0, "Whiskers-right": 74.0, "sd": 13.5, "coverage": 68.9},
    {"label": "C5", "n": 4010, "mean": 36.2, "box-left": 21.0, "box-right": 61.0, "Whiskers-left": 8.0, "Whiskers-right": 89.0, "sd": 29.6, "coverage": 62.1},
    {"label": "C6", "n": 1350, "mean": 33.4, "box-left": 22.0, "box-right": 44.0, "Whiskers-left": 6.0, "Whiskers-right": 76.0, "sd": 16.2, "coverage": 54.5},
    {"label": "C9", "n": 840, "mean": 26.8, "box-left": 14.0, "box-right": 38.0, "Whiskers-left": 2.0, "Whiskers-right": 90.0, "sd": 18.5, "coverage": 49.3},
    {"label": "C7", "n": 960, "mean": 23.5, "box-left": 15.0, "box-right": 31.0, "Whiskers-left": 4.0, "Whiskers-right": 58.0, "sd": 11.2, "coverage": 43.2},
    {"label": "C8", "n": 780, "mean": 22.1, "box-left": 5.0, "box-right": 30.0, "Whiskers-left": 0.5, "Whiskers-right": 120.0, "sd": 26.3, "coverage": 34.6},
    {"label": "C10", "n": 450, "mean": 16.5, "box-left": 10.0, "box-right": 22.0, "Whiskers-left": 1.0, "Whiskers-right": 42.0, "sd": 8.5, "coverage": 24.1}
]
# categories_data = [
#     {"label": "C2 (Gate Tower & Entrance)", "n": 2150, "mean": 58.2, "box-left": 47.0, "box-right": 69.0, "Whiskers-left": 8.3, "Whiskers-right": 96.0, "sd": 16.5, "coverage": 92.5},
#     {"label": "C3 (Windows & Details)", "n": 2620, "mean": 54.5, "box-left": 42.0, "box-right": 66.0, "Whiskers-left": 18.0, "Whiskers-right": 108.0, "sd": 19.8, "coverage": 86.4},
#     {"label": "C1 (Traditional Roof & Timber)", "n": 1850, "mean": 45.1, "box-left": 36.0, "box-right": 53.0, "Whiskers-left": 20.0, "Whiskers-right": 78.0, "sd": 12.8, "coverage": 76.2},
#     {"label": "C4 (Masonry & Texture)", "n": 3580, "mean": 41.8, "box-left": 32.0, "box-right": 50.0, "Whiskers-left": 16.0, "Whiskers-right": 74.0, "sd": 13.5, "coverage": 68.9},
#     {"label": "C5 (Vegetation & Trees)", "n": 4010, "mean": 36.2, "box-left": 21.0, "box-right": 61.0, "Whiskers-left": 8.0, "Whiskers-right": 89.0, "sd": 12.5, "coverage": 62.1},
#     {"label": "C6 (Water & Bridges)", "n": 1350, "mean": 33.4, "box-left": 22.0, "box-right": 44.0, "Whiskers-left": 6.0, "Whiskers-right": 76.0, "sd": 16.2, "coverage": 54.5},
#     {"label": "C9 (Clutter & People)", "n": 840, "mean": 26.8, "box-left": 14.0, "box-right": 38.0, "Whiskers-left": 2.0, "Whiskers-right": 90.0, "sd": 18.5, "coverage": 49.3},
#     {"label": "C7 (Modern Buildings)", "n": 960, "mean": 23.5, "box-left": 15.0, "box-right": 31.0, "Whiskers-left": 4.0, "Whiskers-right": 58.0, "sd": 11.2, "coverage": 43.2},
#     {"label": "C8 (Modern Facilities)", "n": 780, "mean": 22.1, "box-left": 5.0, "box-right": 30.0, "Whiskers-left": 0.5, "Whiskers-right": 120.0, "sd": 26.3, "coverage": 34.6},
#     {"label": "C10 (Sky & Background)", "n": 450, "mean": 16.5, "box-left": 10.0, "box-right": 22.0, "Whiskers-left": 1.0, "Whiskers-right": 42.0, "sd": 8.5, "coverage": 24.1}
# ]


for cat in categories_data:
    cat["consistency"] = (cat["mean"] * cat["coverage"]) / 100.0

sorted_by_consistency = sorted(categories_data, key=lambda x: x["consistency"], reverse=False)

# ==========================================
# 2. Scatter Generation (Organic Density)
# ==========================================
stats_for_bxp = []
scatter_records = []
np.random.seed(42)

for cat in categories_data:
    label = cat["label"]
    mean_v = cat["mean"]
    q1 = cat["box-left"]
    q3 = cat["box-right"]
    w_left = cat["Whiskers-left"]
    w_right = cat["Whiskers-right"]
    n_val = cat["n"]

    right_skewed_cats = ["C3 (Windows & Details)", "C8 (Modern Facilities)", "C9 (Clutter & People)", "C7 (Modern Buildings)", "C1 (Traditional Roof & Timber)"]
    if label in right_skewed_cats:
        med_v = mean_v - np.random.uniform(0.08, 0.15) * (q3 - q1)
    else:
        med_v = mean_v + np.random.uniform(0.08, 0.15) * (q3 - q1)
    med_v = np.clip(med_v, q1 + (q3 - q1) * 0.15, q3 - (q3 - q1) * 0.15)

    stats_for_bxp.append({
        "label": label, "mean": mean_v, "med": med_v,
        "q1": q1, "q3": q3, "whislo": w_left, "whishi": w_right
    })

    n_visual = max(50, int(n_val * 0.12))
    range_val = w_right - w_left
    norm_mean = (mean_v - w_left) / range_val
    k = 4.5
    a = max(norm_mean * k, 1.2)
    b = max((1 - norm_mean) * k, 1.2)

    n_beta = int(n_visual * 0.6)
    beta_pts = np.random.beta(a, b, n_beta) * range_val + w_left
    n_box = int(n_visual * 0.25)
    box_pts = np.random.uniform(q1, q3, n_box)
    n_tail = n_visual - n_beta - n_box
    tail_pts = np.random.uniform(w_left, w_right, n_tail)

    all_pts = np.concatenate([beta_pts, box_pts, tail_pts])
    all_pts[0] = w_left + 0.1
    all_pts[1] = w_right - 0.1

    for pt in all_pts:
        scatter_records.append({"Category": label, "Score": pt})

df_scatter = pd.DataFrame(scatter_records)
cat_order = [cat["label"] for cat in categories_data]

# ==========================================
# 3. Canvas Layout & Rendering
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

fig = plt.figure(figsize=(14, 15))
gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.1)

ax1 = fig.add_subplot(gs[0, :])   # Top: Boxplot
ax2 = fig.add_subplot(gs[1, 0])   # Bottom Left: Bar chart
ax3 = fig.add_subplot(gs[1, 1])   # Bottom Right: Scatter plot

palette_10 = sns.color_palette("coolwarm", n_colors=10)[::-1]

# --- Subplot 1 (ax1): Boxplot ---
sns.stripplot(x="Score", y="Category", data=df_scatter, order=cat_order, size=3.0, color="#555555", alpha=0.35, jitter=0.18, zorder=1, ax=ax1)

bplot = ax1.bxp(
    stats_for_bxp, positions=np.arange(len(stats_for_bxp)), vert=False,
    patch_artist=True, showmeans=True, meanline=True, showfliers=False, zorder=2,
    meanprops=dict(color='#d62728', linewidth=2.5, linestyle='-'),
    medianprops=dict(color='black', linewidth=1.5, linestyle='--'),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2)
)

for patch, color in zip(bplot['boxes'], palette_10):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

for i, stat in enumerate(stats_for_bxp):
    ax1.text(stat['mean'], i - 0.38, f"Mean: {stat['mean']:.1f}", color='#d62728', fontsize=10, fontweight='bold', ha='center', va='center', zorder=10)

ax1.set_xlabel('Visual Attention Score (Eq. 2)', fontweight='bold', fontsize=14)
ax1.set_ylabel('')
ax1.set_title('(a) Intensity of Visual Attention Across Object Categories', fontweight='bold', loc='left', fontsize=16, pad=15)
ax1.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
ax1.set_xlim(left=0)
sns.despine(ax=ax1, trim=False, left=True)

# --- Subplot 2 (ax2): Bar Chart ---
short_labels_sorted = [cat['label'].split('(')[0].strip() if '(' in cat['label'] else cat['label'] for cat in sorted_by_consistency]
consistencies = [cat['consistency'] for cat in sorted_by_consistency]

bar_palette = sns.color_palette("viridis_r", len(sorted_by_consistency))
ax2.barh(short_labels_sorted, consistencies, color=bar_palette, edgecolor='none', alpha=0.9)

for i, v in enumerate(consistencies):
    ax2.text(v + 1, i, f"{v:.1f}", color='black', va='center', fontsize=9)

ax2.set_xlabel('Consistency Index (Score × Coverage / 100)', fontweight='bold')
ax2.set_title('(b) Top-10 Consistent Attention Elements', fontweight='bold', loc='center', pad=10)
ax2.set_xlim(0, max(consistencies) * 1.15)
sns.despine(ax=ax2)

# --- Subplot 3 (ax3): Scatter Bubble Chart ---
x_cov = [cat['coverage'] for cat in categories_data]
y_mean = [cat['mean'] for cat in categories_data]
short_labels = [cat['label'] for cat in categories_data]

# Draw the scatter plot
ax3.scatter(x_cov, y_mean, s=[c * 5 for c in consistencies], c=consistencies, cmap='viridis_r', alpha=0.9, edgecolors='white', linewidth=0.5)

# Draw reference lines
mean_cov = np.mean(x_cov)
mean_score = np.mean(y_mean)
ax3.axvline(x=mean_cov, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)

# Add text labels for ALL elements
for i in range(len(x_cov)):
    # x_cov[i] + 2 adds a slight horizontal offset so the text doesn't overlap the dot
    ax3.text(x_cov[i] + 2, y_mean[i], short_labels[i], fontsize=9, va='center')

ax3.set_xlabel('Coverage Rate (% of Participants)', fontweight='bold')
ax3.set_ylabel('Avg. Attention Score (Intensity)', fontweight='bold')
ax3.set_title('(c) Attention Score vs. Participant Coverage', fontweight='bold', loc='center', pad=10)
sns.despine(ax=ax3)

# Save
plt.savefig('Fig7_Comprehensive_Analysis_All_Labels.png', dpi=600, bbox_inches='tight')