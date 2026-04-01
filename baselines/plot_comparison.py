"""
baselines/plot_comparison.py

绘制主方法与三种基线方法的公平对比图。

输出：
1. 柱状图：四种方法在三个样本上的反演 RMSE
2. 平均 RMSE 对比图
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 数据准备
# ============================================================

# 三个样本的结果（单位：μm）
samples = ['sample_00900', 'sample_00532', 'sample_00608']
samples_short = ['00900', '00532', '00608']

# 四种方法的反演 RMSE（从运行结果中提取）
affine_rmse = [50.72, 51.10, 44.56]
poly_rmse = [52.48, 20.40, 25.19]
rbf_rmse = [35.08, 28.78, 26.74]
main_rmse = [21.63, 14.88, 17.60]

# 计算平均值
affine_avg = np.mean(affine_rmse)
poly_avg = np.mean(poly_rmse)
rbf_avg = np.mean(rbf_rmse)
main_avg = np.mean(main_rmse)

print("=" * 60)
print("平均反演 RMSE (μm)")
print("=" * 60)
print(f"Affine:          {affine_avg:.2f} μm")
print(f"Poly:            {poly_avg:.2f} μm")
print(f"Weighted RBF:    {rbf_avg:.2f} μm")
print(f"Main (FVCCalib): {main_avg:.2f} μm")
print("=" * 60)

# ============================================================
# 图1：三个样本的分组柱状图
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(samples_short))
width = 0.2

bars1 = ax.bar(x - 1.5*width, affine_rmse, width, label='Affine', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, poly_rmse, width, label='Poly', color='#f39c12', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, rbf_rmse, width, label='Weighted RBF', color='#3498db', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, main_rmse, width, label='Main (FVCCalib)', color='#2ecc71', alpha=0.8)

# 在柱子上标注数值
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)
autolabel(bars4)

ax.set_xlabel('样本', fontsize=12)
ax.set_ylabel('反演 RMSE (μm)', fontsize=12)
ax.set_title('主方法与基线方法的反演精度对比（三个样本）', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(samples_short)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# 保存图1
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "outputs", "figures")
os.makedirs(output_dir, exist_ok=True)

fig1_path = os.path.join(output_dir, "comparison_by_sample.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"\n图1已保存: {fig1_path}")
plt.close()

# ============================================================
# 图2：平均 RMSE 对比图
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Affine', 'Poly', 'Weighted RBF', 'Main\n(FVCCalib)']
avg_rmse = [affine_avg, poly_avg, rbf_avg, main_avg]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

bars = ax.bar(methods, avg_rmse, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 标注数值
for i, (bar, val) in enumerate(zip(bars, avg_rmse)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f'{val:.2f} μm',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold')

# 添加目标精度参考线
target_accuracy = 3.0
ax.axhline(y=target_accuracy, color='red', linestyle='--', linewidth=2, label=f'目标精度 {target_accuracy} μm')

ax.set_ylabel('平均反演 RMSE (μm)', fontsize=12)
ax.set_title('主方法与基线方法的平均反演精度对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 设置 y 轴范围，让差异更明显
ax.set_ylim(0, max(avg_rmse) * 1.2)

plt.tight_layout()

# 保存图2
fig2_path = os.path.join(output_dir, "comparison_average.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"图2已保存: {fig2_path}")
plt.close()

# ============================================================
# 图3：误差改善率（相对于 Affine）
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# 计算相对于 Affine 的改善率
improvement_poly = (affine_avg - poly_avg) / affine_avg * 100
improvement_rbf = (affine_avg - rbf_avg) / affine_avg * 100
improvement_main = (affine_avg - main_avg) / affine_avg * 100

methods_imp = ['Poly', 'Weighted RBF', 'Main (FVCCalib)']
improvements = [improvement_poly, improvement_rbf, improvement_main]
colors_imp = ['#f39c12', '#3498db', '#2ecc71']

bars = ax.bar(methods_imp, improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)

# 标注数值
for bar, val in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold')

ax.set_ylabel('相对于 Affine 的误差改善率 (%)', fontsize=12)
ax.set_title('各方法相对于 Affine 基线的精度改善', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(improvements) * 1.2)

plt.tight_layout()

# 保存图3
fig3_path = os.path.join(output_dir, "comparison_improvement.png")
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"图3已保存: {fig3_path}")
plt.close()

print("\n所有图表生成完成！")