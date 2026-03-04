"""
MUST望远镜 FVC 可视化模块
包含四个独立可视化：
  1. 光斑图像可视化（仿真光斑 + 3D强度图）
  2. 质心定位过程可视化（拟合前后对比 + 残差）
  3. 标定板标定点可视化（格网分布 + 标定残差）
  4. 坐标变换可视化（像素→焦面映射 + 变换精度）

运行方式：
    python visualization.py              # 生成全部四个可视化
    python visualization.py --part 1     # 仅生成光斑可视化
    python visualization.py --part 2     # 仅生成质心定位可视化
    python visualization.py --part 3     # 仅生成标定点可视化
    python visualization.py --part 4     # 仅生成坐标变换可视化
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

# ── 中文字体配置 ─────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ─────────────────────────────────────────────────────────────

# ── 导入项目模块 ──────────────────────────────────────────────
from config import (
    PIXEL_SIZE_UM, DEMAGNIFICATION, FOCAL_PLANE_SCALE_UM_PX,
    SPOT_SIGMA_PX, SPOT_PEAK_COUNTS, BACKGROUND_COUNTS,
    TARGET_ACCURACY_UM, RANDOM_SEED,
    REFERENCE_GRID_SPACING_MM, REFERENCE_GRID_ORIGIN_MM, OUTPUT_DIR
)
from spot_generator import generate_gaussian_spot
from gaussian_detector import fit_gaussian
from coordinate_transform import FVCCalibrator

# ── 输出目录 ─────────────────────────────────────────────────
VIZ_DIR = os.path.join(OUTPUT_DIR, "figures", "visualization")
os.makedirs(VIZ_DIR, exist_ok=True)

# ── 与 main_pipeline.py 保持一致的常量 ────────────────────────
REF_GRID_SIDE = 5
DISTORTION_DEGREE = 2


# ============================================================
# 共用工具函数
# ============================================================

def build_reference_grid(n_side=REF_GRID_SIDE,
                         spacing_mm=REFERENCE_GRID_SPACING_MM,
                         origin_mm=REFERENCE_GRID_ORIGIN_MM):
    ref_focal = []
    ox = origin_mm[0] * 1000
    oy = origin_mm[1] * 1000
    spacing_um = spacing_mm * 1000
    for i in range(n_side):
        for j in range(n_side):
            ref_focal.append([ox + i * spacing_um, oy + j * spacing_um])
    return np.array(ref_focal, dtype=float)


def focal_to_pixel(focal_um, scale=None, rotation_deg=0.5,
                   offset_px=(3000.0, 2500.0)):
    if scale is None:
        scale = 1.0 / FOCAL_PLANE_SCALE_UM_PX
    angle = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    focal = np.atleast_2d(focal_um).astype(float)
    return (scale * (R @ focal.T).T) + np.array(offset_px)


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))


def add_colorbar(fig, ax, im, label=''):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=9)
    return cb


# ============================================================
# 可视化1：光斑图像
# ============================================================

def viz_spot(rng, save=True):
    """光斑图像可视化：2D灰度图 + 强度剖面 + 3D强度曲面"""
    print("\n[1/4] 生成光斑可视化...")

    patch_size = 60
    cx = patch_size / 2 + 2.3   # 故意加亚像素偏移，更真实
    cy = patch_size / 2 - 1.7

    spot = generate_gaussian_spot(cx, cy, image_size=patch_size, rng=rng)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle('光纤端面光斑仿真图像', fontsize=14, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── 子图1：2D灰度图 ──
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(spot, cmap='hot', origin='upper', aspect='equal')
    add_colorbar(fig, ax1, im1, 'counts')
    # 标注真实质心位置
    ax1.plot(cx, cy, '+', color='cyan', markersize=14, markeredgewidth=2,
             label=f'真实质心 ({cx:.1f}, {cy:.1f})')
    # 标注1sigma和2sigma椭圆
    for nsig, ls in [(1, '-'), (2, '--')]:
        ell = Ellipse((cx, cy), width=2*nsig*SPOT_SIGMA_PX, height=2*nsig*SPOT_SIGMA_PX,
                      edgecolor='lime', facecolor='none', linestyle=ls, linewidth=1.2,
                      label=f'{nsig}σ = {nsig*SPOT_SIGMA_PX:.1f} px')
        ax1.add_patch(ell)
    ax1.legend(fontsize=7.5, loc='upper right')
    ax1.set_title('2D光斑灰度图（热度图）', fontsize=10)
    ax1.set_xlabel('像素 X')
    ax1.set_ylabel('像素 Y')

    # 像素尺寸标注
    ax1.annotate('', xy=(5, patch_size-5), xytext=(15, patch_size-5),
                 arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax1.text(10, patch_size-8, '10 px', ha='center', va='top',
             color='white', fontsize=7)

    # ── 子图2：X/Y方向强度剖面 ──
    ax2 = fig.add_subplot(gs[1])
    x_profile = spot[int(round(cy)), :]
    y_profile = spot[:, int(round(cx))]
    px_range = np.arange(patch_size)

    ax2.plot(px_range, x_profile, 'b-o', markersize=3, linewidth=1.5,
             label='X方向剖面')
    ax2.plot(px_range, y_profile, 'r-s', markersize=3, linewidth=1.5,
             label='Y方向剖面')

    # 拟合高斯曲线叠加
    x_fine = np.linspace(0, patch_size-1, 300)
    peak = SPOT_PEAK_COUNTS
    bg = BACKGROUND_COUNTS
    gauss_fit = bg + peak * np.exp(-0.5*((x_fine - cx)/SPOT_SIGMA_PX)**2)
    ax2.plot(x_fine, gauss_fit, 'g--', linewidth=1.5, label='理论高斯曲线', alpha=0.8)

    ax2.axvline(cx, color='cyan', linestyle=':', linewidth=1.2, label=f'质心 x={cx:.1f}')
    ax2.axhline(bg, color='gray', linestyle=':', linewidth=1, alpha=0.6, label=f'背景={bg}')
    ax2.set_xlabel('像素坐标')
    ax2.set_ylabel('强度 (counts)')
    ax2.set_title('强度剖面（X/Y方向）', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # SNR标注
    noise_std = np.std(spot[:5, :5])  # 角落估算背景噪声
    snr = (peak - bg) / noise_std if noise_std > 0 else 0
    ax2.text(0.02, 0.97, f'SNR ≈ {snr:.0f}', transform=ax2.transAxes,
             va='top', fontsize=9, color='green',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── 子图3：3D强度曲面 ──
    ax3 = fig.add_subplot(gs[2], projection='3d')
    X3, Y3 = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    # 降采样加速渲染
    step = 2
    surf = ax3.plot_surface(X3[::step, ::step], Y3[::step, ::step],
                            spot[::step, ::step],
                            cmap='hot', alpha=0.85, linewidth=0, antialiased=True)
    ax3.set_xlabel('X (px)', fontsize=8)
    ax3.set_ylabel('Y (px)', fontsize=8)
    ax3.set_zlabel('强度', fontsize=8)
    ax3.set_title('3D强度曲面', fontsize=10)
    ax3.view_init(elev=30, azim=-60)
    fig.colorbar(surf, ax=ax3, shrink=0.4, aspect=8, label='counts')

    # 物理尺寸参数注释框
    info = (f'传感器像素: {PIXEL_SIZE_UM} μm\n'
            f'缩放比: {DEMAGNIFICATION}×\n'
            f'焦面尺度: {FOCAL_PLANE_SCALE_UM_PX:.1f} μm/px\n'
            f'光斑σ: {SPOT_SIGMA_PX} px = {SPOT_SIGMA_PX*PIXEL_SIZE_UM:.2f} μm\n'
            f'峰值: {SPOT_PEAK_COUNTS} counts\n'
            f'背景: {BACKGROUND_COUNTS} counts')
    fig.text(0.01, 0.01, info, fontsize=7.5, va='bottom',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "1_spot_image.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {path}")
    plt.close()
    return spot, cx, cy


# ============================================================
# 可视化2：质心定位过程
# ============================================================

def viz_centroid(spot, true_cx, true_cy, rng, save=True):
    """质心定位过程可视化：原始图像 → 高斯拟合 → 残差 → 收敛曲线"""
    print("\n[2/4] 生成质心定位可视化...")

    result = fit_gaussian(spot, sigma_init=SPOT_SIGMA_PX)
    fit_cx = result['x0']
    fit_cy = result['y0']
    patch_size = spot.shape[0]

    # 重建拟合高斯图像
    x_arr = np.arange(patch_size, dtype=float)
    y_arr = np.arange(patch_size, dtype=float)
    X, Y = np.meshgrid(x_arr, y_arr)
    fit_img = (BACKGROUND_COUNTS +
               result['amplitude'] * np.exp(
                -0.5 * (((X - fit_cx) / SPOT_SIGMA_PX) ** 2 +
                        ((Y - fit_cy) / SPOT_SIGMA_PX) ** 2)))

    residual = spot.astype(float) - fit_img

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('高斯拟合质心定位过程', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    vmin, vmax = spot.min(), spot.max()
    cmap_main = 'hot'
    cmap_res = 'RdBu_r'

    # ── 子图1：原始光斑 ──
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(spot, cmap=cmap_main, origin='upper', vmin=vmin, vmax=vmax)
    add_colorbar(fig, ax1, im1, 'counts')
    ax1.plot(true_cx, true_cy, '+', color='cyan', markersize=16, markeredgewidth=2.5,
             label=f'真实质心')
    ax1.set_title('原始光斑图像', fontsize=10)
    ax1.set_xlabel('像素 X')
    ax1.set_ylabel('像素 Y')
    ax1.legend(fontsize=8, loc='upper right')

    # ── 子图2：拟合结果 ──
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(fit_img, cmap=cmap_main, origin='upper', vmin=vmin, vmax=vmax)
    add_colorbar(fig, ax2, im2, 'counts')
    ax2.plot(fit_cx, fit_cy, 'x', color='yellow', markersize=16, markeredgewidth=2.5,
             label=f'拟合质心')
    ax2.plot(true_cx, true_cy, '+', color='cyan', markersize=12, markeredgewidth=1.5,
             alpha=0.7, label='真实质心')
    # 用箭头连接两个质心
    ax2.annotate('', xy=(fit_cx, fit_cy), xytext=(true_cx, true_cy),
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax2.set_title('高斯拟合重建图像', fontsize=10)
    ax2.set_xlabel('像素 X')
    ax2.legend(fontsize=8, loc='upper right')

    err_px = np.sqrt((fit_cx - true_cx)**2 + (fit_cy - true_cy)**2)
    err_um = err_px * FOCAL_PLANE_SCALE_UM_PX
    info2 = (f'拟合质心: ({fit_cx:.4f}, {fit_cy:.4f})\n'
             f'真实质心: ({true_cx:.4f}, {true_cy:.4f})\n'
             f'误差: {err_px:.5f} px = {err_um:.3f} μm\n'
             f'SNR: {result["snr"]:.1f}')
    ax2.text(0.02, 0.02, info2, transform=ax2.transAxes, va='bottom', fontsize=7.5,
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.85), color='white')

    # ── 子图3：残差图 ──
    ax3 = fig.add_subplot(gs[2])
    res_max = max(abs(residual.min()), abs(residual.max()))
    im3 = ax3.imshow(residual, cmap=cmap_res, origin='upper',
                     vmin=-res_max, vmax=res_max)
    add_colorbar(fig, ax3, im3, 'residual (counts)')
    ax3.plot(fit_cx, fit_cy, 'x', color='black', markersize=10, markeredgewidth=2)
    ax3.set_title('残差图（原始 - 拟合）', fontsize=10)
    ax3.set_xlabel('像素 X')

    rms_res = float(np.std(residual))
    ax3.text(0.02, 0.97, f'残差 RMS = {rms_res:.1f} counts',
             transform=ax3.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ── 子图4：多次拟合精度统计（直方图） ──
    ax4 = fig.add_subplot(gs[3])
    n_repeat = 300
    errs_px = []
    for _ in range(n_repeat):
        s = generate_gaussian_spot(true_cx, true_cy, image_size=patch_size, rng=rng)
        r = fit_gaussian(s, sigma_init=SPOT_SIGMA_PX)
        if r['success']:
            errs_px.append(np.sqrt((r['x0']-true_cx)**2 + (r['y0']-true_cy)**2))

    errs_um = np.array(errs_px) * FOCAL_PLANE_SCALE_UM_PX
    ax4.hist(errs_um, bins=35, color='steelblue', edgecolor='white',
             alpha=0.85, density=True)
    rms_um = float(np.std(errs_um))
    ax4.axvline(rms_um, color='red', linestyle='--', linewidth=1.8,
                label=f'RMS={rms_um:.3f}μm')
    ax4.axvline(TARGET_ACCURACY_UM, color='green', linestyle='-', linewidth=1.5,
                label=f'目标={TARGET_ACCURACY_UM}μm')
    ax4.set_xlabel('质心误差 (μm)')
    ax4.set_ylabel('概率密度')
    ax4.set_title(f'质心定位误差分布\n(N={n_repeat}次重复实验)', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "2_centroid_fitting.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {path}")
    plt.close()


# ============================================================
# 可视化3：标定板标定点
# ============================================================

def viz_calibration(rng, save=True):
    """标定板可视化：格网分布 + 标定前后残差 + 畸变场"""
    print("\n[3/4] 生成标定点可视化...")

    ref_focal_um = build_reference_grid()
    ref_px_ideal = focal_to_pixel(ref_focal_um)
    ref_px_detected = ref_px_ideal + rng.normal(0, 0.01, ref_px_ideal.shape)

    calibrator = FVCCalibrator(poly_degree=DISTORTION_DEGREE)
    calibrator.calibrate(ref_px_detected, ref_focal_um, verbose=False)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('标定板基准光纤分布与标定过程', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── 子图1：像素坐标系中的格网分布 ──
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(ref_px_detected[:, 0], ref_px_detected[:, 1],
                s=80, c='royalblue', zorder=5, label='检测到的基准点')
    ax1.scatter(ref_px_ideal[:, 0], ref_px_ideal[:, 1],
                s=30, c='red', marker='+', zorder=6, linewidths=1.5,
                label='理想格网位置')

    # 绘制格网连线
    n = REF_GRID_SIDE
    for i in range(n):
        row_idx = [i*n + j for j in range(n)]
        ax1.plot(ref_px_detected[row_idx, 0], ref_px_detected[row_idx, 1],
                 'b-', alpha=0.3, linewidth=0.8)
    for j in range(n):
        col_idx = [i*n + j for i in range(n)]
        ax1.plot(ref_px_detected[col_idx, 0], ref_px_detected[col_idx, 1],
                 'b-', alpha=0.3, linewidth=0.8)

    # 标注序号
    for k, (px, py) in enumerate(ref_px_detected):
        ax1.annotate(str(k+1), (px, py), textcoords='offset points',
                     xytext=(5, 5), fontsize=6.5, color='darkblue')

    ax1.set_xlabel('像素 X')
    ax1.set_ylabel('像素 Y')
    ax1.set_title(f'像素坐标系中的基准格网\n({REF_GRID_SIDE}×{REF_GRID_SIDE}={REF_GRID_SIDE**2}个基准点)', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 传感器范围示意
    ax1.set_xlim([ref_px_detected[:, 0].min() - 100, ref_px_detected[:, 0].max() + 100])
    ax1.set_ylim([ref_px_detected[:, 1].min() - 100, ref_px_detected[:, 1].max() + 100])

    # ── 子图2：焦面坐标系中的分布与标定残差矢量 ──
    ax2 = fig.add_subplot(gs[1])

    # 仿射残差（标定前）
    from coordinate_transform import fit_affine, apply_affine
    A, t = fit_affine(ref_px_detected, ref_focal_um)
    affine_pred = apply_affine(ref_px_detected, A, t)
    affine_err = ref_focal_um - affine_pred

    # 最终残差（标定后）
    final_pred = calibrator.transform(ref_px_detected)
    final_err = ref_focal_um - final_pred

    # 画焦面格网
    ref_x_mm = ref_focal_um[:, 0] / 1000
    ref_y_mm = ref_focal_um[:, 1] / 1000
    ax2.scatter(ref_x_mm, ref_y_mm, s=60, c='gray', zorder=3, alpha=0.5,
                label='焦面真值位置')

    # 仿射残差矢量（放大显示）
    scale_vis = 800
    q1 = ax2.quiver(ref_x_mm, ref_y_mm,
                    affine_err[:, 0] * scale_vis / 1000,
                    affine_err[:, 1] * scale_vis / 1000,
                    color='orange', alpha=0.7, scale=1, scale_units='xy',
                    width=0.005, label=f'仿射残差(×{scale_vis})')

    q2 = ax2.quiver(ref_x_mm, ref_y_mm,
                    final_err[:, 0] * scale_vis / 1000,
                    final_err[:, 1] * scale_vis / 1000,
                    color='blue', alpha=0.9, scale=1, scale_units='xy',
                    width=0.003, label=f'最终残差(×{scale_vis})')

    ax2.set_xlabel('焦面 X (mm)')
    ax2.set_ylabel('焦面 Y (mm)')
    ax2.set_title('焦面坐标系中的标定残差矢量\n（残差已放大显示）', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    rms_affine = float(np.sqrt(np.mean(affine_err**2)))
    rms_final = float(np.sqrt(np.mean(final_err**2)))
    info = (f'仿射残差 RMS: {rms_affine:.3f} μm\n'
            f'最终残差 RMS: {rms_final:.3f} μm\n'
            f'畸变阶数: {DISTORTION_DEGREE}阶多项式\n'
            f'参数比: {REF_GRID_SIDE**2 * 2 / 6:.0f}:1')
    ax2.text(0.02, 0.02, info, transform=ax2.transAxes, va='bottom', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ── 子图3：标定前后误差对比柱状图 ──
    ax3 = fig.add_subplot(gs[2])

    err_affine_per_pt = np.sqrt(affine_err[:, 0]**2 + affine_err[:, 1]**2)
    err_final_per_pt = np.sqrt(final_err[:, 0]**2 + final_err[:, 1]**2)

    pts = np.arange(1, REF_GRID_SIDE**2 + 1)
    width = 0.35
    bars1 = ax3.bar(pts - width/2, err_affine_per_pt, width,
                    label=f'仿射变换 (RMS={rms_affine:.3f}μm)',
                    color='orange', alpha=0.8, edgecolor='white')
    bars2 = ax3.bar(pts + width/2, err_final_per_pt, width,
                    label=f'含畸变校正 (RMS={rms_final:.3f}μm)',
                    color='steelblue', alpha=0.8, edgecolor='white')

    ax3.axhline(TARGET_ACCURACY_UM, color='red', linestyle='--', linewidth=1.5,
                label=f'目标精度 {TARGET_ACCURACY_UM}μm')
    ax3.set_xlabel('基准点编号')
    ax3.set_ylabel('定位误差 (μm)')
    ax3.set_title('各基准点标定前后误差对比', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(pts[::2])

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "3_calibration_points.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {path}")
    plt.close()

    return calibrator, ref_focal_um


# ============================================================
# 可视化4：坐标变换
# ============================================================

def viz_coordinate_transform(calibrator, ref_focal_um, rng, save=True):
    """坐标变换可视化：映射关系 + 变换精度 + 误差分布"""
    print("\n[4/4] 生成坐标变换可视化...")

    # 生成待测光纤数据
    focal_min = ref_focal_um.min(axis=0)
    focal_max = ref_focal_um.max(axis=0)
    margin = 0.10
    focal_range = focal_max - focal_min
    target_low = focal_min + margin * focal_range
    target_high = focal_max - margin * focal_range

    n_targets = 80
    target_focal_true = rng.uniform(target_low, target_high, (n_targets, 2))
    target_px_true = focal_to_pixel(target_focal_true)

    # 与主流程一致：通过生成光斑→高斯拟合获得检测坐标
    patch_size = 50
    target_px_detected = []
    for px, py in target_px_true:
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        patch = generate_gaussian_spot(true_cx, true_cy,
                                       image_size=patch_size, rng=rng)
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX)
        if result['success']:
            offset_x = round(px) - patch_size // 2
            offset_y = round(py) - patch_size // 2
            target_px_detected.append([result['x0'] + offset_x,
                                       result['y0'] + offset_y])
        else:
            target_px_detected.append([px, py])

    target_px_detected = np.array(target_px_detected)
    target_focal_pred = calibrator.transform(target_px_detected)

    err = target_focal_true - target_focal_pred
    err_r = np.sqrt(err[:, 0]**2 + err[:, 1]**2)

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle('像素坐标 → 焦面坐标变换过程与精度', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45)

    # ── 子图1：坐标映射示意图（像素坐标系） ──
    ax1 = fig.add_subplot(gs[0])
    ref_px = focal_to_pixel(ref_focal_um)

    ax1.scatter(ref_px[:, 0], ref_px[:, 1], s=100, c='red',
                marker='D', zorder=5, label='基准光纤（像素坐标）', alpha=0.8)
    ax1.scatter(target_px_detected[:, 0], target_px_detected[:, 1], s=30,
                c='steelblue', zorder=4, alpha=0.7, label='待测光纤（像素坐标）')

    # 绘制变换方向箭头示意（选几个点）
    sample_idx = [0, 12, 24, n_targets//2]
    for idx in sample_idx[:3]:
        px_x, px_y = target_px_detected[idx]
        ax1.annotate('', xy=(px_x + 60, px_y - 60),
                     xytext=(px_x, px_y),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    ax1.set_xlabel('像素 X (px)')
    ax1.set_ylabel('像素 Y (px)')
    ax1.set_title('像素坐标系\n（传感器平面，原点在左上角）', fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.02,
             f'传感器像素: {PIXEL_SIZE_UM}μm\n坐标原点: 图像左上角\n↓ 坐标变换（标定矩阵）',
             transform=ax1.transAxes, va='bottom', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.9))

    # ── 子图2：焦面坐标系 ──
    ax2 = fig.add_subplot(gs[1])
    ref_x_mm = ref_focal_um[:, 0] / 1000
    ref_y_mm = ref_focal_um[:, 1] / 1000
    tgt_true_x_mm = target_focal_true[:, 0] / 1000
    tgt_true_y_mm = target_focal_true[:, 1] / 1000
    tgt_pred_x_mm = target_focal_pred[:, 0] / 1000
    tgt_pred_y_mm = target_focal_pred[:, 1] / 1000

    ax2.scatter(ref_x_mm, ref_y_mm, s=100, c='red', marker='D',
                zorder=5, label='基准光纤（焦面坐标）', alpha=0.8)
    ax2.scatter(tgt_true_x_mm, tgt_true_y_mm, s=30, c='steelblue',
                zorder=4, alpha=0.5, label='待测真值')
    ax2.scatter(tgt_pred_x_mm, tgt_pred_y_mm, s=15, c='orange',
                marker='x', zorder=6, alpha=0.8, label='变换预测值')

    ax2.set_xlabel('焦面 X (mm)')
    ax2.set_ylabel('焦面 Y (mm)')
    ax2.set_title('焦面坐标系\n（物理空间，原点由设计定义）', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.text(0.02, 0.02,
             f'焦面尺度: {FOCAL_PLANE_SCALE_UM_PX:.1f}μm/px\n坐标原点: ({REFERENCE_GRID_ORIGIN_MM[0]}, {REFERENCE_GRID_ORIGIN_MM[1]})mm',
             transform=ax2.transAxes, va='bottom', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#e6f2ff', alpha=0.9))

    # ── 子图3：误差矢量图（焦面坐标系） ──
    ax3 = fig.add_subplot(gs[2])
    rms_r = float(np.sqrt(np.mean(err_r**2)))

    sc = ax3.scatter(tgt_true_x_mm, tgt_true_y_mm,
                     c=err_r, cmap='RdYlGn_r',
                     s=50, zorder=4,
                     norm=Normalize(vmin=0, vmax=TARGET_ACCURACY_UM))
    plt.colorbar(sc, ax=ax3, label='误差 (μm)', shrink=0.9)

    # 误差矢量（放大500倍）
    scale_v = 500
    ax3.quiver(tgt_true_x_mm, tgt_true_y_mm,
               err[:, 0] / 1000 * scale_v,
               err[:, 1] / 1000 * scale_v,
               alpha=0.6, color='gray',
               scale=1, scale_units='xy', width=0.002)

    ax3.set_xlabel('焦面 X (mm)')
    ax3.set_ylabel('焦面 Y (mm)')
    ax3.set_title(f'坐标变换误差矢量图\n（矢量已放大{scale_v}倍）', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    info3 = (f'合成 RMS: {rms_r:.3f} μm\n'
             f'X RMS: {np.std(err[:,0]):.3f} μm\n'
             f'Y RMS: {np.std(err[:,1]):.3f} μm\n'
             f'最大误差: {err_r.max():.3f} μm')
    ax3.text(0.02, 0.97, info3, transform=ax3.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ── 子图4：误差圆分布 + CDF ──
    ax4 = fig.add_subplot(gs[3])
    ax4_twin = ax4.twinx()

    theta = np.linspace(0, 2*np.pi, 200)
    colors = ['blue', 'green', 'red']
    labels = [f'RMS={rms_r:.3f}μm', f'95%={np.percentile(err_r,95):.3f}μm',
              f'目标={TARGET_ACCURACY_UM}μm']
    radii = [rms_r, np.percentile(err_r, 95), TARGET_ACCURACY_UM]
    styles = ['--', '-.', '-']
    for r, c, lbl, ls in zip(radii, colors, labels, styles):
        ax4.plot(r*np.cos(theta), r*np.sin(theta), color=c, linestyle=ls,
                 linewidth=1.8, label=lbl)

    ax4.scatter(err[:, 0], err[:, 1], s=12, alpha=0.5, color='steelblue')
    ax4.set_xlabel('X误差 (μm)')
    ax4.set_ylabel('Y误差 (μm)')
    ax4.set_title('误差分布圆图\n（焦面坐标系）', fontsize=10)
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # CDF曲线（右轴）
    sorted_err = np.sort(err_r)
    cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    ax4_twin.plot([], [], 'purple', linestyle=':', linewidth=1.5, label='CDF')
    ax4_twin.set_ylabel('累积概率', color='purple', fontsize=9)
    ax4_twin.tick_params(axis='y', labelcolor='purple')

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "4_coordinate_transform.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {path}")
    plt.close()


# ============================================================
# 主入口
# ============================================================

def main(part=0):
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 55)
    print("  MUST FVC 可视化模块")
    print(f"  输出目录: {VIZ_DIR}")
    print("=" * 55)

    if part in (0, 1):
        spot, cx, cy = viz_spot(rng)
    else:
        # 其他部分也需要spot数据
        patch_size = 60
        cx = patch_size/2 + 2.3
        cy = patch_size/2 - 1.7
        spot = generate_gaussian_spot(cx, cy, image_size=patch_size, rng=rng)

    if part in (0, 2):
        viz_centroid(spot, cx, cy, rng)

    calibrator, ref_focal_um = None, None
    if part in (0, 3, 4):
        calibrator, ref_focal_um = viz_calibration(rng)

    if part in (0, 4):
        if calibrator is None:
            calibrator, ref_focal_um = viz_calibration(rng, save=False)
        viz_coordinate_transform(calibrator, ref_focal_um, rng)

    print("\n全部可视化完成！")
    print(f"图像保存在: {VIZ_DIR}")
    print("  1_spot_image.png        - 光斑图像")
    print("  2_centroid_fitting.png  - 质心定位过程")
    print("  3_calibration_points.png- 标定点分布与残差")
    print("  4_coordinate_transform.png - 坐标变换")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MUST FVC 可视化模块')
    parser.add_argument('--part', type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help='0=全部, 1=光斑, 2=质心, 3=标定, 4=坐标变换')
    args = parser.parse_args()
    main(part=args.part)

