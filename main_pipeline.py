"""
MUST望远镜 FVC 光纤位置测量系统 - 主流程
流程：仿真生成光斑 → 高斯拟合检测 → 标定 → 坐标变换 → 精度评估

新增功能：
    - 径向畸变仿真（模拟真实镜头）
    - 3阶多项式标定（适应大视场）
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import (
    PIXEL_SIZE_UM, DEMAGNIFICATION, FOCAL_PLANE_SCALE_UM_PX,
    SPOT_SIGMA_PX, SPOT_PEAK_COUNTS, BACKGROUND_COUNTS,
    NUM_TARGET_FIBERS,
    REFERENCE_GRID_SPACING_MM, REFERENCE_GRID_ORIGIN_MM,
    TARGET_ACCURACY_UM, TARGET_ACCURACY_PX,
    RANDOM_SEED, OUTPUT_DIR,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    DISTORTION_K1, DISTORTION_K2, FOCAL_RADIUS_UM,
    REF_GRID_SIDE,  # 新增
    POLY_ORDER,
    ELLIPTICAL_SPOT_PROB, ELLIPTICITY_RANGE
)
from spot_generator import generate_gaussian_spot
from gaussian_detector import GaussianDetector, fit_gaussian
from coordinate_transform import FVCCalibrator
from spot_detector import detect_all_spots
from gaussian_detector import fit_with_fallback


DISTORTION_DEGREE = POLY_ORDER  # 使用config中的3阶设置


# ============================================================
# 工具函数
# ============================================================

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)


def build_reference_grid(n_side=REF_GRID_SIDE,
                         spacing_mm=REFERENCE_GRID_SPACING_MM,
                         origin_mm=REFERENCE_GRID_ORIGIN_MM):
    """构建基准光纤焦面坐标（规则格网）

    Parameters
    ----------
    n_side      : int    格网边长（n×n）
    spacing_mm  : float  光纤间距 (mm)
    origin_mm   : tuple  格网原点 (x, y) mm

    Returns
    -------
    ref_focal : (N, 2) array  基准光纤焦面坐标 (μm)
    """
    ref_focal = []
    ox = origin_mm[0] * 1000  # mm → μm
    oy = origin_mm[1] * 1000
    spacing_um = spacing_mm * 1000
    for i in range(n_side):
        for j in range(n_side):
            ref_focal.append([ox + i * spacing_um, oy + j * spacing_um])
    return np.array(ref_focal, dtype=float)


def focal_to_pixel(focal_um, scale=None, rotation_deg=0.5,
                   offset_px=None):   # 改为None，动态计算    (3000.0, 2500.0)):
    """仿真专用：焦面坐标 (μm) → 像素坐标 (pixel)

    新增：径向畸变仿真（模拟真实镜头）

    流程：
        1. 焦面坐标 → 仿射变换 → 理想像素坐标
        2. 叠加径向畸变（桶形畸变）
        3. 可选：亚像素随机抖动

    Parameters
    ----------
    focal_um     : (N, 2) array  焦面坐标 [x, y] (μm)
    scale        : float         缩放因子 (px/μm)，None时用默认值
    rotation_deg : float         旋转角度（度）
    offset_px    : tuple         平移偏移 (x, y) px

    Returns
    -------
    px : (N, 2) array  像素坐标 [x, y]
    """
    if scale is None:
        scale = 1.0 / FOCAL_PLANE_SCALE_UM_PX  # px/μm
    if offset_px is None:
        # 让焦面原点(0,0)映射到图像中心，消除不对称性
        offset_px = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)  # (7104, 5328)

    # 步骤1：仿射变换（旋转 + 缩放 + 平移）
    angle = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    focal = np.atleast_2d(focal_um).astype(float)
    px = (scale * (R @ focal.T).T) + np.array(offset_px)

    # ──────────────────────────────────────────────────────
    # 步骤2：径向畸变（新增）
    # ──────────────────────────────────────────────────────
    cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2  # 图像中心（光轴位置）

    # 计算每个点到图像中心的焦面距离（μm）
    dx_um = (px[:, 0] - cx) * FOCAL_PLANE_SCALE_UM_PX
    dy_um = (px[:, 1] - cy) * FOCAL_PLANE_SCALE_UM_PX
    r_um = np.sqrt(dx_um**2 + dy_um**2)

    # 归一化半径（0到1之间，焦面边缘为1）
    r_norm = r_um / FOCAL_RADIUS_UM
    r2 = r_norm ** 2
    r4 = r2 ** 2

    # 径向畸变因子
    # k1 < 0 产生桶形畸变（边缘向内收缩）
    # k2 > 0 使畸变在边缘加速增长
    distortion_factor = 1.0 + DISTORTION_K1 * r2 + DISTORTION_K2 * r4

    # 应用畸变（从光轴中心向外放射状变形）
    px[:, 0] = cx + (px[:, 0] - cx) * distortion_factor
    px[:, 1] = cy + (px[:, 1] - cy) * distortion_factor
    # ──────────────────────────────────────────────────────

    return px


# ============================================================
# 步骤1：高斯拟合检测精度独立测试
# ============================================================

def run_detection_accuracy_test(n_trials=500, verbose=True):
    """单独测试高斯拟合质心精度（不涉及坐标变换）

    Parameters
    ----------
    n_trials : int   试验次数
    verbose  : bool  是否打印详细信息

    Returns
    -------
    stats    : dict  精度统计结果
    errors_x : array X方向误差 (px)
    errors_y : array Y方向误差 (px)
    """
    print("\n" + "=" * 50)
    print("步骤1：高斯拟合质心精度测试")
    print("=" * 50)

    rng = np.random.default_rng(RANDOM_SEED)
    patch_size = 50

    errors_x, errors_y, snrs = [], [], []
    n_failed = 0

    for _ in range(n_trials):
        # 随机生成真值位置（patch中心附近）
        true_x = patch_size / 2 + rng.uniform(-3.0, 3.0)
        true_y = patch_size / 2 + rng.uniform(-3.0, 3.0)

        # 生成仿真光斑（支持椭圆）
        patch = generate_gaussian_spot(true_x, true_y, image_size=patch_size, rng=rng,
                                       ellipticity_prob=ELLIPTICAL_SPOT_PROB)

        # 高斯拟合检测（使用椭圆模型）
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                              use_elliptical=True)

        if result['success']:
            errors_x.append(result['x0'] - true_x)
            errors_y.append(result['y0'] - true_y)
            snrs.append(result['snr'])
        else:
            n_failed += 1

    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    rms_x = float(np.std(errors_x))
    rms_y = float(np.std(errors_y))
    rms_r = float(np.sqrt(rms_x ** 2 + rms_y ** 2))
    rms_um = rms_r * FOCAL_PLANE_SCALE_UM_PX
    mean_snr = float(np.mean(snrs)) if snrs else 0.0

    stats = {
        'n_trials': n_trials,
        'n_success': n_trials - n_failed,
        'n_failed': n_failed,
        'rms_x_px': rms_x,
        'rms_y_px': rms_y,
        'rms_r_px': rms_r,
        'rms_um': rms_um,
        'mean_snr': mean_snr,
        'bias_x_px': float(np.mean(errors_x)),
        'bias_y_px': float(np.mean(errors_y)),
    }

    if verbose:
        print(f"  试验次数:   {n_trials}  (成功: {n_trials - n_failed})")
        print(f"  平均 SNR:   {mean_snr:.1f}")
        print(f"  X  RMS:     {rms_x:.5f} px")
        print(f"  Y  RMS:     {rms_y:.5f} px")
        print(f"  合成 RMS:   {rms_r:.5f} px  =  {rms_um:.3f} um (焦面尺度)")
        print(f"  X  偏差:    {stats['bias_x_px']:.5f} px")
        print(f"  Y  偏差:    {stats['bias_y_px']:.5f} px")
        status = "✓ 达标" if rms_um <= TARGET_ACCURACY_UM else "✗ 未达标"
        print(f"  目标 {TARGET_ACCURACY_UM} um → {status}")

    # 误差分布图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(errors_x * FOCAL_PLANE_SCALE_UM_PX, bins=40,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('X误差 (um)')
    axes[0].set_ylabel('频次')
    axes[0].set_title(f'X方向误差分布  RMS={rms_x * FOCAL_PLANE_SCALE_UM_PX:.3f}um')

    axes[1].hist(errors_y * FOCAL_PLANE_SCALE_UM_PX, bins=40,
                 color='darkorange', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Y误差 (um)')
    axes[1].set_ylabel('频次')
    axes[1].set_title(f'Y方向误差分布  RMS={rms_y * FOCAL_PLANE_SCALE_UM_PX:.3f}um')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "figures", "detection_error_hist.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  误差分布图已保存: {fig_path}")

    return stats, errors_x, errors_y


# ============================================================
# 步骤2：标定流程
# ============================================================

def run_calibration(rng, verbose=True):
    """用基准光纤标定 FVC 坐标变换模型（3阶多项式）

    Parameters
    ----------
    rng     : np.random.Generator  随机数生成器
    verbose : bool 是否打印详细信息

    Returns
    -------
    calibrator    : FVCCalibrator  标定器对象
    ref_focal_um  : array          基准光纤焦面坐标 (μm)
    ref_px_detected : array        基准光纤检测像素坐标
    calib_report  : dict           标定报告
    """
    print("\n" + "=" * 50)
    print("步骤2：基准光纤标定")
    print("=" * 50)

    # 构建基准光纤焦面坐标
    ref_focal_um = build_reference_grid(n_side=REF_GRID_SIDE)
    N_ref = len(ref_focal_um)
    spacing_um = REFERENCE_GRID_SPACING_MM * 1000
    coverage_mm = (REF_GRID_SIDE - 1) * REFERENCE_GRID_SPACING_MM

    # 计算参数比（3阶多项式每方向10个参数）
    n_params_per_dim = (DISTORTION_DEGREE + 1) * (DISTORTION_DEGREE + 2) // 2
    param_ratio = (N_ref * 2) / (n_params_per_dim * 2)

    print(f"  基准光纤数: {N_ref}  ({REF_GRID_SIDE}x{REF_GRID_SIDE} 格网)")
    print(f"  间距: {REFERENCE_GRID_SPACING_MM} mm，覆盖范围: {coverage_mm}x{coverage_mm} mm")
    print(f"  多项式阶数: {DISTORTION_DEGREE}  (参数比 ≈ {param_ratio:.1f}:1)")

    # # 焦面坐标 → 像素坐标（含畸变）
    # ref_px_ideal = focal_to_pixel(ref_focal_um)
    # # 加入基准光纤检测误差（模拟真实检测的微小误差）
    # ref_px_detected = ref_px_ideal + rng.normal(0, 0.01, ref_px_ideal.shape)
    # 新做法：生成仿真光斑并用高斯拟合检测
    ref_px_ideal = focal_to_pixel(ref_focal_um)
    ref_px_detected = []
    for px, py in ref_px_ideal:
        patch_size = 50
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        # 生成仿真光斑（支持椭圆）
        patch = generate_gaussian_spot(true_cx, true_cy,
                                       image_size=patch_size, rng=rng,
                                       ellipticity_prob=ELLIPTICAL_SPOT_PROB)

        # 高斯拟合检测（使用椭圆模型）
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                              use_elliptical=True)
        if result['success']:
            offset_x = round(px) - patch_size // 2
            offset_y = round(py) - patch_size // 2
            detected_x = result['x0'] + offset_x
            detected_y = result['y0'] + offset_y
            ref_px_detected.append([detected_x, detected_y])
        else:
            # 检测失败退回理想值
            ref_px_detected.append([px, py])
    ref_px_detected = np.array(ref_px_detected)

    # 执行标定
    calibrator = FVCCalibrator(poly_degree=DISTORTION_DEGREE)
    calib_report = calibrator.calibrate(ref_px_detected, ref_focal_um, verbose=verbose)

    # 加这两行调试
    print(f"  [调试] bias_x={calibrator.bias_x:.4f} um, bias_y={calibrator.bias_y:.4f} um")
    print(f"  [调试] is_calibrated={calibrator.is_calibrated}")

    # ──────────────────────────────────────────────────────
    # 可视化：基准光纤仿真图像 + 高斯拟合检测结果
    # ──────────────────────────────────────────────────────
    _visualize_reference_spots(ref_focal_um, ref_px_ideal,
                                ref_px_detected, rng)
    return calibrator, ref_focal_um, ref_px_detected, calib_report


def _visualize_reference_spots(ref_focal_um, ref_px_ideal,
                               ref_px_detected, rng):
    """可视化基准光纤仿真图像和高斯拟合检测结果

    生成两张图：
      1. 全局分布图：49个基准点在焦面坐标系中的分布
      2. 单光斑细节图：随机抽取6个基准点，展示仿真光斑和拟合结果
    """
    fig_path_global = os.path.join(OUTPUT_DIR, "figures", "ref_spots_global.png")
    fig_path_detail = os.path.join(OUTPUT_DIR, "figures", "ref_spots_detail.png")

    # ── 图1：全局分布图 ──────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 绘制基准点焦面坐标
    ax.scatter(ref_focal_um[:, 0] / 1000, ref_focal_um[:, 1] / 1000,
               s=80, c='steelblue', marker='o', zorder=3, label='焦面真值')

    # 绘制检测误差放大矢量（放大500倍便于观察）
    scale_factor = 500
    detection_err = ref_px_detected - ref_px_ideal  # 像素误差
    detection_err_um = detection_err * FOCAL_PLANE_SCALE_UM_PX  # 转μm
    ax.quiver(ref_focal_um[:, 0] / 1000, ref_focal_um[:, 1] / 1000,
              detection_err_um[:, 0] * scale_factor,
              detection_err_um[:, 1] * scale_factor,
              color='red', alpha=0.7, width=0.003,
              label=f'检测误差×{scale_factor}')

    ax.set_xlabel('焦面 X (mm)')
    ax.set_ylabel('焦面 Y (mm)')
    ax.set_title(f'基准光纤分布（{len(ref_focal_um)}个，7×7格网）\n'
                 f'红色箭头为检测误差放大{scale_factor}倍')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path_global, dpi=150)
    plt.close()
    print(f"  基准点分布图已保存: {fig_path_global}")

    # ── 图2：单光斑细节图（随机抽取6个） ────────────────
    n_show = 6
    indices = rng.choice(len(ref_focal_um), size=n_show, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    patch_size = 50

    for plot_idx, fiber_idx in enumerate(indices):
        px, py = ref_px_ideal[fiber_idx]
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))

        # 重新生成该光斑（用固定种子保证可重复，支持椭圆）
        patch_rng = np.random.default_rng(RANDOM_SEED + fiber_idx)
        patch = generate_gaussian_spot(true_cx, true_cy,
                                       image_size=patch_size, rng=patch_rng,
                                       ellipticity_prob=ELLIPTICAL_SPOT_PROB)

        # 高斯拟合（使用椭圆模型）
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                              use_elliptical=True)

        ax = axes[plot_idx]
        im = ax.imshow(patch, origin='lower', cmap='hot',
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 标注真值位置（蓝色十字）
        ax.plot(true_cx, true_cy, 'b+', markersize=12,
                markeredgewidth=2, label='真值')

        # 标注拟合位置（红色圆圈）
        if result['success']:
            ax.plot(result['x0'], result['y0'], 'ro', markersize=8,
                    fillstyle='none', markeredgewidth=2, label='拟合')
            err_px = np.sqrt((result['x0'] - true_cx) ** 2 +
                             (result['y0'] - true_cy) ** 2)
            err_um = err_px * FOCAL_PLANE_SCALE_UM_PX

            # 提取椭圆参数
            sigma_x = result.get('sigma_x') or result.get('sigma') or 0.0
            sigma_y = result.get('sigma_y') or sigma_x
            theta_rad = result.get('theta') or 0.0
            theta_deg = np.rad2deg(theta_rad)
            ellipticity = sigma_y / sigma_x if sigma_x > 0 else 1.0
            snr_val = result.get('snr') or result.get('SNR') or 0.0

            # 显示椭圆参数
            title = (f'光纤#{fiber_idx}  SNR={snr_val:.0f}\n'
                     f'误差={err_um:.3f}μm  椭圆率={ellipticity:.2f}\n'
                     f'σx={sigma_x:.2f} σy={sigma_y:.2f} θ={theta_deg:.0f}°')
        else:
            title = f'光纤#{fiber_idx}  拟合失败'

        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

    plt.suptitle('基准光纤仿真光斑与高斯拟合结果（随机抽取6个）',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path_detail, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  光斑细节图已保存: {fig_path_detail}")

# ============================================================
# 步骤3：待测光纤完整测量流程
# ============================================================

def run_measurement(calibrator, ref_focal_um, n_targets=None, rng=None,verbose=True,
                    use_real_image=False,
                    real_image_path=None):
    """模拟待测光纤完整测量流程

    Parameters
    ----------
    calibrator      : FVCCalibrator  标定器对象
    ref_focal_um    : array          基准光纤焦面坐标（用于确定测量范围）
    n_targets       : int            待测光纤数量
    rng             : Generator      随机数生成器
    verbose         : bool           是否打印详细信息
    use_real_image  : bool           是否使用真实图像
    real_image_path : str            真实图像路径

    Returns
    -------
    stats : dict  测量精度统计
    err   : array 误差数组（仿真模式）或None（真实图像模式）
    """
    print("\n" + "=" * 50)
    print("步骤3：待测光纤测量精度评估")
    print("=" * 50)

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 1)
    if n_targets is None:
        n_targets = NUM_TARGET_FIBERS

    # 待测点范围从基准格网实际覆盖范围动态计算
    margin = 0.05  # 留10%边距，避免外推    从0.1改为0.05
    focal_min = ref_focal_um.min(axis=0)
    focal_max = ref_focal_um.max(axis=0)
    focal_range = focal_max - focal_min
    target_low = focal_min + margin * focal_range
    target_high = focal_max - margin * focal_range

    print(f"  基准格网覆盖: X [{focal_min[0]/1000:.1f}, {focal_max[0]/1000:.1f}] mm, "
          f"Y [{focal_min[1]/1000:.1f}, {focal_max[1]/1000:.1f}] mm")
    print(f"  待测点范围:   X [{target_low[0]/1000:.1f}, {target_high[0]/1000:.1f}] mm, "
          f"Y [{target_low[1]/1000:.1f}, {target_high[1]/1000:.1f}] mm (留10%边距)")

    # 生成待测光纤焦面真值
    target_focal_true = rng.uniform(target_low, target_high, (n_targets, 2))

    # 焦面坐标 → 像素坐标（含畸变）
    target_px_true = focal_to_pixel(target_focal_true)

    # 像素坐标检测
    print(f"  生成 {n_targets} 个待测光纤仿真光斑并检测...")
    target_px_detected = []
    detection_success = []
    method_counts = {'gaussian': 0, 'centroid': 0, 'failed': 0}
    t_start = time.time()

    if use_real_image and real_image_path is not None:
        # ──────────────────────────────────────────────────────
        # 真实图像模式（预留接口，暂未实现）
        # ──────────────────────────────────────────────────────
        print(f"  [真实图像模式] 读取: {real_image_path}")
        if real_image_path.endswith('.fits') or real_image_path.endswith('.fit'):
            try:
                import astropy.io.fits as fits
                real_image = fits.getdata(real_image_path).astype(float)
            except ImportError:
                raise ImportError("读取FITS文件需要安装astropy: pip install astropy")
        else:
            from PIL import Image
            real_image = np.array(Image.open(real_image_path).convert('L'), dtype=float)

        print(f"  图像尺寸: {real_image.shape[1]}×{real_image.shape[0]} px")

        # 粗检测：找到所有光斑位置
        rough_positions = detect_all_spots(real_image, method='threshold', verbose=verbose)

        # 精检测：对每个光斑做高斯拟合
        patch_half = 25
        for cx, cy in rough_positions:
            x0_int = int(round(cx))
            y0_int = int(round(cy))

            # 边界检查
            if (x0_int < patch_half or y0_int < patch_half or
                    x0_int + patch_half >= real_image.shape[1] or
                    y0_int + patch_half >= real_image.shape[0]):
                continue

            # 提取patch
            patch = real_image[y0_int - patch_half: y0_int + patch_half,
                               x0_int - patch_half: x0_int + patch_half]

            # 高斯拟合（带降级策略）
            result = fit_with_fallback(patch)

            if result['success']:
                # 转回全图坐标
                full_x = result['x0'] + x0_int - patch_half
                full_y = result['y0'] + y0_int - patch_half
                target_px_detected.append([full_x, full_y])
                detection_success.append(True)
                method_counts[result.get('method', 'gaussian')] += 1
            else:
                method_counts['failed'] += 1

        target_px_detected = np.array(target_px_detected) if target_px_detected else np.zeros((0, 2))
        target_focal_true = None  # 真实图像无焦面真值

    else:
        # ──────────────────────────────────────────────────────
        # 仿真模式
        # ──────────────────────────────────────────────────────
        for px, py in target_px_true:
            # 生成仿真光斑（patch中心对应像素坐标的小数部分）
            patch_size = 50
            true_cx = patch_size / 2 + (px - round(px))
            true_cy = patch_size / 2 + (py - round(py))
            patch = generate_gaussian_spot(true_cx, true_cy, image_size=patch_size, rng=rng,
                                           ellipticity_prob=ELLIPTICAL_SPOT_PROB)

            # 高斯拟合检测（使用椭圆模型）
            result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                                  use_elliptical=True)

            if result['success']:
                # 转回全图坐标
                offset_x = round(px) - patch_size // 2
                offset_y = round(py) - patch_size // 2
                detected_x = result['x0'] + offset_x
                detected_y = result['y0'] + offset_y
                target_px_detected.append([detected_x, detected_y])
                detection_success.append(True)
                method_counts['gaussian'] += 1
            else:
                # 检测失败，用粗略位置
                target_px_detected.append([px, py])
                detection_success.append(False)
                method_counts['failed'] += 1

    t_detect = time.time() - t_start
    target_px_detected = np.array(target_px_detected)
    n_success = sum(detection_success)
    print(f"  检测完成: {n_success}/{n_targets if not use_real_image else len(rough_positions)} "
          f"成功  耗时 {t_detect:.2f}s")
    print(f"  方法统计: 高斯拟合={method_counts['gaussian']}  "
          f"质心法={method_counts['centroid']}  失败={method_counts['failed']}")

    # 真实图像模式：无焦面真值，跳过精度统计
    if use_real_image:
        print("  [真实图像模式] 检测完成，无焦面真值，跳过精度统计")
        return {'n_detected': n_success, 'mode': 'real_image'}, None

    # ──────────────────────────────────────────────────────
    # 仿真模式：坐标变换 + 精度统计
    # ──────────────────────────────────────────────────────

    # 坐标变换：像素坐标 → 焦面坐标
    target_focal_predicted = calibrator.transform(target_px_detected)

    # 精度统计（仅统计检测成功的点）
    success_mask = np.array(detection_success)
    err = target_focal_true[success_mask] - target_focal_predicted[success_mask]
    err_x = err[:, 0]
    err_y = err[:, 1]
    err_r = np.sqrt(err_x ** 2 + err_y ** 2)

    rms_x = float(np.std(err_x))
    rms_y = float(np.std(err_y))
    rms_r = float(np.sqrt(np.mean(err_r ** 2)))
    bias_x = float(np.mean(err_x))
    bias_y = float(np.mean(err_y))
    max_err = float(np.max(err_r))
    p95_err = float(np.percentile(err_r, 95))

    stats = {
        'n_targets': n_targets,
        'n_detected': n_success,
        'rms_x_um': rms_x,
        'rms_y_um': rms_y,
        'rms_r_um': rms_r,
        'bias_x_um': bias_x,
        'bias_y_um': bias_y,
        'max_err_um': max_err,
        'p95_err_um': p95_err,
        'target_accuracy_um': TARGET_ACCURACY_UM,
        'pass': rms_r <= TARGET_ACCURACY_UM,
    }

    # ──────────────────────────────────────────────────────
    # 新增：生成完整结果表
    # ──────────────────────────────────────────────────────
    result_table = _build_result_table(
        target_focal_true, target_px_true,
        target_px_detected, target_focal_predicted,
        success_mask
    )
    _save_result_table(result_table)
    _plot_result_figures(result_table, rms_r, p95_err)

    if verbose:
        print(f"\n  ── 测量精度统计 (N={n_success}) ──")
        print(f"  X  RMS:    {rms_x:.3f} um")
        print(f"  Y  RMS:    {rms_y:.3f} um")
        print(f"  合成 RMS:  {rms_r:.3f} um")
        print(f"  X  偏差:   {bias_x:.3f} um")
        print(f"  Y  偏差:   {bias_y:.3f} um")
        print(f"  最大误差:  {max_err:.3f} um")
        print(f"  95%分位:   {p95_err:.3f} um")
        status = "✓ 达标" if rms_r <= TARGET_ACCURACY_UM else "✗ 未达标"
        print(f"  目标 {TARGET_ACCURACY_UM} um → {status}")

    # 输出图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：误差矢量图
    ax = axes[0]
    sc = ax.quiver(
        target_focal_true[success_mask, 0] / 1000,
        target_focal_true[success_mask, 1] / 1000,
        err_x, err_y,
        err_r,
        cmap='RdYlGn_r', alpha=0.8,
        width=0.003
    )
    plt.colorbar(sc, ax=ax, label='误差 (um)')
    ax.set_xlabel('焦面 X (mm)')
    ax.set_ylabel('焦面 Y (mm)')
    ax.set_title(f'位置误差矢量图  RMS={rms_r:.3f}um')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 右图：误差分布（焦面坐标系）
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rms_r * np.cos(theta), rms_r * np.sin(theta),
            'b--', linewidth=2, label=f'RMS={rms_r:.3f}um')
    ax.plot(TARGET_ACCURACY_UM * np.cos(theta), TARGET_ACCURACY_UM * np.sin(theta),
            'r-', linewidth=2, label=f'目标={TARGET_ACCURACY_UM}um')
    ax.scatter(err_x, err_y, s=15, alpha=0.5, color='steelblue')
    ax.set_xlabel('X误差 (um)')
    ax.set_ylabel('Y误差 (um)')
    ax.set_title('误差分布（焦面坐标系）')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "figures", "measurement_accuracy.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  精度分布图已保存: {fig_path}")

    return stats, err, result_table


# ============================================================
# 步骤4：保存结果报告
# ============================================================

def save_report(detection_stats, calib_report, measurement_stats):
    """保存完整测试报告为JSON文件

    Parameters
    ----------
    detection_stats   : dict  步骤1检测精度统计
    calib_report      : dict  步骤2标定报告
    measurement_stats : dict  步骤3测量精度统计
    """
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'pixel_size_um': PIXEL_SIZE_UM,
            'demagnification': DEMAGNIFICATION,
            'focal_plane_scale_um_per_px': FOCAL_PLANE_SCALE_UM_PX,
            'spot_sigma_px': SPOT_SIGMA_PX,
            'target_accuracy_um': TARGET_ACCURACY_UM,
            'ref_grid_side': REF_GRID_SIDE,
            'distortion_poly_degree': DISTORTION_DEGREE,
            'distortion_k1': DISTORTION_K1,
            'distortion_k2': DISTORTION_K2,
        },
        'detection_accuracy': detection_stats,
        'calibration': calib_report,
        'measurement_accuracy': measurement_stats,'overall_pass': measurement_stats.get('pass', False),
    }

    report_path = os.path.join(OUTPUT_DIR, "results", "accuracy_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存: {report_path}")


# ============================================================
# 主流程
# ============================================================

def main(mode='full'):
    """主流程入口

    Parameters
    ----------
    mode : str  运行模式
        'full'      - 完整流程（步骤1+2+3）
        'detection' - 仅步骤1（检测精度测试）
        'calibration' - 仅步骤2（标定）
        'measurement' - 步骤2+3（标定+测量）
    """
    ensure_output_dir()

    # 打印系统信息
    print("╔══════════════════════════════════════════════════╗")
    print("║   MUST望远镜 FVC 光纤位置测量系统               ║")
    print(f"║   目标精度: {TARGET_ACCURACY_UM} um    模式: {mode:20s} ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  焦面尺度: {FOCAL_PLANE_SCALE_UM_PX:.2f} um/px")
    print(f"  精度换算: {TARGET_ACCURACY_UM} um = {TARGET_ACCURACY_PX:.4f} px")
    print(f"  基准格网: {REF_GRID_SIDE}x{REF_GRID_SIDE} = {REF_GRID_SIDE ** 2} 个基准点")
    print(f"  畸变模型: {DISTORTION_DEGREE} 阶多项式")
    print(f"  畸变系数: k1={DISTORTION_K1}, k2={DISTORTION_K2}")

    rng = np.random.default_rng(RANDOM_SEED)

    # 步骤1：检测精度测试
    if mode in ['full', 'detection']:
        detection_stats, _, _ = run_detection_accuracy_test(n_trials=500, verbose=True)
    else:
        detection_stats = None

    # 步骤2：标定
    if mode in ['full', 'calibration', 'measurement']:
        calibrator, ref_focal_um, ref_px_detected, calib_report = run_calibration(rng, verbose=True)
    else:
        print("\n跳过标定步骤")
        return

    # 步骤3：测量精度评估
    if mode in ['full', 'measurement']:
        measurement_stats, _, result_table = run_measurement(
            calibrator, ref_focal_um,
            n_targets=NUM_TARGET_FIBERS,
            rng=rng,
            verbose=True
        )
    else:
        measurement_stats = None

    # 保存报告
    if mode == 'full' and detection_stats and measurement_stats:
        save_report(detection_stats, calib_report, measurement_stats)

    # 最终结论
    print("\n" + "=" * 50)
    print("最终结论")
    print("=" * 50)
    if detection_stats:
        print(f"  高斯拟合质心精度:  {detection_stats['rms_um']:.3f} um")
        if calib_report:
            print(f"  标定残差:          {calib_report['final_rms_um']:.3f} um")
    if measurement_stats:
        print(f"  端对端测量精度:    {measurement_stats['rms_r_um']:.3f} um")
        status = "✓ 系统达标" if measurement_stats['pass'] else "✗ 系统未达标"
        print(f"  综合评估:          {status}")

    # ── 重标定状态检查 ────────────────────────────────────────────
    if mode in ['full', 'calibration', 'measurement']:
        print(f"\n{'='*60}")
        print(f"  重标定状态检查")
        print(f"{'='*60}")
        recal_check = calibrator.check_recalibration_needed(
            ref_px_detected,
            ref_focal_um,
            rms_threshold_um=2.0,
            bias_threshold_um=5.0,
            verbose=True
        )
        if recal_check['need_recalibration']:
            print(f"\n  [建议] 执行: {recal_check['recalibration_type']}")
            print(f"  [说明] 当前标定状态不满足精度要求，")
            print(f"         请重新采集基准光纤图像并执行重标定。")
        else:
            print(f"\n  [确认] 当前标定状态良好，无需重标定。")

    print("\n完成！")

# ============================================================
# 结果表生成、保存与可视化（新增）
# ============================================================

def _build_result_table(target_focal_true, target_px_true,
                        target_px_detected, target_focal_predicted,
                        success_mask):
    """构建完整结果表

    Parameters
    ----------
    target_focal_true      : (N,2) 焦面真值坐标 (μm)
    target_px_true         : (N,2) 像素坐标真值
    target_px_detected     : (N,2) 像素坐标检测值
    target_focal_predicted : (N,2) 焦面预测坐标 (μm)
    success_mask           : (N,)  检测成功掩码

    Returns
    -------
    table : list of dict  每行对应一根光纤的完整结果
    """
    table = []
    N = len(target_focal_true)

    for i in range(N):
        row = {
            'fiber_id':      i,
            'u_px':          float(target_px_detected[i, 0]),   # 检测像素坐标 X
            'v_px':          float(target_px_detected[i, 1]),   # 检测像素坐标 Y
            'X_true_um':     float(target_focal_true[i, 0]),    # 焦面真值 X (μm)
            'Y_true_um':     float(target_focal_true[i, 1]),    # 焦面真值 Y (μm)
            'X_meas_um':     float(target_focal_predicted[i, 0]),  # 焦面测量值 X (μm)
            'Y_meas_um':     float(target_focal_predicted[i, 1]),  # 焦面测量值 Y (μm)
            'dX_um':         float(target_focal_true[i, 0] - target_focal_predicted[i, 0]),
            'dY_um':         float(target_focal_true[i, 1] - target_focal_predicted[i, 1]),
            'radial_err_um': float(np.sqrt(
                                (target_focal_true[i, 0] - target_focal_predicted[i, 0])**2 +
                                (target_focal_true[i, 1] - target_focal_predicted[i, 1])**2
                             )),
            'detect_success': bool(success_mask[i]),
        }
        table.append(row)

    return table


def _save_result_table(table):
    """保存结果表为 CSV 和 JSON 两种格式

    CSV 格式便于 Excel 打开查看；
    JSON 格式便于程序读取。

    Parameters
    ----------
    table : list of dict  _build_result_table 的返回值
    """
    import csv

    # ── CSV ──────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "results", "fiber_positions.csv")
    fieldnames = [
        'fiber_id',
        'u_px', 'v_px',
        'X_true_um', 'Y_true_um',
        'X_meas_um', 'Y_meas_um',
        'dX_um', 'dY_um',
        'radial_err_um',
        'detect_success',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    print(f"\n  结果表 CSV 已保存: {csv_path}")

    # ── JSON ─────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "results", "fiber_positions.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(f"  结果表 JSON 已保存: {json_path}")

    # ── 打印前10行预览 ───────────────────────────────────
    print("\n  ── 结果表预览（前10行）──")
    header = (f"{'ID':>4}  {'u(px)':>8}  {'v(px)':>8}  "
              f"{'X_true':>10}  {'Y_true':>10}  "
              f"{'X_meas':>10}  {'Y_meas':>10}  "
              f"{'dX(μm)':>8}  {'dY(μm)':>8}  {'r_err(μm)':>10}")
    print("  " + header)
    print("  " + "-" * len(header))
    for row in table[:10]:
        line = (f"  {row['fiber_id']:>4}  "
                f"{row['u_px']:>8.2f}  {row['v_px']:>8.2f}  "
                f"{row['X_true_um']:>10.2f}  {row['Y_true_um']:>10.2f}  "
                f"{row['X_meas_um']:>10.2f}  {row['Y_meas_um']:>10.2f}  "
                f"{row['dX_um']:>8.3f}  {row['dY_um']:>8.3f}  "
                f"{row['radial_err_um']:>10.3f}")
        print(line)
    print(f"  ... (共 {len(table)} 行)")


def _plot_result_figures(table, rms_r, p95_err):
    """生成结果可视化图（两张）

    图1：焦面坐标散点图（测量值分布）
    图2：XY 误差矢量图

    Parameters
    ----------
    table   : list of dict  结果表
    rms_r   : float         径向误差 RMS (μm)
    p95_err : float         95% 分位误差 (μm)
    """
    # 提取数据
    X_true  = np.array([r['X_true_um']     for r in table]) / 1000   # μm → mm
    Y_true  = np.array([r['Y_true_um']     for r in table]) / 1000
    X_meas  = np.array([r['X_meas_um']     for r in table]) / 1000
    Y_meas  = np.array([r['Y_meas_um']     for r in table]) / 1000
    dX      = np.array([r['dX_um']         for r in table])           # μm
    dY      = np.array([r['dY_um']         for r in table])
    r_err   = np.array([r['radial_err_um'] for r in table])

    # ── 图1：焦面测量值散点图 ────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    sc = ax.scatter(X_meas, Y_meas,
                    c=r_err, cmap='RdYlGn_r',
                    s=20, alpha=0.7, vmin=0, vmax=p95_err * 1.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('径向误差 (μm)')

    ax.set_xlabel('焦面 X (mm)')
    ax.set_ylabel('焦面 Y (mm)')
    ax.set_title(f'焦面坐标测量结果（{len(table)} 根光纤）\n'
                 f'RMS={rms_r:.3f} μm  P95={p95_err:.3f} μm')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig_path1 = os.path.join(OUTPUT_DIR, "figures", "focal_xy_result.png")
    plt.tight_layout()
    plt.savefig(fig_path1, dpi=150)
    plt.close()
    print(f"\n  焦面坐标散点图已保存: {fig_path1}")

    # ── 图2：XY 误差矢量图 ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：矢量图（颜色=误差大小）
    ax = axes[0]
    q = ax.quiver(X_true, Y_true, dX, dY, r_err,
                  cmap='RdYlGn_r', alpha=0.8,
                  width=0.003, scale=None)
    plt.colorbar(q, ax=ax, label='径向误差 (μm)')
    ax.set_xlabel('焦面 X (mm)')
    ax.set_ylabel('焦面 Y (mm)')
    ax.set_title(f'XY 误差矢量图\n箭头方向=误差方向，颜色=误差大小')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 右：dX vs dY 散点图
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(rms_r * np.cos(theta), rms_r * np.sin(theta),
            'b--', linewidth=2, label=f'RMS = {rms_r:.3f} μm')
    ax.plot(p95_err * np.cos(theta), p95_err * np.sin(theta),
            'r-', linewidth=2, label=f'P95 = {p95_err:.3f} μm')
    ax.plot(TARGET_ACCURACY_UM * np.cos(theta),
            TARGET_ACCURACY_UM * np.sin(theta),
            'g:', linewidth=2, label=f'目标 = {TARGET_ACCURACY_UM} μm')
    ax.scatter(dX, dY, s=12, alpha=0.4, color='steelblue')
    ax.set_xlabel('dX (μm)')
    ax.set_ylabel('dY (μm)')
    ax.set_title('误差分布（焦面坐标系）')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig_path2 = os.path.join(OUTPUT_DIR, "figures", "xy_error_vector.png")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=150)
    plt.close()
    print(f"  XY 误差矢量图已保存: {fig_path2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MUST FVC 光纤位置测量系统')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'detection', 'calibration', 'measurement'],
                        help='运行模式：full(完整流程), detection(仅检测), calibration(仅标定), measurement(标定+测量)')
    args = parser.parse_args()
    main(mode=args.mode)
