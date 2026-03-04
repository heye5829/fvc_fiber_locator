"""
MUST望远镜 FVC 光纤位置测量系统 - 主流程
流程：仿真生成光斑 → 高斯拟合检测 → 标定 → 坐标变换 → 精度评估

运行方式：
    python main_pipeline.py
    python main_pipeline.py --mode calibration   # 仅标定
    python main_pipeline.py --mode detection     # 仅检测精度测试
    python main_pipeline.py --mode full          # 完整流程（默认）

坐标系说明：
    像素坐标：原点在图像左上角，x向右，y向下，单位 pixel
    焦面坐标：原点由 REFERENCE_GRID_ORIGIN_MM 定义（焦面板物理中心附近），
              x/y 对应焦面物理方向，单位 um
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 无GUI环境也能保存图像
import matplotlib.pyplot as plt

# ── 中文字体配置（解决方块警告）──────────────────────────────
# Windows系统：优先使用微软雅黑；如无，依次回退到SimHei、DejaVu Sans
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# ─────────────────────────────────────────────────────────────

from config import (
    PIXEL_SIZE_UM, DEMAGNIFICATION, FOCAL_PLANE_SCALE_UM_PX,
    SPOT_SIGMA_PX, SPOT_PEAK_COUNTS, BACKGROUND_COUNTS,
    NUM_TARGET_FIBERS,
    REFERENCE_GRID_SPACING_MM, REFERENCE_GRID_ORIGIN_MM,
    TARGET_ACCURACY_UM, TARGET_ACCURACY_PX,
    RANDOM_SEED, OUTPUT_DIR
)
from spot_generator import generate_gaussian_spot
from gaussian_detector import GaussianDetector, fit_gaussian
from coordinate_transform import FVCCalibrator

from spot_detector import detect_all_spots
from gaussian_detector import fit_with_fallback



# ============================================================
# 常量：过拟合修复方案
# 使用 5×5 = 25 个基准光纤，多项式阶数 2
# 参数比：25点 × 2维 = 50方程，2阶多项式 6参数/方向 → 约8:1，安全
# ============================================================
REF_GRID_SIDE = 5          # 基准格网边长（5×5=25个基准点）
DISTORTION_DEGREE = 2      # 畸变多项式阶数（2阶，不使用config中的3阶）


# ============================================================
# 工具函数
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)


def build_reference_grid(n_side=REF_GRID_SIDE,
                         spacing_mm=REFERENCE_GRID_SPACING_MM,
                         origin_mm=REFERENCE_GRID_ORIGIN_MM):
    """
    构建基准光纤焦面坐标（规则格网）

    坐标系：焦面物理坐标，原点由 origin_mm 定义，单位 um
    格网布局：n_side × n_side，共 n_side^2 个点
    例：n_side=5, spacing=50mm → 覆盖 200mm × 200mm 的焦面区域

    Returns
    -------
    ref_focal_um : (N, 2)  焦面坐标 [x_um, y_um]
    """
    ref_focal = []
    ox = origin_mm[0] * 1000  # mm → um
    oy = origin_mm[1] * 1000
    spacing_um = spacing_mm * 1000
    for i in range(n_side):
        for j in range(n_side):
            ref_focal.append([ox + i * spacing_um, oy + j * spacing_um])
    return np.array(ref_focal, dtype=float)


def focal_to_pixel(focal_um, scale=None, rotation_deg=0.5,
                   offset_px=(3000.0, 2500.0)):
    """
    仿真专用：焦面坐标 (um) → 像素坐标 (pixel)
    包含：缩放、旋转、平移

    坐标系：
        焦面坐标原点 → 像素坐标 offset_px 处（图像内某点，非左上角）
        旋转角 rotation_deg 模拟相机安装偏转

    参数 offset_px 应取传感器中心附近，确保所有光纤都在图像范围内。
    传感器: 14208 × 10656 px，中心约 (7104, 5328)，
    基准格网覆盖 ~200mm = 200000um，像素尺度 ~139um/px ≈ 1440px，
    offset 取 (3000, 2500) 保证格网在图像左半部，待测点也在同一区域。

    真实系统中此函数不存在（真实方向是像素→焦面），仅用于生成仿真数据。
    """
    if scale is None:
        scale = 1.0 / FOCAL_PLANE_SCALE_UM_PX  # px/um
    angle = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    focal = np.atleast_2d(focal_um).astype(float)
    px = (scale * (R @ focal.T).T) + np.array(offset_px)
    return px


# ============================================================
# 步骤1：高斯拟合检测精度独立测试
# ============================================================

def run_detection_accuracy_test(n_trials=500, verbose=True):
    """
    单独测试高斯拟合质心精度（不涉及坐标变换）
    此步骤反映高斯拟合本身能达到的像素级精度下限。
    """
    print("\n" + "=" * 50)
    print("步骤1：高斯拟合质心精度测试")
    print("=" * 50)

    rng = np.random.default_rng(RANDOM_SEED)
    patch_size = 50

    errors_x, errors_y, snrs = [], [], []
    n_failed = 0

    for _ in range(n_trials):
        true_x = patch_size / 2 + rng.uniform(-3.0, 3.0)
        true_y = patch_size / 2 + rng.uniform(-3.0, 3.0)

        patch = generate_gaussian_spot(true_x, true_y, image_size=patch_size, rng=rng)
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX)

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
    """
    用基准光纤标定 FVC 坐标变换模型

    修复说明（过拟合问题）：
    - 原方案：9个基准点 + 3阶多项式（参数数量≈10）→ 过拟合
    - 修复方案：25个基准点(5×5) + 2阶多项式（参数数量=6）→ 参数比≈8:1，安全

    Returns
    -------
    calibrator   : FVCCalibrator  已标定的变换器
    ref_focal_um : (N, 2)         基准光纤焦面坐标 (um)
    ref_px       : (N, 2)         基准光纤像素坐标（含检测噪声）
    calib_report : dict           标定报告
    """
    print("\n" + "=" * 50)
    print("步骤2：基准光纤标定")
    print("=" * 50)

    # 基准光纤焦面坐标（已知真值，格网原点在焦面物理坐标系中）
    ref_focal_um = build_reference_grid(n_side=REF_GRID_SIDE)
    N_ref = len(ref_focal_um)
    spacing_um = REFERENCE_GRID_SPACING_MM * 1000
    coverage_mm = (REF_GRID_SIDE - 1) * REFERENCE_GRID_SPACING_MM
    print(f"  基准光纤数: {N_ref}  ({REF_GRID_SIDE}x{REF_GRID_SIDE} 格网)")
    print(f"  间距: {REFERENCE_GRID_SPACING_MM} mm，覆盖范围: {coverage_mm}x{coverage_mm} mm")
    print(f"  多项式阶数: {DISTORTION_DEGREE}  (参数比 ≈ {N_ref * 2 / 6:.0f}:1)")

    # 焦面坐标 → 像素坐标（仿真真值，像素原点在图像左上角）
    ref_px_ideal = focal_to_pixel(ref_focal_um)

    # 加入基准光纤检测误差（假设 0.01px，略小于待测光纤）
    ref_px_detected = ref_px_ideal + rng.normal(0, 0.01, ref_px_ideal.shape)

    # 执行标定
    calibrator = FVCCalibrator(poly_degree=DISTORTION_DEGREE)
    calib_report = calibrator.calibrate(ref_px_detected, ref_focal_um, verbose=verbose)

    return calibrator, ref_focal_um, ref_px_detected, calib_report


# ============================================================
# 步骤3：待测光纤完整测量流程
# ============================================================


def run_measurement(calibrator, ref_focal_um, n_targets=None, rng=None,
                    verbose=True,
                    use_real_image=False,   # 新增：是否使用真实图像
                    real_image_path=None):  # 新增：真实图像路径

    """
    模拟待测光纤完整测量流程

    修复说明（坐标范围问题）：
    - 待测光纤焦面坐标范围严格限制在基准格网覆盖区域内（插值，不外推）
    - 使用 ref_focal_um 的实际范围动态计算边界，不依赖硬编码数值

    Parameters
    ----------
    calibrator   : 已标定的 FVCCalibrator
    ref_focal_um : 基准格网焦面坐标，用于确定插值边界
    """
    print("\n" + "=" * 50)
    print("步骤3：待测光纤测量精度评估")
    print("=" * 50)

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 1)
    if n_targets is None:
        n_targets = NUM_TARGET_FIBERS

    # ── 关键修复：待测点范围从基准格网实际覆盖范围动态计算 ──
    # 留10%边距确保待测点在格网内部（插值区域），不外推
    margin = 0.10
    focal_min = ref_focal_um.min(axis=0)
    focal_max = ref_focal_um.max(axis=0)
    focal_range = focal_max - focal_min
    target_low = focal_min + margin * focal_range
    target_high = focal_max - margin * focal_range

    print(f"  基准格网覆盖: X [{focal_min[0]/1000:.1f}, {focal_max[0]/1000:.1f}] mm, "
          f"Y [{focal_min[1]/1000:.1f}, {focal_max[1]/1000:.1f}] mm")
    print(f"  待测点范围:   X [{target_low[0]/1000:.1f}, {target_high[0]/1000:.1f}] mm, "
          f"Y [{target_low[1]/1000:.1f}, {target_high[1]/1000:.1f}] mm (留10%边距)")

    # 3.1 生成待测光纤焦面真值（在格网覆盖区域内随机分布）
    target_focal_true = rng.uniform(target_low, target_high, (n_targets, 2))

    # 3.2 焦面坐标 → 像素坐标（仿真真值）
    # 使用与标定完全相同的 focal_to_pixel 参数，保证坐标系一致性
    target_px_true = focal_to_pixel(target_focal_true)

    # 3.3 像素坐标检测
    print(f"  生成 {n_targets} 个待测光纤仿真光斑并检测...")
    target_px_detected = []
    detection_success = []
    method_counts = {'gaussian': 0, 'centroid': 0, 'failed': 0}
    t_start = time.time()

    if use_real_image and real_image_path is not None:
        # ══════════════════════════════════════════════════
        # 真实图像模式：粗检测 → 精确拟合
        # ══════════════════════════════════════════════════
        print(f"  [真实图像模式] 读取: {real_image_path}")

        # 读取图像（支持FITS和PNG/TIFF）
        if real_image_path.endswith('.fits') or real_image_path.endswith('.fit'):
            try:
                import astropy.io.fits as fits
                real_image = fits.getdata(real_image_path).astype(float)
            except ImportError:
                raise ImportError("读取FITS文件需要安装astropy: pip install astropy")
        else:
            from PIL import Image
            real_image = np.array(Image.open(real_image_path).convert('L'),
                                  dtype=float)

        print(f"  图像尺寸: {real_image.shape[1]}×{real_image.shape[0]} px")

        # 第一层：全图粗检测
        rough_positions = detect_all_spots(real_image,
                                           method='threshold',
                                           verbose=verbose)

        # 第二层：对每个候选位置精确拟合
        patch_half = 25
        for cx, cy in rough_positions:
            x0_int = int(round(cx))
            y0_int = int(round(cy))

            # 边界检查
            if (x0_int < patch_half or y0_int < patch_half or
                    x0_int + patch_half >= real_image.shape[1] or
                    y0_int + patch_half >= real_image.shape[0]):
                continue

            patch = real_image[y0_int - patch_half: y0_int + patch_half,
                               x0_int - patch_half: x0_int + patch_half]

            result = fit_with_fallback(patch)

            if result['success']:
                # patch内坐标 → 全图像素坐标
                full_x = result['x0'] + x0_int - patch_half
                full_y = result['y0'] + y0_int - patch_half
                target_px_detected.append([full_x, full_y])
                detection_success.append(True)
                method_counts[result.get('method', 'gaussian')] += 1
            else:
                method_counts['failed'] += 1

        # 真实图像模式下没有焦面真值，精度统计部分跳过
        target_px_detected = np.array(target_px_detected) \
            if target_px_detected else np.zeros((0, 2))
        target_focal_true = None  # 无真值，后续统计会跳过

    else:
        # ══════════════════════════════════════════════════
        # 仿真模式：生成光斑 → 高斯拟合（原有逻辑，完全不变）
        # ══════════════════════════════════════════════════
        for px, py in target_px_true:
            patch_size = 50
            true_cx = patch_size / 2 + (px - round(px))
            true_cy = patch_size / 2 + (py - round(py))
            patch = generate_gaussian_spot(true_cx, true_cy,
                                           image_size=patch_size, rng=rng)
            result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX)

            if result['success']:
                offset_x = round(px) - patch_size // 2
                offset_y = round(py) - patch_size // 2
                detected_x = result['x0'] + offset_x
                detected_y = result['y0'] + offset_y
                target_px_detected.append([detected_x, detected_y])
                detection_success.append(True)
                method_counts['gaussian'] += 1
            else:
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

    # 真实图像模式无精度真值，提前返回
    if use_real_image:
        print("  [真实图像模式] 检测完成，无焦面真值，跳过精度统计")
        return {'n_detected': n_success, 'mode': 'real_image'}, target_px_detected


    # 3.4 坐标变换：像素坐标 → 焦面坐标
    # 像素原点（左上角）通过标定矩阵映射到焦面坐标系
    target_focal_predicted = calibrator.transform(target_px_detected)

    # 3.5 精度统计（仅成功检测的点）
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

    # 3.6 输出图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 误差矢量图（焦面坐标，mm单位显示）
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

    # 误差分布圆图
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

    return stats, err


# ============================================================
# 步骤4：保存结果报告
# ============================================================

def save_report(detection_stats, calib_report, measurement_stats):
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
        },
        'detection_accuracy': detection_stats,
        'calibration': calib_report,
        'measurement_accuracy': measurement_stats,
        'overall_pass': measurement_stats.get('pass', False),
    }
    path = os.path.join(OUTPUT_DIR, "results", "accuracy_report.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存: {path}")
    return report


# ============================================================
# 主入口
# ============================================================

def main(mode='full'):
    ensure_output_dir()
    rng = np.random.default_rng(RANDOM_SEED)

    print("╔══════════════════════════════════════════════════╗")
    print("║   MUST望远镜 FVC 光纤位置测量系统               ║")
    print(f"║   目标精度: {TARGET_ACCURACY_UM} um    模式: {mode:<10}           ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  焦面尺度: {FOCAL_PLANE_SCALE_UM_PX:.2f} um/px")
    print(f"  精度换算: {TARGET_ACCURACY_UM} um = {TARGET_ACCURACY_PX:.4f} px")
    print(f"  基准格网: {REF_GRID_SIDE}x{REF_GRID_SIDE} = {REF_GRID_SIDE**2} 个基准点")
    print(f"  畸变模型: {DISTORTION_DEGREE} 阶多项式")

    detection_stats = {}
    calib_report = {}
    measurement_stats = {}
    calibrator = None
    ref_focal_um = None

    if mode in ('detection', 'full'):
        detection_stats, _, _ = run_detection_accuracy_test(n_trials=500)

    if mode in ('calibration', 'full'):
        calibrator, ref_focal_um, _, calib_report = run_calibration(rng)

    if mode == 'full':
        # 仿真模式（默认，不变）
        measurement_stats, _ = run_measurement(calibrator, ref_focal_um, rng=rng)

        # # 真实图像模式（接入实测数据时使用）
        # measurement_stats, detected_positions = run_measurement(
        #     calibrator,
        #     ref_focal_um,
        #     rng=rng,
        #     use_real_image=True,
        #     real_image_path="path/to/fvc_image.fits"     # 改成真实图像路径
        # )

        report = save_report(detection_stats, calib_report, measurement_stats)

        print("\n" + "=" * 50)
        print("最终结论")
        print("=" * 50)
        print(f"  高斯拟合质心精度:  {detection_stats.get('rms_um', 0):.3f} um")
        print(f"  标定残差:          {calib_report.get('final_rms_um', 0):.3f} um")
        print(f"  端对端测量精度:    {measurement_stats.get('rms_r_um', 0):.3f} um")
        overall = "✓ 系统达标" if measurement_stats.get('pass') else "✗ 系统未达标，需优化"
        print(f"  综合评估:          {overall}")

    print("\n完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MUST FVC 光纤位置测量系统')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'detection', 'calibration'],
                        help='运行模式: full=完整流程, detection=仅检测测试, calibration=仅标定')
    args = parser.parse_args()
    main(mode=args.mode)

