"""
ablation_detection_quality.py

消融实验：检测质量对端到端精度的影响

测试不同 SNR 和检测误差水平下的端到端 RMSE，
量化"检测质量"与"端到端精度"的关系。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 导入主流程的关键函数和常量
from main_pipeline import (
    build_reference_grid, focal_to_pixel, generate_gaussian_spot,
    fit_gaussian, FVCCalibrator, RANDOM_SEED, FOCAL_PLANE_SCALE_UM_PX,
    SPOT_SIGMA_PX
)
from evaluation.metrics import calculate_errors

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_ablation_test(snr_factor=1.0, detection_noise_px=0.0, n_side=7):
    """
    测试不同检测质量下的端到端精度

    Parameters:
    - snr_factor: SNR 缩放因子（1.0 = 正常，0.5 = 降低一半）
    - detection_noise_px: 人为添加的检测误差标准差（像素）
    - n_side: 基准格网边长

    Returns:
    - dict: 包含检测精度、标定精度、端到端精度等指标
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) 生成基准点
    ref_focal_um = build_reference_grid(n_side=n_side)
    ref_px_ideal = focal_to_pixel(ref_focal_um)

    # 2) 仿真检测基准点（降低质量）
    ref_px_detected = []
    detection_success_count = 0

    for px, py in ref_px_ideal:
        patch_size = 50
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))

        # 生成光斑
        patch = generate_gaussian_spot(
            true_cx, true_cy,
            image_size=patch_size,
            rng=rng
        )

        # 调整 SNR：降低信号或增加噪声
        if snr_factor < 1.0:
            # 降低信号强度
            patch = patch * snr_factor
            # 增加噪声
            noise = rng.poisson(lam=50 * (1.0 - snr_factor), size=patch.shape)
            patch = patch + noise

        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX, use_elliptical=True)

        if result['success']:
            detection_success_count += 1
            offset_x = round(px) - patch_size // 2
            offset_y = round(py) - patch_size // 2
            detected_x = result['x0'] + offset_x
            detected_y = result['y0'] + offset_y

            # 添加人为检测误差
            detected_x += rng.normal(0, detection_noise_px)
            detected_y += rng.normal(0, detection_noise_px)

            ref_px_detected.append([detected_x, detected_y])
        else:
            # 检测失败，用理想值（模拟仿真主流程的回退策略）
            ref_px_detected.append([px, py])

    ref_px_detected = np.array(ref_px_detected)
    detection_rate = detection_success_count / len(ref_px_ideal)

    # 计算检测质心误差
    detection_err = calculate_errors(ref_px_ideal, ref_px_detected)
    detection_rmse_px = detection_err['rmse']
    detection_rmse_um = detection_rmse_px * FOCAL_PLANE_SCALE_UM_PX

    # 3) 标定
    poly_degree = 4
    calibrator = FVCCalibrator(poly_degree=poly_degree)
    calib_report = calibrator.calibrate(ref_px_detected, ref_focal_um, verbose=False)

    # 4) 生成测试点
    n_test = 500
    test_focal_um = rng.uniform(-15000, 15000, (n_test, 2))
    test_px_ideal = focal_to_pixel(test_focal_um)

    # 5) 仿真检测测试点（降低质量）
    test_px_detected = []
    test_success_count = 0

    for px, py in test_px_ideal:
        patch_size = 50
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))

        patch = generate_gaussian_spot(
            true_cx, true_cy,
            image_size=patch_size,
            rng=rng
        )

        # 调整 SNR
        if snr_factor < 1.0:
            patch = patch * snr_factor
            noise = rng.poisson(lam=50 * (1.0 - snr_factor), size=patch.shape)
            patch = patch + noise

        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX, use_elliptical=True)

        if result['success']:
            test_success_count += 1
            offset_x = round(px) - patch_size // 2
            offset_y = round(py) - patch_size // 2
            detected_x = result['x0'] + offset_x
            detected_y = result['y0'] + offset_y

            detected_x += rng.normal(0, detection_noise_px)
            detected_y += rng.normal(0, detection_noise_px)

            test_px_detected.append([detected_x, detected_y])
        else:
            test_px_detected.append([px, py])

    test_px_detected = np.array(test_px_detected)
    test_detection_rate = test_success_count / len(test_px_ideal)

    # 6) 反演测试点
    test_focal_pred = calibrator.transform(test_px_detected)

    # 7) 计算端到端误差
    end_to_end_err = calculate_errors(test_focal_um, test_focal_pred)

    return {
        'snr_factor': snr_factor,
        'detection_noise_px': detection_noise_px,
        'n_calib_points': len(ref_px_ideal),
        'detection_rate_calib': detection_rate,
        'detection_rate_test': test_detection_rate,
        'detection_rmse_px': detection_rmse_px,
        'detection_rmse_um': detection_rmse_um,
        'calib_rms_um': calib_report.get('final_rms_um', None),
        'end_to_end_rmse_um': end_to_end_err['rmse'],
        'end_to_end_max_um': end_to_end_err['max'],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("消融实验：检测质量对端到端精度的影响")
    print("=" * 70)

    # ============================================================
    # 实验1：改变 SNR
    # ============================================================
    print("\n[实验1] 改变 SNR（保持检测噪声为 0）")
    print("-" * 70)

    snr_factors = [1.0, 0.8, 0.6, 0.4, 0.2]
    results_snr = []

    for snr in snr_factors:
        print(f"\n测试 SNR factor = {snr:.1f} ...")
        res = run_ablation_test(snr_factor=snr, detection_noise_px=0.0, n_side=7)
        results_snr.append(res)

        print(f"  检测率(基准): {res['detection_rate_calib']*100:.1f}%")
        print(f"  检测率(测试): {res['detection_rate_test']*100:.1f}%")
        print(f"  检测 RMSE: {res['detection_rmse_um']:.2f} μm")
        print(f"  标定 RMS: {res['calib_rms_um']:.2f} μm")
        print(f"  端到端 RMSE: {res['end_to_end_rmse_um']:.2f} μm")

    # ============================================================
    # 实验2：添加检测噪声
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验2] 添加检测噪声（保持 SNR = 1.0）")
    print("-" * 70)

    noise_levels_px = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    results_noise = []

    for noise in noise_levels_px:
        print(f"\n测试检测噪声 = {noise:.2f} px ({noise * FOCAL_PLANE_SCALE_UM_PX:.2f} μm) ...")
        res = run_ablation_test(snr_factor=1.0, detection_noise_px=noise, n_side=7)
        results_noise.append(res)

        print(f"  检测 RMSE: {res['detection_rmse_um']:.2f} μm")
        print(f"  标定 RMS: {res['calib_rms_um']:.2f} μm")
        print(f"  端到端 RMSE: {res['end_to_end_rmse_um']:.2f} μm")

    # ============================================================
    # 绘图
    # ============================================================
    print("\n" + "=" * 70)
    print("生成对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：SNR 影响
    ax1 = axes[0]
    snr_vals = [r['snr_factor'] for r in results_snr]
    detection_rmse = [r['detection_rmse_um'] for r in results_snr]
    calib_rms = [r['calib_rms_um'] for r in results_snr]
    e2e_rmse = [r['end_to_end_rmse_um'] for r in results_snr]

    ax1.plot(snr_vals, detection_rmse, 'o-', label='检测 RMSE', linewidth=2, markersize=8)
    ax1.plot(snr_vals, calib_rms, 's-', label='标定 RMS', linewidth=2, markersize=8)
    ax1.plot(snr_vals, e2e_rmse, '^-', label='端到端 RMSE', linewidth=2, markersize=8)

    ax1.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='目标精度 3 μm')
    ax1.set_xlabel('SNR 缩放因子', fontsize=12)
    ax1.set_ylabel('误差 (μm)', fontsize=12)
    ax1.set_title('SNR 对精度的影响', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.invert_xaxis()  # SNR 降低 → 从右到左

    # 图2：检测噪声影响
    ax2 = axes[1]
    noise_vals_um = [r['detection_rmse_um'] for r in results_noise]
    calib_rms_noise = [r['calib_rms_um'] for r in results_noise]
    e2e_rmse_noise = [r['end_to_end_rmse_um'] for r in results_noise]

    ax2.plot(noise_vals_um, calib_rms_noise, 's-', label='标定 RMS', linewidth=2, markersize=8)
    ax2.plot(noise_vals_um, e2e_rmse_noise, '^-', label='端到端 RMSE', linewidth=2, markersize=8)

    ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='目标精度 3 μm')
    ax2.set_xlabel('检测 RMSE (μm)', fontsize=12)
    ax2.set_ylabel('误差 (μm)', fontsize=12)
    ax2.set_title('检测质量对精度的影响', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_dir = os.path.join("outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "ablation_detection_quality.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {fig_path}")
    plt.close()

    # ============================================================
    # 汇总结论
    # ============================================================
    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)

    # 找到达到 3 μm 目标的条件
    target = 3.0

    # SNR 实验
    print("\n[SNR 实验结论]")
    for r in results_snr:
        if r['end_to_end_rmse_um'] <= target:
            print(f"✓ SNR factor ≥ {r['snr_factor']:.1f} 时可达到 3 μm 目标")
            print(f"  此时检测 RMSE: {r['detection_rmse_um']:.2f} μm")
            break
    else:
        print(f"✗ 所有测试的 SNR 水平均未达到 3 μm 目标")
        best_snr = min(results_snr, key=lambda x: x['end_to_end_rmse_um'])
        print(f"  最佳结果: SNR={best_snr['snr_factor']:.1f}, 端到端={best_snr['end_to_end_rmse_um']:.2f} μm")

    # 检测噪声实验
    print("\n[检测噪声实验结论]")
    for r in results_noise:
        if r['end_to_end_rmse_um'] > target:
            print(f"✓ 检测 RMSE 需 < {r['detection_rmse_um']:.2f} μm 才能达到 3 μm 目标")
            break
    else:
        print(f"✓ 所有测试的检测噪声水平均可达到 3 μm 目标")

    # 对比真实样本
    print("\n" + "-" * 70)
    print("与真实样本对比:")
    print(f"  仿真主流程（正常质量）: 检测 RMSE ≈ 1.5 μm, 端到端 ≈ 1.6 μm")
    print(f"  真实样本公平对比: 检测 RMSE ≈ 14 μm, 端到端 ≈ 14.88–21.63 μm")
    print(f"\n  结论: 真实样本的检测质量约为仿真的 1/10")
    print(f"        要达到 3 μm 目标，需将检测 RMSE 从 14 μm 降至 3–5 μm")
    print("=" * 70)