import numpy as np
import matplotlib.pyplot as plt
from main_pipeline import (
    FVCCalibrator, build_reference_grid, focal_to_pixel,
    generate_gaussian_spot, fit_gaussian, SPOT_SIGMA_PX,
    REFERENCE_GRID_SPACING_MM, ELLIPTICAL_SPOT_PROB
)

# ============================================================
# 适配函数：兼容不同的方法名
# ============================================================
def pixel_to_focal_adaptive(calibrator, px):
    """
    自动适配 FVCCalibrator 的方法名
    """
    if hasattr(calibrator, 'pixel_to_focal'):
        return calibrator.pixel_to_focal(px)
    elif hasattr(calibrator, 'transform'):
        return calibrator.transform(px)
    elif hasattr(calibrator, 'predict'):
        return calibrator.predict(px)
    else:
        # 手动计算
        X = calibrator.poly_features.transform(px)
        focal_x = calibrator.poly_x.predict(X)
        focal_y = calibrator.poly_y.predict(X)
        focal_xy = np.column_stack([focal_x, focal_y])
        # 减去偏差
        if hasattr(calibrator, 'bias_x') and hasattr(calibrator, 'bias_y'):
            focal_xy[:, 0] -= calibrator.bias_x
            focal_xy[:, 1] -= calibrator.bias_y
        return focal_xy


def test_noise_robustness():
    """测试不同噪声水平下的精度"""
    noise_levels = [0, 5, 10, 20, 50, 100]
    results = []
    
    rng = np.random.default_rng(42)
    ref_focal_um = build_reference_grid(n_side=7)
    
    for noise in noise_levels:
        print(f"\n测试噪声水平: {noise}")
        
        # 生成基准点（带噪声）
        ref_px_ideal = focal_to_pixel(ref_focal_um)
        ref_px_detected = []
        
        for px, py in ref_px_ideal:
            patch_size = 50
            true_cx = patch_size / 2 + (px - round(px))
            true_cy = patch_size / 2 + (py - round(py))
            patch = generate_gaussian_spot(true_cx, true_cy,
                                           image_size=patch_size, rng=rng,
                                           ellipticity_prob=0.3)
            # 添加噪声
            patch += rng.normal(0, noise, patch.shape)
            patch = np.clip(patch, 0, None)
            
            result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX, use_elliptical=True)
            if result['success']:
                offset_x = round(px) - patch_size // 2
                offset_y = round(py) - patch_size // 2
                detected_x = result['x0'] + offset_x
                detected_y = result['y0'] + offset_y
                ref_px_detected.append([detected_x, detected_y])
            else:
                ref_px_detected.append([px, py])
        
        ref_px_detected = np.array(ref_px_detected)
        
        # 标定
        calibrator = FVCCalibrator(poly_degree=4)
        report = calibrator.calibrate(ref_px_detected, ref_focal_um, verbose=False)
        
        # 生成测试点
        coverage_mm = 6 * REFERENCE_GRID_SPACING_MM
        test_focal_um = rng.uniform(
            [-coverage_mm/2*1000*0.9, -coverage_mm/2*1000*0.9],
            [coverage_mm/2*1000*0.9, coverage_mm/2*1000*0.9],
            size=(500, 2)
        )
        
        # 测试精度（带噪声）
        test_px_ideal = focal_to_pixel(test_focal_um)
        test_px_detected = []
        
        for px, py in test_px_ideal:
            patch_size = 50
            true_cx = patch_size / 2 + (px - round(px))
            true_cy = patch_size / 2 + (py - round(py))
            patch = generate_gaussian_spot(true_cx, true_cy,
                                           image_size=patch_size, rng=rng,
                                           ellipticity_prob=0.3)
            patch += rng.normal(0, noise, patch.shape)
            patch = np.clip(patch, 0, None)
            
            result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX, use_elliptical=True)
            if result['success']:
                offset_x = round(px) - patch_size // 2
                offset_y = round(py) - patch_size // 2
                detected_x = result['x0'] + offset_x
                detected_y = result['y0'] + offset_y
                test_px_detected.append([detected_x, detected_y])
            else:
                test_px_detected.append([px, py])
        
        test_px_detected = np.array(test_px_detected)
        
        # 使用适配函数
        test_focal_meas = pixel_to_focal_adaptive(calibrator, test_px_detected)
        
        errors = test_focal_um - test_focal_meas
        rmse = np.sqrt(np.mean(errors**2))
        
        results.append({
            'noise': noise,
            'calib_rmse': report['final_rms_um'],
            'test_rmse': rmse
        })
        print(f"  标定残差: {report['final_rms_um']:.3f} μm")
        print(f"  测试精度: {rmse:.3f} μm")
    
    # 绘制结果
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    noises = [r['noise'] for r in results]
    test_rmses = [r['test_rmse'] for r in results]
    
    plt.plot(noises, test_rmses, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=3.0, color='r', linestyle='--', label='目标精度 3.0 μm')
    plt.xlabel('背景噪声水平', fontsize=12)
    plt.ylabel('测试精度 RMSE (μm)', fontsize=12)
    plt.title('噪声鲁棒性测试', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/noise_robustness.png', dpi=150)
    print("\n噪声鲁棒性图已保存: outputs/figures/noise_robustness.png")
    
    return results


def test_ellipticity_robustness():
    """测试不同椭圆比例下的精度"""
    ellipticity_probs = [0, 0.2, 0.5, 0.8, 1.0]
    results = []
    
    rng = np.random.default_rng(42)
    ref_focal_um = build_reference_grid(n_side=7)
    
    for prob in ellipticity_probs:
        print(f"\n测试椭圆光斑比例: {prob*100:.0f}%")
        
        # 生成基准点
        ref_px_ideal = focal_to_pixel(ref_focal_um)
        ref_px_detected = []
        
        for px, py in ref_px_ideal:
            patch_size = 50
            true_cx = patch_size / 2 + (px - round(px))
            true_cy = patch_size / 2 + (py - round(py))
            patch = generate_gaussian_spot(true_cx, true_cy,
                                           image_size=patch_size, rng=rng,
                                           ellipticity_prob=prob)
            result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX, use_elliptical=True)
            if result['success']:
                offset_x = round(px) - patch_size // 2
                offset_y = round(py) - patch_size // 2
                detected_x = result['x0'] + offset_x
                detected_y = result['y0'] + offset_y
                ref_px_detected.append([detected_x, detected_y])
            else:
                ref_px_detected.append([px, py])
        
        ref_px_detected = np.array(ref_px_detected)
        
        # 标定
        calibrator = FVCCalibrator(poly_degree=4)
        report = calibrator.calibrate(ref_px_detected, ref_focal_um, verbose=False)
        
        # 测试精度
        coverage_mm = 6 * REFERENCE_GRID_SPACING_MM
        test_focal_um = rng.uniform(
            [-coverage_mm/2*1000*0.9, -coverage_mm/2*1000*0.9],
            [coverage_mm/2*1000*0.9, coverage_mm/2*1000*0.9],
            size=(500, 2)
        )
        
        test_px_ideal = focal_to_pixel(test_focal_um)
        test_px_detected = []
        
        for px, py in test_px_ideal:
            patch_size = 50
            true_cx = patch_size / 2 + (px - round(px))
            true_cy = patch_size / 2 + (py - round(py))
            patch = generate_gaussian_spot(true_cx, true_cy,
                                           image_size=patch_size, rng=rng,
                                           ellipticity_prob=prob)
            result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX, use_elliptical=True)
            if result['success']:
                offset_x = round(px) - patch_size // 2
                offset_y = round(py) - patch_size // 2
                detected_x = result['x0'] + offset_x
                detected_y = result['y0'] + offset_y
                test_px_detected.append([detected_x, detected_y])
            else:
                test_px_detected.append([px, py])

        test_px_detected = np.array(test_px_detected)

        # 使用适配函数
        test_focal_meas = pixel_to_focal_adaptive(calibrator, test_px_detected)

        errors = test_focal_um - test_focal_meas
        rmse = np.sqrt(np.mean(errors ** 2))

        results.append({
            'ellipticity_prob': prob,
            'calib_rmse': report['final_rms_um'],
            'test_rmse': rmse
        })
        print(f"  标定残差: {report['final_rms_um']:.3f} μm")
        print(f"  测试精度: {rmse:.3f} μm")

    # 绘制结果
    plt.figure(figsize=(10, 5))
    probs = [r['ellipticity_prob'] * 100 for r in results]
    test_rmses = [r['test_rmse'] for r in results]

    plt.plot(probs, test_rmses, 'o-', linewidth=2, markersize=8, color='green')
    plt.axhline(y=3.0, color='r', linestyle='--', label='目标精度 3.0 μm')
    plt.xlabel('椭圆光斑比例 (%)', fontsize=12)
    plt.ylabel('测试精度 RMSE (μm)', fontsize=12)
    plt.title('椭圆光斑鲁棒性测试', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/ellipticity_robustness.png', dpi=150)
    print("\n椭圆鲁棒性图已保存: outputs/figures/ellipticity_robustness.png")

    return results


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("系统鲁棒性测试")
    print("=" * 60)

    # 测试1：噪声鲁棒性
    print("\n[测试1] 噪声鲁棒性")
    noise_results = test_noise_robustness()

    # 测试2：椭圆光斑鲁棒性
    print("\n[测试2] 椭圆光斑鲁棒性")
    ellipticity_results = test_ellipticity_robustness()

    # 汇总报告
    print("\n" + "=" * 60)
    print("鲁棒性测试汇总")
    print("=" * 60)

    print("\n噪声鲁棒性：")
    for r in noise_results:
        status = "✓" if r['test_rmse'] < 3.0 else "✗"
        print(f"  噪声 {r['noise']:3d}:  {r['test_rmse']:.3f} μm  {status}")

    print("\n椭圆光斑鲁棒性：")
    for r in ellipticity_results:
        status = "✓" if r['test_rmse'] < 3.0 else "✗"
        print(f"  椭圆 {r['ellipticity_prob'] * 100:3.0f}%:  {r['test_rmse']:.3f} μm  {status}")

    print("\n完成！")