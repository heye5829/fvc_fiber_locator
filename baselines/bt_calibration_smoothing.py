"""
baselines/bt_calibration_smoothing.py
Baseline 6: 2D Gaussian + 带有 L2 正则化的全局多项式 (Ridge Regression)
通过正则化项抑制高阶多项式的系数，实现坐标标定过程中的平滑，抗噪性能强。
"""

import os
import sys
import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from evaluation.metrics import calculate_errors


def run_ridge_poly_baseline(image_path, label_path, order=3, alpha=1.0, max_det_error_px=1.0):
    """
    带有 L2 正则化的多项式回归标定方法

    参数:
        order: 多项式阶数
        alpha: 正则化强度 (L2 惩罚项权重)
    """
    image = np.load(image_path).astype(np.float32)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fiber_data = label["fibers"]
    detector = GaussianDetector()
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    results_list, _ = detector.detect_all(image, seed_positions)

    matched_calib_px, matched_calib_mm = [], []
    matched_target_px, matched_target_mm = [], []
    valid_indices, failed_calib, failed_target = [], [], []

    calib_total = sum(1 for f in fiber_data if f["is_calib"])
    target_total = sum(1 for f in fiber_data if not f["is_calib"])
    calib_ok = target_ok = 0

    for i, res in enumerate(results_list):
        det_x, det_y = res.get('x_global', np.nan), res.get('y_global', np.nan)
        true_x, true_y = fiber_data[i]["true_x_px"], fiber_data[i]["true_y_px"]
        is_calib = fiber_data[i]["is_calib"]

        ok = (res.get('success', False) and np.isfinite(det_x) and np.isfinite(det_y)
              and np.hypot(det_x - true_x, det_y - true_y) < max_det_error_px)

        if not ok:
            if is_calib:
                failed_calib.append(i)
            else:
                failed_target.append(i)
            continue

        valid_indices.append(i)
        if is_calib:
            calib_ok += 1
            matched_calib_px.append([det_x, det_y])
            matched_calib_mm.append([fiber_data[i]["true_x_mm"], fiber_data[i]["true_y_mm"]])
        else:
            target_ok += 1
            matched_target_px.append([det_x, det_y])
            matched_target_mm.append([fiber_data[i]["true_x_mm"], fiber_data[i]["true_y_mm"]])

    success_rate_all = len(valid_indices) / len(fiber_data)
    src, dst = np.array(matched_calib_px), np.array(matched_calib_mm)
    test_px, true_target_mm = np.array(matched_target_px), np.array(matched_target_mm)

    if len(src) < (order + 1) * (order + 2) // 2:
        # 如果点数少于特征数，Ridge 仍然可以运行，但给出警告
        pass

        # 构建带正则化的多项式流水线: 特征工程 -> 标准化 -> Ridge回归

    # 标准化 (StandardScaler) 对正则化回归至关重要
    def create_ridge_model(degree, l2_alpha):
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=True)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=l2_alpha))
        ])

    model_x = create_ridge_model(order, alpha)
    model_y = create_ridge_model(order, alpha)

    # 分别训练 x 和 y 方向的标定映射
    model_x.fit(src, dst[:, 0])
    model_y.fit(src, dst[:, 1])

    # 预测并评估
    pred_x = model_x.predict(test_px)
    pred_y = model_y.predict(test_px)
    pred_mm = np.column_stack([pred_x, pred_y])

    transform_err = calculate_errors(true_target_mm * 1000.0, pred_mm * 1000.0)

    # 质心精度计算
    det_success_px = np.array([[results_list[i]['x_global'], results_list[i]['y_global']] for i in valid_indices])
    true_success_px = np.array([[fiber_data[i]['true_x_px'], fiber_data[i]['true_y_px']] for i in valid_indices])
    centroid_err = calculate_errors(true_success_px, det_success_px)

    return {
        "centroid_rmse_px": centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "success_rate_all": success_rate_all,
        "calib_used": len(src),
        "target_tested": len(test_px),
        "order": order,
        "alpha": alpha,
        "failed_calib_count": len(failed_calib),
        "failed_target_count": len(failed_target),
    }


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_names = ["sample_00900", "sample_00532", "sample_00608"]

    print(f"[Config] 焦面尺度: 139.12 μm/px | 目标精度: 3.0 μm")
    print(f"[Config] 模型策略: Ridge Polynomial Regression (L2 Smoothing)")

    # 参数扫描：寻找最佳的正则化强度 alpha
    orders = [3, 4]
    alphas = [0.01, 0.1, 1.0, 10.0]

    all_results = []
    for order in orders:
        for alpha in alphas:
            print(f"\nTesting: Order={order}, Alpha={alpha}")
            for sample_name in sample_names:
                img_path = os.path.join(base_path, "dataset", "images", f"{sample_name}.npy")
                lbl_path = os.path.join(base_path, "dataset", "labels", f"{sample_name}.json")
                try:
                    res = run_ridge_poly_baseline(img_path, lbl_path, order=order, alpha=alpha)
                    all_results.append((sample_name, order, alpha, res))
                    print(f"  {sample_name}: Transform RMSE = {res['transform_rmse_um']:.2f} μm")
                except Exception as e:
                    print(f"  {sample_name} Error: {e}")

    # 汇总显示
    print("\n" + "=" * 85)
    print(f"{'Sample':<12} | {'Order':>5} | {'Alpha':>8} | {'RMSE(μm)':>10} | {'Centroid(px)':>12}")
    print("-" * 85)
    for sn, od, al, rs in all_results:
        print(f"{sn:<12} | {od:>5} | {al:>8.2f} | {rs['transform_rmse_um']:>10.2f} | {rs['centroid_rmse_px']:>12.4f}")