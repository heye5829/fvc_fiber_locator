"""
baselines/bt_weighted_rbf.py
Baseline 3: 加权径向基函数插值 (Weighted RBF)
使用 Thin Plate Spline 核函数
"""

import os
import sys
import json
import numpy as np
from scipy.interpolate import RBFInterpolator

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from evaluation.metrics import calculate_errors


def run_rbf_baseline(image_path, label_path, kernel='thin_plate_spline',
                     smoothing=0.0, max_det_error_px=1.0):
    image = np.load(image_path).astype(np.float32)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fiber_data = label["fibers"]
    detector = GaussianDetector()
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    results_list, _ = detector.detect_all(image, seed_positions)

    matched_calib_px = []
    matched_calib_mm = []
    matched_target_px = []
    matched_target_mm = []

    valid_indices = []
    failed_calib = []
    failed_target = []

    calib_total = sum(1 for f in fiber_data if f["is_calib"])
    target_total = sum(1 for f in fiber_data if not f["is_calib"])
    calib_ok = 0
    target_ok = 0

    for i, res in enumerate(results_list):
        det_x = res.get('x_global', np.nan)
        det_y = res.get('y_global', np.nan)
        true_x = fiber_data[i]["true_x_px"]
        true_y = fiber_data[i]["true_y_px"]
        is_calib = fiber_data[i]["is_calib"]

        ok = (
            res.get('success', False)
            and np.isfinite(det_x)
            and np.isfinite(det_y)
            and np.hypot(det_x - true_x, det_y - true_y) < max_det_error_px
        )

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
            matched_calib_mm.append([
                fiber_data[i]["true_x_mm"],
                fiber_data[i]["true_y_mm"]
            ])
        else:
            target_ok += 1
            matched_target_px.append([det_x, det_y])
            matched_target_mm.append([
                fiber_data[i]["true_x_mm"],
                fiber_data[i]["true_y_mm"]
            ])

    success_rate_all = len(valid_indices) / len(fiber_data)
    success_rate_calib = calib_ok / calib_total if calib_total > 0 else 0.0
    success_rate_target = target_ok / target_total if target_total > 0 else 0.0

    src = np.array(matched_calib_px, dtype=float)
    dst = np.array(matched_calib_mm, dtype=float)

    if len(src) < 3:
        raise RuntimeError(f"可用基准点过少：{len(src)}")

    rbf_x = RBFInterpolator(src, dst[:, 0], kernel=kernel, smoothing=smoothing)
    rbf_y = RBFInterpolator(src, dst[:, 1], kernel=kernel, smoothing=smoothing)

    test_px = np.array(matched_target_px, dtype=float)
    true_target_mm = np.array(matched_target_mm, dtype=float)

    if len(test_px) == 0:
        raise RuntimeError("没有可用于评估的待测光纤点")

    pred_x = rbf_x(test_px)
    pred_y = rbf_y(test_px)
    pred_mm = np.column_stack([pred_x, pred_y])

    transform_err = calculate_errors(true_target_mm * 1000.0, pred_mm * 1000.0)

    det_success_px = np.array(
        [[results_list[i]['x_global'], results_list[i]['y_global']] for i in valid_indices],
        dtype=float
    )
    true_success_px = np.array(
        [[fiber_data[i]['true_x_px'], fiber_data[i]['true_y_px']] for i in valid_indices],
        dtype=float
    )
    centroid_err = calculate_errors(true_success_px, det_success_px)

    return {
        "centroid_rmse_px": centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "success_rate_all": success_rate_all,
        "success_rate_calib": success_rate_calib,
        "success_rate_target": success_rate_target,
        "calib_used": len(src),
        "target_tested": len(test_px),
        "kernel": kernel,
        "smoothing": smoothing,
        "failed_calib_count": len(failed_calib),
        "failed_target_count": len(failed_target),
    }


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sample_names = [
        "sample_00900",
        "sample_00532",
        "sample_00608",
    ]

    print(f"[Config] 焦面尺度: 139.12 μm/px")
    print(f"[Config] 目标精度: 3.0 μm = 0.0216 px")
    print(f"[Config] RBF kernel: thin_plate_spline")

    all_results = []

    for sample_name in sample_names:
        img_path = os.path.join(base_path, "dataset", "images", f"{sample_name}.npy")
        lbl_path = os.path.join(base_path, "dataset", "labels", f"{sample_name}.json")

        print(f"\n--- 正在测试 Baseline 3 (RBF): {sample_name} ---")
        try:
            res = run_rbf_baseline(img_path, lbl_path,
                                   kernel='thin_plate_spline',
                                   smoothing=0.0,
                                   max_det_error_px=1.0)

            print(f"总检测率: {res['success_rate_all']*100:.1f}%")
            print(f"基准光纤检测率: {res['success_rate_calib']*100:.1f}%")
            print(f"待测光纤检测率: {res['success_rate_target']*100:.1f}%")
            print(f"使用基准点: {res['calib_used']} (kernel: {res['kernel']})")
            print(f"测试目标点: {res['target_tested']}")
            print(f"未通过检测/门控的基准光纤: {res['failed_calib_count']}")
            print(f"未通过检测/门控的待测光纤: {res['failed_target_count']}")
            print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px")
            print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")

            all_results.append((sample_name, res))

        except Exception as e:
            print(f"运行出错: {e}")

    print("\n================ 汇总结果 ================")
    for sample_name, res in all_results:
        print(
            f"{sample_name:12s} | "
            f"all={res['success_rate_all']*100:5.1f}% | "
            f"calib={res['success_rate_calib']*100:5.1f}% | "
            f"target={res['success_rate_target']*100:5.1f}% | "
            f"used={res['calib_used']:2d} | "
            f"test={res['target_tested']:2d} | "
            f"kernel={res['kernel']:20s} | "
            f"centroid={res['centroid_rmse_px']:.4f} px | "
            f"transform={res['transform_rmse_um']:.2f} μm"
        )