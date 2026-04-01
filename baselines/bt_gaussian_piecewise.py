"""
baselines/bt_gaussian_piecewise.py
Baseline 4: 2D Gaussian + 分块局部多项式
将焦面分成多个区域，每个区域独立拟合低阶多项式
使用距离加权平均多个区域的预测，避免边界不连续
"""

import os
import sys
import json
import numpy as np
from skimage.transform import PolynomialTransform

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from evaluation.metrics import calculate_errors


def assign_to_regions(points, x_min, x_max, y_min, y_max, n_regions_x=2, n_regions_y=2):
    """将点分配到网格区域"""
    x_bins = np.linspace(x_min, x_max, n_regions_x + 1)
    y_bins = np.linspace(y_min, y_max, n_regions_y + 1)

    region_ids = []
    for x, y in points:
        i = np.searchsorted(x_bins, x, side='right') - 1
        j = np.searchsorted(y_bins, y, side='right') - 1
        i = np.clip(i, 0, n_regions_x - 1)
        j = np.clip(j, 0, n_regions_y - 1)
        region_ids.append(i * n_regions_y + j)

    return np.array(region_ids)


def get_region_center(region_id, src, calib_regions, x_min, x_max, y_min, y_max, n_regions_x, n_regions_y):
    """获取区域中心坐标"""
    mask = (calib_regions == region_id)
    if mask.sum() > 0:
        return src[mask].mean(axis=0)
    else:
        # 如果区域没有点，用网格中心
        i_grid = region_id // n_regions_y
        j_grid = region_id % n_regions_y
        return np.array([
            x_min + (i_grid + 0.5) * (x_max - x_min) / n_regions_x,
            y_min + (j_grid + 0.5) * (y_max - y_min) / n_regions_y
        ])


def run_piecewise_baseline(image_path, label_path, n_regions_x=2, n_regions_y=2,
                           order=2, max_det_error_px=1.0, use_weighted_avg=True):
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
    test_px = np.array(matched_target_px, dtype=float)
    true_target_mm = np.array(matched_target_mm, dtype=float)

    if len(src) < 6:
        raise RuntimeError(f"基准点过少：{len(src)}")

    # 计算全局范围（用于分区）
    all_px = np.vstack([src, test_px])
    x_min, y_min = all_px.min(axis=0)
    x_max, y_max = all_px.max(axis=0)

    # 分配区域
    calib_regions = assign_to_regions(src, x_min, x_max, y_min, y_max, n_regions_x, n_regions_y)

    # 每个区域独立拟合
    n_regions = n_regions_x * n_regions_y
    transforms = {}
    region_stats = {}

    min_points = (order + 1) * (order + 2) // 2

    for region_id in range(n_regions):
        mask = (calib_regions == region_id)
        region_src = src[mask]
        region_dst = dst[mask]

        fallback_type = False

        if len(region_src) < min_points:
            # 方案 2：扩展到相邻区域
            expanded_mask = np.zeros(len(src), dtype=bool)
            for i, rid in enumerate(calib_regions):
                # 如果在当前区域或相邻区域
                i_curr = region_id // n_regions_y
                j_curr = region_id % n_regions_y
                i_other = rid // n_regions_y
                j_other = rid % n_regions_y

                if abs(i_curr - i_other) <= 1 and abs(j_curr - j_other) <= 1:
                    expanded_mask[i] = True

            region_src = src[expanded_mask]
            region_dst = dst[expanded_mask]

            if len(region_src) < min_points:
                # 还是不够，用全局
                region_src = src
                region_dst = dst
                fallback_type = 'global'
            else:
                fallback_type = 'neighbor'

        region_stats[region_id] = {
            'n_calib': len(region_src),
            'fallback': fallback_type
        }

        tform = PolynomialTransform()
        tform.estimate(region_src, region_dst, order=order)
        transforms[region_id] = tform

    # 方案 3：加权平均相邻区域的预测
    pred_mm = np.zeros_like(true_target_mm)

    if use_weighted_avg:
        for i, test_point in enumerate(test_px):
            # 计算到每个区域中心的距离
            weights = []
            predictions = []

            for region_id in range(n_regions):
                region_center = get_region_center(
                    region_id, src, calib_regions,
                    x_min, x_max, y_min, y_max,
                    n_regions_x, n_regions_y
                )

                # 距离加权
                dist = np.linalg.norm(test_point - region_center)
                weight = 1.0 / (dist + 1e-6)

                tform = transforms[region_id]
                pred = tform(test_point.reshape(1, -1))[0]

                weights.append(weight)
                predictions.append(pred)

            # 加权平均
            weights = np.array(weights)
            weights /= weights.sum()
            pred_mm[i] = np.average(predictions, axis=0, weights=weights)
    else:
        # 不使用加权平均，直接用所在区域的变换
        test_regions = assign_to_regions(test_px, x_min, x_max, y_min, y_max, n_regions_x, n_regions_y)
        for i, region_id in enumerate(test_regions):
            tform = transforms[region_id]
            pred_mm[i] = tform(test_px[i:i+1])[0]

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

    # 统计 fallback 类型
    n_fallback_global = sum(1 for s in region_stats.values() if s['fallback'] == 'global')
    n_fallback_neighbor = sum(1 for s in region_stats.values() if s['fallback'] == 'neighbor')

    return {
        "centroid_rmse_px": centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "success_rate_all": success_rate_all,
        "success_rate_calib": success_rate_calib,
        "success_rate_target": success_rate_target,
        "calib_used": len(src),
        "target_tested": len(test_px),
        "n_regions_x": n_regions_x,
        "n_regions_y": n_regions_y,
        "order": order,
        "n_fallback_global": n_fallback_global,
        "n_fallback_neighbor": n_fallback_neighbor,
        "use_weighted_avg": use_weighted_avg,
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
    print(f"[Config] 分块策略: 2x2 网格, 每块 2 阶多项式, 加权平均")

    all_results = []

    for sample_name in sample_names:
        img_path = os.path.join(base_path, "dataset", "images", f"{sample_name}.npy")
        lbl_path = os.path.join(base_path, "dataset", "labels", f"{sample_name}.json")

        print(f"\n--- 正在测试 Baseline 4 (Piecewise Poly): {sample_name} ---")
        try:
            res = run_piecewise_baseline(img_path, lbl_path,
                                         n_regions_x=2,
                                         n_regions_y=2,
                                         order=2,
                                         max_det_error_px=1.0,
                                         use_weighted_avg=True)

            print(f"总检测率: {res['success_rate_all'] * 100:.1f}%")
            print(f"基准光纤检测率: {res['success_rate_calib'] * 100:.1f}%")
            print(f"待测光纤检测率: {res['success_rate_target'] * 100:.1f}%")
            print(
                f"使用基准点: {res['calib_used']} (分块: {res['n_regions_x']}x{res['n_regions_y']}, 阶数: {res['order']})")
            print(f"测试目标点: {res['target_tested']}")
            print(f"Fallback: global={res['n_fallback_global']}, neighbor={res['n_fallback_neighbor']}")
            print(f"加权平均: {res['use_weighted_avg']}")
            print(f"未通过检测/门控的基准光纤: {res['failed_calib_count']}")
            print(f"未通过检测/门控的待测光纤: {res['failed_target_count']}")
            print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px")
            print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")

            all_results.append((sample_name, res))

        except Exception as e:
            print(f"运行出错: {e}")
            import traceback

            traceback.print_exc()

    print("\n================ 汇总结果 ================")
    for sample_name, res in all_results:
        print(
            f"{sample_name:12s} | "
            f"all={res['success_rate_all'] * 100:5.1f}% | "
            f"calib={res['success_rate_calib'] * 100:5.1f}% | "
            f"target={res['success_rate_target'] * 100:5.1f}% | "
            f"used={res['calib_used']:2d} | "
            f"test={res['target_tested']:2d} | "
            f"grid={res['n_regions_x']}x{res['n_regions_y']} | "
            f"order={res['order']} | "
            f"fb_g={res['n_fallback_global']} | "
            f"fb_n={res['n_fallback_neighbor']} | "
            f"weighted={res['use_weighted_avg']} | "
            f"centroid={res['centroid_rmse_px']:.4f} px | "
            f"transform={res['transform_rmse_um']:.2f} μm"
        )