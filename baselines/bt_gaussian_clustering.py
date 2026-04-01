"""
baselines/bt_gaussian_clustering.py
Baseline 5: 2D Gaussian + 基于聚类的分组多项式
根据基准点的空间分布自动聚类，每个聚类独立拟合多项式
相比固定网格分块，聚类能更好地适应点的实际分布
"""

import os
import sys
import json
import numpy as np
from sklearn.cluster import KMeans
from skimage.transform import PolynomialTransform

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from evaluation.metrics import calculate_errors


def run_clustering_baseline(image_path, label_path, n_clusters=4, order=2,
                            max_det_error_px=1.0, use_weighted_avg=True):
    """
    基于聚类的分组多项式方法

    参数:
        n_clusters: 聚类数量（类似于分块数）
        order: 多项式阶数
        use_weighted_avg: 是否使用距离加权平均多个聚类的预测
    """
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

    # 自适应调整聚类数：不能超过基准点数量
    min_points_per_cluster = (order + 1) * (order + 2) // 2
    max_possible_clusters = len(src) // min_points_per_cluster
    actual_n_clusters = min(n_clusters, max_possible_clusters, len(src))

    if actual_n_clusters < 2:
        # 点太少，退化为全局方法
        actual_n_clusters = 1

    # 对基准点进行 K-Means 聚类
    if actual_n_clusters == 1:
        calib_labels = np.zeros(len(src), dtype=int)
        cluster_centers = src.mean(axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        calib_labels = kmeans.fit_predict(src)
        cluster_centers = kmeans.cluster_centers_

    # 每个聚类独立拟合多项式
    transforms = {}
    cluster_stats = {}

    for cluster_id in range(actual_n_clusters):
        mask = (calib_labels == cluster_id)
        cluster_src = src[mask]
        cluster_dst = dst[mask]

        fallback = False
        if len(cluster_src) < min_points_per_cluster:
            # 点太少，用全局数据
            cluster_src = src
            cluster_dst = dst
            fallback = True

        cluster_stats[cluster_id] = {
            'n_calib': len(cluster_src),
            'fallback': fallback
        }

        tform = PolynomialTransform()
        tform.estimate(cluster_src, cluster_dst, order=order)
        transforms[cluster_id] = tform

    # 预测待测点
    pred_mm = np.zeros_like(true_target_mm)

    if use_weighted_avg and actual_n_clusters > 1:
        # 加权平均：根据到聚类中心的距离
        for i, test_point in enumerate(test_px):
            weights = []
            predictions = []

            for cluster_id in range(actual_n_clusters):
                center = cluster_centers[cluster_id]
                dist = np.linalg.norm(test_point - center)
                weight = 1.0 / (dist + 1e-6)

                tform = transforms[cluster_id]
                pred = tform(test_point.reshape(1, -1))[0]

                weights.append(weight)
                predictions.append(pred)

            weights = np.array(weights)
            weights /= weights.sum()
            pred_mm[i] = np.average(predictions, axis=0, weights=weights)
    else:
        # 直接用最近聚类的变换
        for i, test_point in enumerate(test_px):
            # 找到最近的聚类中心
            dists = [np.linalg.norm(test_point - center) for center in cluster_centers]
            nearest_cluster = np.argmin(dists)

            tform = transforms[nearest_cluster]
            pred_mm[i] = tform(test_point.reshape(1, -1))[0]

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

    n_fallback = sum(1 for s in cluster_stats.values() if s['fallback'])

    return {
        "centroid_rmse_px": centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "success_rate_all": success_rate_all,
        "success_rate_calib": success_rate_calib,
        "success_rate_target": success_rate_target,
        "calib_used": len(src),
        "target_tested": len(test_px),
        "n_clusters_requested": n_clusters,
        "n_clusters_actual": actual_n_clusters,
        "order": order,
        "n_fallback": n_fallback,
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
    print(f"[Config] 聚类策略: K-Means, 每聚类 2 阶多项式")

    all_results = []

    # 测试不同的聚类数
    cluster_configs = [2, 3, 4]

    for n_clusters in cluster_configs:
        print(f"\n{'=' * 60}")
        print(f"测试聚类数: {n_clusters}")
        print(f"{'=' * 60}")

        for sample_name in sample_names:
            img_path = os.path.join(base_path, "dataset", "images", f"{sample_name}.npy")
            lbl_path = os.path.join(base_path, "dataset", "labels", f"{sample_name}.json")

            print(f"\n--- {sample_name} ---")
            try:
                res = run_clustering_baseline(img_path, lbl_path,
                                              n_clusters=n_clusters,
                                              order=2,
                                              max_det_error_px=1.0,
                                              use_weighted_avg=True)

                print(f"检测率: all={res['success_rate_all'] * 100:.1f}% "
                      f"calib={res['success_rate_calib'] * 100:.1f}% "
                      f"target={res['success_rate_target'] * 100:.1f}%")
                print(f"使用基准点: {res['calib_used']}")
                print(f"聚类: 请求={res['n_clusters_requested']}, 实际={res['n_clusters_actual']}, "
                      f"fallback={res['n_fallback']}")
                print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px")
                print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")

                all_results.append((sample_name, n_clusters, res))

            except Exception as e:
                print(f"运行出错: {e}")
                import traceback

                traceback.print_exc()

        # 汇总：找出每个样本的最佳聚类数
        print("\n" + "=" * 80)
        print("最佳配置汇总（按样本）")
        print("=" * 80)

        for sample_name in sample_names:
            sample_results = [(nc, r) for (sn, nc, r) in all_results if sn == sample_name]
            if not sample_results:
                continue

            best = min(sample_results, key=lambda x: x[1]['transform_rmse_um'])
            n_clusters, res = best

            print(f"\n{sample_name}:")
            print(f"  最佳聚类数: {n_clusters} (实际: {res['n_clusters_actual']})")
            print(f"  反演精度: {res['transform_rmse_um']:.2f} μm")
            print(f"  质心精度: {res['centroid_rmse_px']:.6f} px")
            print(f"  检测率: {res['success_rate_all'] * 100:.1f}%")

        # 完整对比表
        print("\n" + "=" * 80)
        print("完整对比表")
        print("=" * 80)
        print(f"{'Sample':<12} | {'Clusters':>8} | {'Actual':>6} | {'RMSE(μm)':>10} | {'Det(%)':>7}")
        print("-" * 80)

        for sample_name, n_clusters, res in all_results:
            print(f"{sample_name:<12} | {n_clusters:>8} | {res['n_clusters_actual']:>6} | "
                  f"{res['transform_rmse_um']:>10.2f} | {res['success_rate_all'] * 100:>7.1f}")