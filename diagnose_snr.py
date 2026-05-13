"""
diagnose_snr.py

目标：
1. 在与当前主流程一致的检测配置下，分析每个 fiber 的局部信号质量
2. 统计 det_err_px 与 SNR / 背景 / 噪声 / 边缘距离 的关系
3. 判断当前瓶颈更偏向：
   - 低 SNR / 背景噪声问题
   - 还是 spot 模型失配问题

说明：
- 与 diagnose_detector.py 保持同样的检测门控思想
- 与 bt_main_fvccalibrator.py 保持同样的 GaussianDetector 配置思路
- 当前默认使用：
    use_photutils=False
    use_elliptical=True
    n_iter=2
    half_win=9
"""

import os
import sys
import json
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector


# 与现有脚本保持一致
MAX_DET_ERROR_PX = 1.0
SCALE_UM_PER_PX = 139.12


def robust_std(arr):
    """基于 MAD 的稳健噪声估计。"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad


def safe_stats(arr):
    """安全统计，避免空数组报错。"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "rmse": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def extract_patch_with_padding(image, cx, cy, half_win):
    """
    以 (cx, cy) 为中心提取 patch。
    若靠近边界则自动裁切，不补零。
    返回:
        patch, x0, x1, y0, y1
    """
    h, w = image.shape
    x0 = max(0, cx - half_win)
    x1 = min(w, cx + half_win + 1)
    y0 = max(0, cy - half_win)
    y1 = min(h, cy + half_win + 1)
    patch = image[y0:y1, x0:x1]
    return patch, x0, x1, y0, y1


def estimate_patch_quality(image, seed_x, seed_y, half_win=9):
    """
    基于 seed 附近 patch 估计局部质量指标。
    不依赖 detector 内部细节，作为统一诊断口径。

    返回:
        peak
        background
        noise_std
        snr
        window_sum
        distance_to_edge
        patch_shape
    """
    h, w = image.shape
    cx = int(round(seed_x))
    cy = int(round(seed_y))

    patch, x0, x1, y0, y1 = extract_patch_with_padding(image, cx, cy, half_win)
    patch = np.asarray(patch, dtype=float)

    if patch.size == 0:
        return {
            "peak": np.nan,
            "background": np.nan,
            "noise_std": np.nan,
            "snr": np.nan,
            "window_sum": np.nan,
            "distance_to_edge": 0.0,
            "patch_shape": [0, 0],
        }

    flat = patch.ravel()

    # 背景：用低分位近似
    background = float(np.percentile(flat, 20))

    # 峰值
    peak = float(np.max(flat))

    # 噪声：仅使用低于中位数的一半像素做稳健估计
    med = np.median(flat)
    bg_pixels = flat[flat <= med]
    if bg_pixels.size < 10:
        bg_pixels = flat
    noise_std = float(robust_std(bg_pixels))

    # 防止除零
    if not np.isfinite(noise_std) or noise_std < 1e-6:
        noise_std = 1e-6

    snr = float((peak - background) / noise_std)
    window_sum = float(np.sum(np.clip(flat - background, 0, None)))

    # 到图像边缘的最近距离
    distance_to_edge = float(min(cx, cy, w - 1 - cx, h - 1 - cy))

    return {
        "peak": peak,
        "background": background,
        "noise_std": noise_std,
        "snr": snr,
        "window_sum": window_sum,
        "distance_to_edge": distance_to_edge,
        "patch_shape": [int(patch.shape[0]), int(patch.shape[1])],
    }


def snr_bucket_name(snr):
    """SNR 分桶。"""
    if not np.isfinite(snr):
        return "nan"
    if snr < 5:
        return "<5"
    elif snr < 10:
        return "5-10"
    elif snr < 20:
        return "10-20"
    else:
        return ">=20"


def analyze_sample(sample_name, image, fiber_data, detector, half_win=9):
    """
    分析单个样本，输出点级别诊断结果和样本统计。
    """
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    results_list, _ = detector.detect_all(image, seed_positions)

    per_fiber = []
    engine_counts = {}

    for idx, (fib, res) in enumerate(zip(fiber_data, results_list)):
        true_x = float(fib["true_x_px"])
        true_y = float(fib["true_y_px"])
        is_calib = bool(fib["is_calib"])

        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)
        success = bool(res.get("success", False))
        engine = res.get("engine", "unknown")
        engine_counts[engine] = engine_counts.get(engine, 0) + 1

        if success and np.isfinite(det_x) and np.isfinite(det_y):
            det_err_px = float(np.hypot(det_x - true_x, det_y - true_y))
        else:
            det_err_px = np.nan

        in_gate = bool(np.isfinite(det_err_px) and det_err_px < MAX_DET_ERROR_PX)

        q = estimate_patch_quality(image, true_x, true_y, half_win=half_win)

        item = {
            "sample_name": sample_name,
            "fiber_index": int(idx),
            "is_calib": is_calib,
            "true_x_px": true_x,
            "true_y_px": true_y,
            "true_x_mm": float(fib.get("true_x_mm", np.nan)),
            "true_y_mm": float(fib.get("true_y_mm", np.nan)),

            "success": success,
            "engine": engine,
            "det_x_px": float(det_x) if np.isfinite(det_x) else np.nan,
            "det_y_px": float(det_y) if np.isfinite(det_y) else np.nan,
            "det_err_px": det_err_px,
            "det_err_um": float(det_err_px * SCALE_UM_PER_PX) if np.isfinite(det_err_px) else np.nan,
            "in_gate": in_gate,

            "peak": q["peak"],
            "background": q["background"],
            "noise_std": q["noise_std"],
            "snr": q["snr"],
            "snr_bucket": snr_bucket_name(q["snr"]),
            "window_sum": q["window_sum"],
            "distance_to_edge": q["distance_to_edge"],
            "patch_shape": q["patch_shape"],
        }
        per_fiber.append(item)

    # 样本整体统计
    det_err_all = [x["det_err_px"] for x in per_fiber if np.isfinite(x["det_err_px"])]
    det_err_gate = [x["det_err_px"] for x in per_fiber if x["in_gate"]]

    sample_summary = {
        "sample_name": sample_name,
        "n_total": int(len(per_fiber)),
        "n_success": int(sum(1 for x in per_fiber if x["success"])),
        "n_in_gate": int(sum(1 for x in per_fiber if x["in_gate"])),
        "success_rate": float(sum(1 for x in per_fiber if x["success"]) / len(per_fiber)),
        "in_gate_rate": float(sum(1 for x in per_fiber if x["in_gate"]) / len(per_fiber)),
        "engine_counts": engine_counts,
        "det_err_all_stats": safe_stats(det_err_all),
        "det_err_gate_stats": safe_stats(det_err_gate),
    }

    # 按 SNR 分桶统计
    bucket_names = ["<5", "5-10", "10-20", ">=20", "nan"]
    bucket_summary = {}

    for b in bucket_names:
        rows = [x for x in per_fiber if x["snr_bucket"] == b]
        n = len(rows)
        if n == 0:
            bucket_summary[b] = {
                "count": 0,
                "success_rate": np.nan,
                "in_gate_rate": np.nan,
                "rmse_all_px": np.nan,
                "rmse_gate_px": np.nan,
                "p95_all_px": np.nan,
                "mean_snr": np.nan,
                "mean_peak": np.nan,
                "mean_background": np.nan,
                "mean_noise_std": np.nan,
            }
            continue

        err_all = [x["det_err_px"] for x in rows if np.isfinite(x["det_err_px"])]
        err_gate = [x["det_err_px"] for x in rows if x["in_gate"]]

        s_all = safe_stats(err_all)
        s_gate = safe_stats(err_gate)

        bucket_summary[b] = {
            "count": int(n),
            "success_rate": float(sum(1 for x in rows if x["success"]) / n),
            "in_gate_rate": float(sum(1 for x in rows if x["in_gate"]) / n),
            "rmse_all_px": s_all["rmse"],
            "rmse_gate_px": s_gate["rmse"],
            "p95_all_px": s_all["p95"],
            "mean_snr": float(np.mean([x["snr"] for x in rows if np.isfinite(x["snr"])])) if any(np.isfinite(x["snr"]) for x in rows) else np.nan,
            "mean_peak": float(np.mean([x["peak"] for x in rows if np.isfinite(x["peak"])])) if any(np.isfinite(x["peak"]) for x in rows) else np.nan,
            "mean_background": float(np.mean([x["background"] for x in rows if np.isfinite(x["background"])])) if any(np.isfinite(x["background"]) for x in rows) else np.nan,
            "mean_noise_std": float(np.mean([x["noise_std"] for x in rows if np.isfinite(x["noise_std"])])) if any(np.isfinite(x["noise_std"]) for x in rows) else np.nan,
        }

    # 高误差点
    high_err = [x for x in per_fiber if np.isfinite(x["det_err_px"]) and x["det_err_px"] > 0.2]
    high_err.sort(key=lambda x: -x["det_err_px"])
    top_bad = high_err[:20]

    return {
        "sample_summary": sample_summary,
        "bucket_summary": bucket_summary,
        "top_bad_points": top_bad,
        "per_fiber": per_fiber,
    }


def print_sample_report(result):
    """
    打印单个样本的摘要报告。
    """
    summary = result["sample_summary"]
    buckets = result["bucket_summary"]
    top_bad = result["top_bad_points"]

    print(f"\n{'='*72}")
    print(f"样本: {summary['sample_name']}")
    print(f"{'='*72}")
    print(f"总点数:         {summary['n_total']}")
    print(f"success=True:   {summary['n_success']}  ({summary['success_rate']*100:.1f}%)")
    print(f"通过门控(<1px): {summary['n_in_gate']}  ({summary['in_gate_rate']*100:.1f}%)")
    print(f"引擎分布:       {summary['engine_counts']}")

    s_all = summary["det_err_all_stats"]
    s_gate = summary["det_err_gate_stats"]

    print(f"\n── 全部 success 点误差 ──")
    print(f"RMSE: {s_all['rmse']:.4f} px = {s_all['rmse']*SCALE_UM_PER_PX:.2f} μm")
    print(f"P50 : {s_all['p50']:.4f} px")
    print(f"P95 : {s_all['p95']:.4f} px")
    print(f"Max : {s_all['max']:.4f} px")

    print(f"\n── 门控内点误差 ──")
    print(f"RMSE: {s_gate['rmse']:.4f} px = {s_gate['rmse']*SCALE_UM_PER_PX:.2f} μm")
    print(f"P50 : {s_gate['p50']:.4f} px")
    print(f"P95 : {s_gate['p95']:.4f} px")
    print(f"Max : {s_gate['max']:.4f} px")

    print(f"\n── SNR 分桶统计 ──")
    print(f"{'桶':<8} {'count':>6} {'succ%':>8} {'gate%':>8} "
          f"{'RMSE_all(px)':>14} {'RMSE_gate(px)':>15} {'P95(px)':>10}")
    print("-"*72)
    for b in ["<5", "5-10", "10-20", ">=20", "nan"]:
        r = buckets[b]
        succ = r["success_rate"] * 100 if np.isfinite(r["success_rate"]) else np.nan
        gate = r["in_gate_rate"] * 100 if np.isfinite(r["in_gate_rate"]) else np.nan
        print(f"{b:<8} {r['count']:>6d} {succ:>7.1f}% {gate:>7.1f}% "
              f"{r['rmse_all_px']:>14.4f} {r['rmse_gate_px']:>15.4f} {r['p95_all_px']:>10.4f}")

    print(f"\n── Top 高误差点（det_err_px > 0.2，最多20个）──")
    if len(top_bad) == 0:
        print("无")
    else:
        print(f"{'idx':>4} {'calib':>5} {'err(px)':>10} {'err(um)':>10} "
              f"{'snr':>10} {'peak':>10} {'bg':>10} {'noise':>10} {'edge_dist':>10}")
        for x in top_bad:
            print(f"{x['fiber_index']:>4d} "
                  f"{str(x['is_calib']):>5} "
                  f"{x['det_err_px']:>10.4f} "
                  f"{x['det_err_um']:>10.2f} "
                  f"{x['snr']:>10.2f} "
                  f"{x['peak']:>10.2f} "
                  f"{x['background']:>10.2f} "
                  f"{x['noise_std']:>10.2f} "
                  f"{x['distance_to_edge']:>10.1f}")


def summarize_across_samples(all_results):
    """
    做一个跨样本的简表，便于看 00532 / 00608 / 00900 的差异。
    """
    print(f"\n\n{'='*72}")
    print("跨样本汇总")
    print(f"{'='*72}")
    print(f"{'sample':<14} {'succ%':>8} {'gate%':>8} {'gate_RMSE(px)':>16} "
          f"{'gate_RMSE(um)':>16}")
    print("-"*72)

    for sample_name, result in all_results.items():
        s = result["sample_summary"]
        gate_rmse = s["det_err_gate_stats"]["rmse"]
        print(f"{sample_name:<14} "
              f"{s['success_rate']*100:>7.1f}% "
              f"{s['in_gate_rate']*100:>7.1f}% "
              f"{gate_rmse:>16.4f} "
              f"{gate_rmse*SCALE_UM_PER_PX:>16.2f}")


def save_results_json(all_results, save_path):
    """
    保存为 JSON。
    """
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    dataset_dir = os.path.join(base_dir, "dataset")
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")

    output_dir = os.path.join(base_dir, "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        "sample_00900",
        "sample_00532",
        "sample_00608",
    ]

    # 与当前主流程一致的检测配置
    detector = GaussianDetector(
        use_photutils=False,
        use_elliptical=True,
        n_iter=2,
        half_win=9,
    )

    print(f"[Config] 焦面尺度: {SCALE_UM_PER_PX} μm/px")
    print(f"[Config] 门控阈值: {MAX_DET_ERROR_PX} px")
    print(f"[Config] 检测器: use_photutils=False, use_elliptical=True, n_iter=2, half_win=9")

    all_results = {}

    for sample in samples:
        image_path = os.path.join(image_dir, f"{sample}.npy")
        label_path = os.path.join(label_dir, f"{sample}.json")

        if not os.path.exists(image_path):
            print(f"找不到图像文件: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"找不到标签文件: {label_path}")
            continue

        image = np.load(image_path).astype(np.float32)
        with open(label_path, "r", encoding="utf-8") as f:
            fiber_data = json.load(f)["fibers"]

        result = analyze_sample(
            sample_name=sample,
            image=image,
            fiber_data=fiber_data,
            detector=detector,
            half_win=9,
        )
        all_results[sample] = result
        print_sample_report(result)

    summarize_across_samples(all_results)

    save_path = os.path.join(output_dir, "snr_diagnosis_results.json")
    save_results_json(all_results, save_path)