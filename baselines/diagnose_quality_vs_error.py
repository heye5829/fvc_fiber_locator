"""
diagnose_quality_vs_error.py

用途：
  分析 main_method_on_dataset_results.json 中的逐点质量指标与反演误差的关系。

输入：
  outputs/results/main_method_on_dataset_results.json

输出：
  1. 控制台打印：
     - 每个样本的基本指标
     - snr_local 分桶 vs err_norm_um
     - seed_shift_px 分桶 vs err_norm_um
     - distance_to_edge 分桶 vs err_norm_um
     - det_err_px 分桶 vs err_norm_um
     - Top-K 最大反演误差点

  2. JSON 文件：
     outputs/results/quality_vs_error_diagnosis.json

  3. CSV 文件：
     outputs/results/quality_vs_error_points.csv

说明：
  - 本脚本只做诊断，不改变主方法结果。
  - 重点判断：
      a) 低 snr_local 是否对应更大的 transform error
      b) seed_shift_px 是否能预测大误差
      c) 边缘点是否误差更大
      d) 大误差是否集中在少数点/局部区域
"""

import os
import json
import csv
import math
import numpy as np


# =========================
# 路径配置
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_JSON = os.path.join(
    BASE_DIR, "outputs", "results", "main_method_on_dataset_results.json"
)

OUT_JSON = os.path.join(
    BASE_DIR, "outputs", "results", "quality_vs_error_diagnosis.json"
)

OUT_CSV = os.path.join(
    BASE_DIR, "outputs", "results", "quality_vs_error_points.csv"
)


# =========================
# 工具函数
# =========================

def safe_float(x):
    """安全转 float。失败或非有限值返回 None。"""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def fmt(x, ndigits=2):
    """打印时格式化数值。"""
    if x is None:
        return "None"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "None"
        return f"{x:.{ndigits}f}"
    return str(x)


def calc_error_stats(values):
    """
    对误差数组计算统计量。
    values: list[float]
    """
    arr = np.array(
        [v for v in values if v is not None and np.isfinite(v)],
        dtype=float
    )

    if len(arr) == 0:
        return {
            "count": 0,
            "rmse": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "min": None,
            "std": None,
        }

    return {
        "count": int(len(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr)),
    }


def make_bin_label(low, high):
    if low is None:
        return f"<{high}"
    if high is None:
        return f">={low}"
    return f"[{low}, {high})"


def assign_bin(value, bins):
    """
    value: float
    bins: list[(low, high)]
      low=None 表示 -inf
      high=None 表示 +inf
    """
    if value is None or not np.isfinite(value):
        return "missing"

    for low, high in bins:
        if low is None and value < high:
            return make_bin_label(low, high)
        if high is None and value >= low:
            return make_bin_label(low, high)
        if low is not None and high is not None and low <= value < high:
            return make_bin_label(low, high)

    return "out_of_range"


def grouped_error_stats(records, field_name, bins):
    """
    按某个质量字段分桶，统计 err_norm_um。
    """
    groups = {}

    for r in records:
        value = safe_float(r.get(field_name))
        label = assign_bin(value, bins)
        groups.setdefault(label, []).append(r)

    output = {}
    for low, high in bins:
        label = make_bin_label(low, high)
        rs = groups.get(label, [])
        errs = [safe_float(r.get("err_norm_um")) for r in rs]
        output[label] = calc_error_stats(errs)

        # 附带该桶内质量字段统计，方便观察
        values = np.array(
            [
                safe_float(r.get(field_name))
                for r in rs
                if safe_float(r.get(field_name)) is not None
            ],
            dtype=float
        )
        if len(values) > 0:
            output[label][f"{field_name}_mean"] = float(np.mean(values))
            output[label][f"{field_name}_median"] = float(np.percentile(values, 50))
            output[label][f"{field_name}_min"] = float(np.min(values))
            output[label][f"{field_name}_max"] = float(np.max(values))
        else:
            output[label][f"{field_name}_mean"] = None
            output[label][f"{field_name}_median"] = None
            output[label][f"{field_name}_min"] = None
            output[label][f"{field_name}_max"] = None

    # missing 单独记录
    if "missing" in groups:
        rs = groups["missing"]
        errs = [safe_float(r.get("err_norm_um")) for r in rs]
        output["missing"] = calc_error_stats(errs)

    return output


def pearson_corr(records, x_name, y_name="err_norm_um"):
    """
    计算 Pearson 相关系数。
    """
    xs = []
    ys = []

    for r in records:
        x = safe_float(r.get(x_name))
        y = safe_float(r.get(y_name))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return None

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return None

    return float(np.corrcoef(xs, ys)[0, 1])


def merge_target_records(sample_result):
    """
    合并：
      per_point_records[index]
      target_transform_records[index]

    只保留：
      - target 点
      - in_gate=True
      - 有 err_norm_um 的点
    """
    per_point_records = sample_result.get("per_point_records", [])
    target_transform_records = sample_result.get("target_transform_records", [])

    per_point_by_idx = {
        int(r["index"]): r
        for r in per_point_records
        if "index" in r
    }

    merged = []

    for tr in target_transform_records:
        idx = int(tr["index"])
        pr = per_point_by_idx.get(idx)

        if pr is None:
            continue

        # 理论上 target_transform_records 已经只包含通过门控目标点；
        # 这里再显式检查一次。
        if bool(pr.get("is_calib", False)):
            continue
        if not bool(pr.get("in_gate", False)):
            continue

        row = {}
        row.update(pr)
        row.update(tr)

        # 统一清洗几个常用字段
        for key in [
            "snr_detector",
            "snr_local",
            "peak",
            "background",
            "noise_std",
            "signal_above_bg",
            "window_sum",
            "distance_to_edge",
            "seed_shift_px",
            "det_err_px",
            "err_norm_um",
            "err_x_um",
            "err_y_um",
            "det_x_px",
            "det_y_px",
            "true_x_px",
            "true_y_px",
        ]:
            if key in row:
                row[key] = safe_float(row[key])

        merged.append(row)

    return merged


def print_bin_table(title, stats_dict):
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'bin':>14s} | "
        f"{'count':>6s} | "
        f"{'rmse':>9s} | "
        f"{'median':>9s} | "
        f"{'p90':>9s} | "
        f"{'p95':>9s} | "
        f"{'max':>9s}"
    )
    print("-" * 78)

    for label, s in stats_dict.items():
        print(
            f"{label:>14s} | "
            f"{s.get('count', 0):6d} | "
            f"{fmt(s.get('rmse')):>9s} | "
            f"{fmt(s.get('median')):>9s} | "
            f"{fmt(s.get('p90')):>9s} | "
            f"{fmt(s.get('p95')):>9s} | "
            f"{fmt(s.get('max')):>9s}"
        )


def print_top_errors(records, top_k=20):
    sorted_records = sorted(
        records,
        key=lambda r: safe_float(r.get("err_norm_um")) or -1,
        reverse=True
    )

    print(f"\nTop {top_k} 最大反演误差点")
    print("-" * 120)
    print(
        f"{'rank':>4s} | "
        f"{'idx':>5s} | "
        f"{'err_um':>9s} | "
        f"{'err_x':>9s} | "
        f"{'err_y':>9s} | "
        f"{'snr_local':>10s} | "
        f"{'snr_det':>8s} | "
        f"{'seed_shift':>10s} | "
        f"{'det_err':>8s} | "
        f"{'edge':>8s} | "
        f"{'x_px':>8s} | "
        f"{'y_px':>8s}"
    )
    print("-" * 120)

    for rank, r in enumerate(sorted_records[:top_k], start=1):
        print(
            f"{rank:4d} | "
            f"{int(r.get('index')):5d} | "
            f"{fmt(r.get('err_norm_um')):>9s} | "
            f"{fmt(r.get('err_x_um')):>9s} | "
            f"{fmt(r.get('err_y_um')):>9s} | "
            f"{fmt(r.get('snr_local')):>10s} | "
            f"{fmt(r.get('snr_detector')):>8s} | "
            f"{fmt(r.get('seed_shift_px'), 4):>10s} | "
            f"{fmt(r.get('det_err_px'), 4):>8s} | "
            f"{fmt(r.get('distance_to_edge')):>8s} | "
            f"{fmt(r.get('det_x_px')):>8s} | "
            f"{fmt(r.get('det_y_px')):>8s}"
        )


def analyze_one_sample(sample_name, sample_result, top_k=20):
    records = merge_target_records(sample_result)

    all_errs = [safe_float(r.get("err_norm_um")) for r in records]
    overall_stats = calc_error_stats(all_errs)

    # 分桶设置
    snr_bins = [
        (None, 5),
        (5, 10),
        (10, 20),
        (20, 50),
        (50, 100),
        (100, None),
    ]

    seed_shift_bins = [
        (None, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.50),
        (0.50, None),
    ]

    distance_to_edge_bins = [
        (None, 30),
        (30, 60),
        (60, 100),
        (100, 200),
        (200, None),
    ]

    det_err_bins = [
        (None, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.50),
        (0.50, 1.00),
        (1.00, None),
    ]

    noise_std_bins = [
        (None, 30),
        (30, 60),
        (60, 100),
        (100, 200),
        (200, None),
    ]

    snr_stats = grouped_error_stats(records, "snr_local", snr_bins)
    seed_shift_stats = grouped_error_stats(records, "seed_shift_px", seed_shift_bins)
    edge_stats = grouped_error_stats(records, "distance_to_edge", distance_to_edge_bins)
    det_err_stats = grouped_error_stats(records, "det_err_px", det_err_bins)
    noise_stats = grouped_error_stats(records, "noise_std", noise_std_bins)

    correlations = {
        "corr_err_vs_snr_local": pearson_corr(records, "snr_local"),
        "corr_err_vs_snr_detector": pearson_corr(records, "snr_detector"),
        "corr_err_vs_seed_shift_px": pearson_corr(records, "seed_shift_px"),
        "corr_err_vs_det_err_px": pearson_corr(records, "det_err_px"),
        "corr_err_vs_distance_to_edge": pearson_corr(records, "distance_to_edge"),
        "corr_err_vs_noise_std": pearson_corr(records, "noise_std"),
        "corr_err_vs_window_sum": pearson_corr(records, "window_sum"),
    }

    top_errors = sorted(
        records,
        key=lambda r: safe_float(r.get("err_norm_um")) or -1,
        reverse=True
    )[:top_k]

    diagnosis = {
        "sample_name": sample_name,
        "num_target_records": int(len(records)),
        "summary_from_main": {
            "centroid_rmse_px": sample_result.get("centroid_rmse_px"),
            "transform_rmse_um": sample_result.get("transform_rmse_um"),
            "transform_max_um": sample_result.get("transform_max_um"),
            "calib_used": sample_result.get("calib_used"),
            "target_tested": sample_result.get("target_tested"),
            "poly_degree": sample_result.get("poly_degree"),
            "final_rms_um": sample_result.get("final_rms_um"),
            "loocv_rms_um": sample_result.get("loocv_rms_um"),
            "overfit_ratio": sample_result.get("overfit_ratio"),
            "snr_local_summary_target_in_gate": sample_result.get(
                "snr_local_summary_target_in_gate"
            ),
            "snr_local_summary_calib_in_gate": sample_result.get(
                "snr_local_summary_calib_in_gate"
            ),
        },
        "overall_error_stats": overall_stats,
        "correlations": correlations,
        "by_snr_local": snr_stats,
        "by_seed_shift_px": seed_shift_stats,
        "by_distance_to_edge": edge_stats,
        "by_det_err_px": det_err_stats,
        "by_noise_std": noise_stats,
        "top_errors": top_errors,
    }

    # 控制台打印
    print("\n" + "=" * 100)
    print(f"样本: {sample_name}")
    print("=" * 100)

    print(
        f"主结果: "
        f"target_tested={sample_result.get('target_tested')} | "
        f"calib_used={sample_result.get('calib_used')} | "
        f"poly={sample_result.get('poly_degree')} | "
        f"transform_rmse={fmt(sample_result.get('transform_rmse_um'))} um | "
        f"max={fmt(sample_result.get('transform_max_um'))} um | "
        f"final_rms={fmt(sample_result.get('final_rms_um'))} um | "
        f"loocv={fmt(sample_result.get('loocv_rms_um'))} um | "
        f"overfit={fmt(sample_result.get('overfit_ratio'))}x"
    )

    print(
        f"合并后的目标点记录数: {len(records)} | "
        f"重新统计 err RMSE={fmt(overall_stats['rmse'])} um | "
        f"median={fmt(overall_stats['median'])} um | "
        f"p90={fmt(overall_stats['p90'])} um | "
        f"max={fmt(overall_stats['max'])} um"
    )

    print("\n相关系数 Pearson corr(err_norm_um, quality)")
    print("-" * 60)
    for k, v in correlations.items():
        print(f"{k:35s}: {fmt(v, 4)}")

    print_bin_table("err_norm_um vs snr_local", snr_stats)
    print_bin_table("err_norm_um vs seed_shift_px", seed_shift_stats)
    print_bin_table("err_norm_um vs distance_to_edge", edge_stats)
    print_bin_table("err_norm_um vs det_err_px", det_err_stats)
    print_bin_table("err_norm_um vs noise_std", noise_stats)

    print_top_errors(records, top_k=top_k)

    return diagnosis, records


def save_flat_csv(all_flat_records, save_path):
    """
    保存所有样本合并后的逐点 CSV，方便用 Excel / pandas 继续画图。
    """
    if len(all_flat_records) == 0:
        print(f"[WARN] 没有逐点记录可保存 CSV: {save_path}")
        return

    fieldnames = [
        "sample_name",
        "index",
        "err_norm_um",
        "err_x_um",
        "err_y_um",
        "snr_local",
        "snr_detector",
        "peak",
        "background",
        "noise_std",
        "signal_above_bg",
        "window_sum",
        "seed_shift_px",
        "det_err_px",
        "distance_to_edge",
        "det_x_px",
        "det_y_px",
        "true_x_px",
        "true_y_px",
        "pred_x_um",
        "pred_y_um",
        "true_x_um",
        "true_y_um",
        "engine",
    ]

    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_flat_records:
            row = {k: r.get(k) for k in fieldnames}
            writer.writerow(row)

    print(f"\n逐点 CSV 已保存: {save_path}")


def main():
    if not os.path.exists(RESULT_JSON):
        raise FileNotFoundError(
            f"找不到结果文件: {RESULT_JSON}\n"
            f"请先运行 baselines/bt_main_fvccalibrator.py"
        )

    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    all_diagnosis = {}
    all_flat_records = []

    for sample_name, sample_result in all_results.items():
        diagnosis, records = analyze_one_sample(
            sample_name,
            sample_result,
            top_k=20,
        )

        all_diagnosis[sample_name] = diagnosis

        for r in records:
            row = dict(r)
            row["sample_name"] = sample_name
            all_flat_records.append(row)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_diagnosis, f, indent=2, ensure_ascii=False)

    print(f"\n诊断 JSON 已保存: {OUT_JSON}")

    save_flat_csv(all_flat_records, OUT_CSV)

    print("\n完成。下一步重点看：")
    print("  1. err_norm_um vs snr_local：低 SNR 桶是否 RMSE 明显更高")
    print("  2. err_norm_um vs seed_shift_px：seed_shift 是否能预测大误差")
    print("  3. err_norm_um vs distance_to_edge：边缘点是否误差更大")
    print("  4. Top 最大误差点是否集中在低 SNR、边缘、或较大 seed_shift 点")


if __name__ == "__main__":
    main()