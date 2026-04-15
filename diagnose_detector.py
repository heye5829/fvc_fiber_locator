"""
diagnose_detector.py

对比三种检测配置在门控内的质心精度：
  A：原始（photutils优先）
  B：scipy_enhanced（倾斜背景，无迭代）
  C：scipy_enhanced（倾斜背景，迭代2次）
  D：椭圆高斯

与 bt_main_fvccalibrator.py 保持相同门控条件（1.0px）
才能得到可比较的质心精度数字
"""
import os
import sys
import json
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector

# 与 bt_main_fvccalibrator.py 保持一致的门控阈值
MAX_DET_ERROR_PX = 1.0


def test_detector(image, fiber_data, detector, label=""):
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    results_list, _ = detector.detect_all(image, seed_positions)

    # 统计引擎分布
    engine_counts = {}
    for res in results_list:
        eng = res.get('engine', 'unknown')
        engine_counts[eng] = engine_counts.get(eng, 0) + 1

    # 分三类统计
    n_total    = len(fiber_data)
    n_success  = 0   # success=True 的数量
    n_in_gate  = 0   # 通过门控的数量

    errors_all  = []   # 所有 success=True 的误差（含大误差）
    errors_gate = []   # 只看通过门控的误差

    for res, fib in zip(results_list, fiber_data):
        if not res.get("success", False):
            continue

        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)

        if not (np.isfinite(det_x) and np.isfinite(det_y)):
            continue

        n_success += 1
        err_px = np.hypot(det_x - fib["true_x_px"],
                          det_y - fib["true_y_px"])
        errors_all.append(err_px)

        if err_px < MAX_DET_ERROR_PX:
            n_in_gate += 1
            errors_gate.append(err_px)

    # 计算统计量
    def stats(arr):
        if len(arr) == 0:
            return {"rmse": np.nan, "mean": np.nan,
                    "p50": np.nan, "p95": np.nan, "max": np.nan}
        a = np.array(arr)
        return {
            "rmse": float(np.sqrt(np.mean(a**2))),
            "mean": float(np.mean(a)),
            "p50":  float(np.percentile(a, 50)),
            "p95":  float(np.percentile(a, 95)),
            "max":  float(np.max(a)),
        }

    s_all  = stats(errors_all)
    s_gate = stats(errors_gate)

    scale = 139.12   # μm/px

    print(f"\n  [{label}]")
    print(f"    引擎分布:       {engine_counts}")
    print(f"    总点数:         {n_total}")
    print(f"    success=True:   {n_success}  "
          f"({n_success/n_total*100:.1f}%)")
    print(f"    通过门控(<1px): {n_in_gate}  "
          f"({n_in_gate/n_total*100:.1f}%)")
    print(f"")
    print(f"    ── 全部success点（含大误差）──")
    print(f"    RMSE: {s_all['rmse']:.4f} px = "
          f"{s_all['rmse']*scale:.2f} μm")
    print(f"    P50:  {s_all['p50']:.4f} px  "
          f"P95: {s_all['p95']:.4f} px  "
          f"Max: {s_all['max']:.4f} px")
    print(f"")
    print(f"    ── 门控内点（与主脚本可比）──")
    print(f"    RMSE: {s_gate['rmse']:.4f} px = "
          f"{s_gate['rmse']*scale:.2f} μm")
    print(f"    P50:  {s_gate['p50']:.4f} px  "
          f"P95: {s_gate['p95']:.4f} px  "
          f"Max: {s_gate['max']:.4f} px")

    return {
        "n_total":   n_total,
        "n_success": n_success,
        "n_in_gate": n_in_gate,
        "all":       s_all,
        "gate":      s_gate,
        "engines":   engine_counts,
    }


if __name__ == "__main__":
    dataset_dir = os.path.join(base_dir, "dataset")
    samples = ["sample_00532", "sample_00608"]


    configs = [
        # 当前基准（half_win=9，FIT_WINDOW_SIGMA=4）
        ("当前_win9",
         GaussianDetector(use_photutils=False,
                          use_elliptical=True,
                          n_iter=1,
                          half_win=9)),  # 当前值

        # 缩小到7
        ("测试_win7",
         GaussianDetector(use_photutils=False,
                          use_elliptical=True,
                          n_iter=1,
                          half_win=7)),

        # 缩小到6（极限）
        ("测试_win6",
         GaussianDetector(use_photutils=False,
                          use_elliptical=True,
                          n_iter=1,
                          half_win=6)),

        # 放大到11（对比验证）
        ("测试_win11",
         GaussianDetector(use_photutils=False,
                          use_elliptical=True,
                          n_iter=1,
                          half_win=11)),
    ]

    all_results = {}

    for sample in samples:
        img = np.load(
            os.path.join(dataset_dir, "images", f"{sample}.npy")
        ).astype(np.float32)

        with open(os.path.join(dataset_dir, "labels",
                               f"{sample}.json")) as f:
            fiber_data = json.load(f)["fibers"]

        print(f"\n{'='*60}")
        print(f"样本：{sample}  (总光纤数={len(fiber_data)})")
        print(f"{'='*60}")

        sample_results = {}
        for label, det in configs:
            r = test_detector(img, fiber_data, det, label=label)
            sample_results[label] = r

        all_results[sample] = sample_results

# ── 修复后的完整文件末尾部分 ──────────────────────────────────

    # ── 汇总对比表 ──────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"汇总对比（门控内RMSE，与主脚本可比）")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'00532门控RMSE':>14} "
          f"{'00532通过率':>11} {'00608门控RMSE':>14} {'00608通过率':>11}")
    print(f"{'-'*60}")

    for label, _ in configs:               # ← 循环只打印汇总表
        r532 = all_results["sample_00532"][label]
        r608 = all_results["sample_00608"][label]
        rmse532 = r532["gate"]["rmse"]
        rmse608 = r608["gate"]["rmse"]
        rate532 = r532["n_in_gate"] / r532["n_total"] * 100
        rate608 = r608["n_in_gate"] / r608["n_total"] * 100
        print(f"  {label:<23} "
              f"{rmse532:.4f}px={rmse532*139.12:.1f}μm  "
              f"{rate532:.1f}%      "
              f"{rmse608:.4f}px={rmse608*139.12:.1f}μm  "
              f"{rate608:.1f}%")
    # ← 循环在这里结束，以下全部在循环外部

    # ── 建议输出（循环外，使用当前configs的key）────────────────
    print(f"\n{'='*60}")
    print(f"分析建议")
    print(f"{'='*60}")

    scale = 139.12

    # 找出门控内RMSE最小的配置
    best_label_532, best_rmse_532 = None, np.inf
    best_label_608, best_rmse_608 = None, np.inf
    best_rate_label_532, best_rate_532 = None, 0.0
    best_rate_label_608, best_rate_608 = None, 0.0

    for label, _ in configs:
        r532    = all_results["sample_00532"][label]
        r608    = all_results["sample_00608"][label]
        rmse532 = r532["gate"]["rmse"]
        rmse608 = r608["gate"]["rmse"]
        rate532 = r532["n_in_gate"] / r532["n_total"] * 100
        rate608 = r608["n_in_gate"] / r608["n_total"] * 100

        if rmse532 < best_rmse_532:
            best_rmse_532  = rmse532
            best_label_532 = label
        if rmse608 < best_rmse_608:
            best_rmse_608  = rmse608
            best_label_608 = label
        if rate532 > best_rate_532:
            best_rate_532       = rate532
            best_rate_label_532 = label
        if rate608 > best_rate_608:
            best_rate_608       = rate608
            best_rate_label_608 = label

    # 以第一个配置作为基准（不再硬编码名称）
    ref_label  = configs[0][0]             # ← 动态取第一个，不硬编码
    ref_532    = all_results["sample_00532"][ref_label]["gate"]["rmse"]
    ref_608    = all_results["sample_00608"][ref_label]["gate"]["rmse"]

    print(f"\n  基准配置: {ref_label}")
    print(f"    00532 RMSE = {ref_532:.4f}px = {ref_532*scale:.2f}μm")
    print(f"    00608 RMSE = {ref_608:.4f}px = {ref_608*scale:.2f}μm")

    print(f"\n  最优精度配置（门控内RMSE最小）：")
    improve_532 = (ref_532 - best_rmse_532) / ref_532 * 100
    improve_608 = (ref_608 - best_rmse_608) / ref_608 * 100
    print(f"    00532 → {best_label_532}")
    print(f"      RMSE: {best_rmse_532:.4f}px = {best_rmse_532*scale:.2f}μm"
          f"  vs基准: {improve_532:+.1f}%")
    print(f"    00608 → {best_label_608}")
    print(f"      RMSE: {best_rmse_608:.4f}px = {best_rmse_608*scale:.2f}μm"
          f"  vs基准: {improve_608:+.1f}%")

    print(f"\n  最高通过率配置：")
    print(f"    00532 → {best_rate_label_532}  ({best_rate_532:.1f}%)")
    print(f"    00608 → {best_rate_label_608}  ({best_rate_608:.1f}%)")

    # 综合评分（精度×通过率）
    print(f"\n  综合评分（RMSE越小×通过率越高 = 越好）：")
    print(f"  {'配置':<20} {'综合分':>10}  {'说明'}")
    print(f"  {'-'*55}")
    scores = []
    for label, _ in configs:
        r532   = all_results["sample_00532"][label]
        rmse   = r532["gate"]["rmse"]
        rate   = r532["n_in_gate"] / r532["n_total"]
        score  = rate / rmse          # 通过率/RMSE，越大越好
        scores.append((label, score, rmse, rate*100))
    scores.sort(key=lambda x: -x[1])
    for label, score, rmse, rate in scores:
        marker = " ← 推荐" if label == scores[0][0] else ""
        print(f"  {label:<20} {score:>10.1f}  "
              f"RMSE={rmse:.4f}px  率={rate:.1f}%{marker}")

    print(f"\n  结论：")
    best_overall = scores[0][0]
    print(f"    综合最优配置 = {best_overall}")
    print(f"    当前精度瓶颈：检测误差 ~0.068px = "
          f"{0.068*scale:.1f}μm")
    print(f"    目标精度：3μm = 0.022px")
    print(f"    下一步：提升质心精度（方向2），"
          f"这是从10μm→3μm的必经之路")