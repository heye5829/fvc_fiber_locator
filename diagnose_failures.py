"""
diagnose_failures.py

分析门控外失败点的特征：
  - 在图像上的位置分布
  - 失败原因（低SNR / 相邻串扰 / 边缘截断 / 检测偏移过大）
  - 与成功点的对比统计
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── 设置中文字体（必须在import plt之后）──────────────────────
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei',
                                    'SimHei',
                                    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from scipy.spatial import KDTree

SCALE_UM_PX   = 139.12
MAX_GATE_PX   = 1.0
WINDOW_HALF   = 12
MIN_AMPLITUDE = 10.0


def analyze_failures(image, fiber_data, detector, sample_name):
    H, W = image.shape
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    results_list, _ = detector.detect_all(image, seed_positions)

    success   = []
    gate_fail = []
    det_fail  = []

    for res, fib in zip(results_list, fiber_data):
        tx = fib["true_x_px"]
        ty = fib["true_y_px"]

        is_edge = (tx < WINDOW_HALF or tx > W - WINDOW_HALF or
                   ty < WINDOW_HALF or ty > H - WINDOW_HALF)

        if not res.get("success", False):
            det_fail.append({"fib": fib, "res": res, "is_edge": is_edge})
            continue

        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)

        if not (np.isfinite(det_x) and np.isfinite(det_y)):
            det_fail.append({"fib": fib, "res": res, "is_edge": is_edge})
            continue

        err_px = np.hypot(det_x - tx, det_y - ty)
        item   = {"fib": fib, "res": res,
                  "err_px": err_px, "is_edge": is_edge}

        if err_px < MAX_GATE_PX:
            success.append(item)
        else:
            gate_fail.append(item)

    print(f"\n{'='*55}")
    print(f"样本：{sample_name}")
    print(f"{'='*55}")
    print(f"  总光纤数:       {len(fiber_data)}")
    print(f"  成功（门控内）: {len(success)}  "
          f"({len(success)/len(fiber_data)*100:.1f}%)")
    print(f"  偏移过大:       {len(gate_fail)}  "
          f"({len(gate_fail)/len(fiber_data)*100:.1f}%)")
    print(f"  检测失败:       {len(det_fail)}  "
          f"({len(det_fail)/len(fiber_data)*100:.1f}%)")

    if gate_fail:
        errs_arr = np.array([x["err_px"] for x in gate_fail])
        amp_arr  = np.array([x["res"].get("amplitude", np.nan)
                             for x in gate_fail])

        print(f"\n  ── 偏移过大点分析 ──────────────────────")
        print(f"  误差范围:  {errs_arr.min():.2f} ~ {errs_arr.max():.2f} px")
        print(f"  误差P50:   {np.median(errs_arr):.2f} px")
        print(f"  幅值均值:  {np.nanmean(amp_arr):.1f}")
        print(f"  边缘点数:  {sum(1 for x in gate_fail if x['is_edge'])}")

        bins = [1, 2, 3, 5, 10, 100]
        print(f"  误差分布:")
        prev = MAX_GATE_PX
        for b in bins:
            cnt = int(np.sum((errs_arr >= prev) & (errs_arr < b)))
            if cnt > 0:
                print(f"    {prev:.1f}~{b:.0f}px: {cnt}个")
            prev = b

    if success and gate_fail:
        succ_amps = np.array([x["res"].get("amplitude", np.nan)
                              for x in success])
        fail_amps = np.array([x["res"].get("amplitude", np.nan)
                              for x in gate_fail])
        succ_amps = succ_amps[np.isfinite(succ_amps)]
        fail_amps = fail_amps[np.isfinite(fail_amps)]

        if len(succ_amps) > 0 and len(fail_amps) > 0:
            print(f"\n  ── 幅值对比（成功 vs 偏移过大）──────")
            print(f"  成功点幅值: mean={succ_amps.mean():.1f}  "
                  f"median={np.median(succ_amps):.1f}")
            print(f"  失败点幅值: mean={fail_amps.mean():.1f}  "
                  f"median={np.median(fail_amps):.1f}")

    _plot_failure_map(image, success, gate_fail, det_fail,
                      sample_name, base_dir)

    return {"success": success, "gate_fail": gate_fail, "det_fail": det_fail}


def _plot_failure_map(image, success, gate_fail, det_fail,
                      sample_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{sample_name} - Detection Result Distribution",
                 fontsize=13)

    ax = axes[0]
    ax.imshow(image, cmap='gray', origin='upper',
              vmin=np.percentile(image, 1),
              vmax=np.percentile(image, 99))
    ax.set_title("Global Distribution")

    if success:
        xs = [x["fib"]["true_x_px"] for x in success]
        ys = [x["fib"]["true_y_px"] for x in success]
        ax.scatter(xs, ys, s=8, c='lime',
                   label=f'Success({len(success)})',
                   alpha=0.7, linewidths=0)

    if gate_fail:
        xs = [x["fib"]["true_x_px"] for x in gate_fail]
        ys = [x["fib"]["true_y_px"] for x in gate_fail]
        ax.scatter(xs, ys, s=25, c='red',
                   label=f'Offset>{MAX_GATE_PX}px({len(gate_fail)})',
                   marker='x', linewidths=1.2)

    if det_fail:
        xs = [x["fib"]["true_x_px"] for x in det_fail]
        ys = [x["fib"]["true_y_px"] for x in det_fail]
        ax.scatter(xs, ys, s=25, c='orange',
                   label=f'Det.Failed({len(det_fail)})',
                   marker='^', linewidths=1.2)

    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    ax2 = axes[1]
    ax2.imshow(image, cmap='gray', origin='upper',
               vmin=np.percentile(image, 1),
               vmax=np.percentile(image, 99))
    ax2.set_title("In-gate Error Magnitude")

    if success:
        xs   = [x["fib"]["true_x_px"] for x in success]
        ys   = [x["fib"]["true_y_px"] for x in success]
        errs = [x["err_px"] for x in success]
        sc   = ax2.scatter(xs, ys, s=12, c=errs,
                           cmap='RdYlGn_r', vmin=0, vmax=0.5,
                           alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax2, label='Error (px)', shrink=0.8)

    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "outputs",
                            f"{sample_name}_failure_map.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  可视化已保存: {out_path}")


def estimate_fiber_spacing(dataset_dir, samples):
    """估计光纤间距，用于确定最优 window_half"""
    print(f"\n\n{'='*55}")
    print(f"光纤间距估计")
    print(f"{'='*55}")

    for sample in samples:
        with open(os.path.join(dataset_dir, "labels",
                               f"{sample}.json")) as f:
            fiber_data_all = json.load(f)["fibers"]

        positions = np.array([[f["true_x_px"], f["true_y_px"]]
                               for f in fiber_data_all])

        tree    = KDTree(positions)
        dists, _ = tree.query(positions, k=2)
        nearest = dists[:, 1]

        med = np.median(nearest)
        print(f"\n  {sample}:")
        print(f"    最近邻间距 min:    {nearest.min():.2f} px")
        print(f"    最近邻间距 median: {med:.2f} px")
        print(f"    最近邻间距 mean:   {nearest.mean():.2f} px")
        print(f"    最近邻间距 P10:    {np.percentile(nearest, 10):.2f} px")
        print(f"    最近邻间距 P90:    {np.percentile(nearest, 90):.2f} px")
        print(f"")
        print(f"    当前 window_half   = {WINDOW_HALF} px")
        print(f"    建议 window_half   = {int(med * 0.45)} px"
              f"  （间距中位数的45%）")
        print(f"    保守 window_half   = {int(med * 0.40)} px"
              f"  （间距中位数的40%）")


if __name__ == "__main__":
    dataset_dir = os.path.join(base_dir, "dataset")
    samples     = ["sample_00532", "sample_00608"]

    detector = GaussianDetector(
        use_photutils=False,
        use_elliptical=True,
        n_iter=1,
    )

    all_results = {}
    for sample in samples:
        img = np.load(
            os.path.join(dataset_dir, "images", f"{sample}.npy")
        ).astype(np.float32)

        with open(os.path.join(dataset_dir, "labels",
                               f"{sample}.json")) as f:
            fiber_data = json.load(f)["fibers"]

        r = analyze_failures(img, fiber_data, detector, sample)
        all_results[sample] = r

    # ── 汇总失败点 ───────────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print(f"失败点汇总")
    print(f"{'='*55}")
    for sample in samples:
        r      = all_results[sample]
        n_tot  = (len(r["success"]) + len(r["gate_fail"])
                  + len(r["det_fail"]))
        n_fail = len(r["gate_fail"]) + len(r["det_fail"])
        print(f"\n  {sample}: {n_fail}/{n_tot} 失败")

        fail_ids = ([x["fib"].get("fiber_id", "?")
                     for x in r["gate_fail"]] +
                    [x["fib"].get("fiber_id", "?")
                     for x in r["det_fail"]])
        print(f"  失败fiber_id（前20）: {sorted(fail_ids)[:20]}")
        print(f"  共 {len(fail_ids)} 个")

    # ── 光纤间距估计（独立于上面的循环）────────────────────────
    estimate_fiber_spacing(dataset_dir, samples)