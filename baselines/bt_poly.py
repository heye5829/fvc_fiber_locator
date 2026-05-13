"""
baselines/bt_poly.py
增强版：支持
1. 三张样本批量测试
2. 拆分 all/calib/target 检测率
3. 统计失败的基准光纤/待测光纤
4. 输出更完整的汇总信息
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

# ★ 新增：打印坐标范围用于调试
def _debug_coords(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)
    fibers = label["fibers"]
    xs_mm  = [f["true_x_mm"] for f in fibers if "true_x_mm" in f]
    xs_px  = [f["true_x_px"] for f in fibers]
    if xs_mm and xs_px:
        span_mm = max(xs_mm) - min(xs_mm)
        span_px = max(xs_px) - min(xs_px)
        scale   = span_mm / span_px * 1000 if span_px > 0 else 0
        print(f"  [调试] 坐标跨度: {span_px:.0f}px = {span_mm:.1f}mm, "
              f"实际尺度={scale:.2f}μm/px")


def run_poly_baseline(image_path, label_path, order=4,
                      det_gate_px=1.0,
                      calib_gate_px=3.0):
    image = np.load(image_path).astype(np.float32)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fiber_data = label["fibers"]
    detector = GaussianDetector()

    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    results_list, _ = detector.detect_all(image, seed_positions)

    # ── 统计变量 ──────────────────────────────────────────────
    matched_calib_px  = []
    matched_calib_mm  = []
    matched_target_px = []
    matched_target_mm = []
    centroid_det_px   = []
    centroid_true_px  = []
    failed_calib      = []
    failed_target     = []

    calib_total  = sum(1 for f in fiber_data if f["is_calib"])
    target_total = sum(1 for f in fiber_data if not f["is_calib"])
    calib_ok  = 0
    target_ok = 0

    for i, res in enumerate(results_list):
        det_x    = res.get('x_global', np.nan)
        det_y    = res.get('y_global', np.nan)
        true_x   = fiber_data[i]["true_x_px"]
        true_y   = fiber_data[i]["true_y_px"]
        is_calib = fiber_data[i]["is_calib"]

        det_success = res.get('success', False)
        if not det_success or not np.isfinite(det_x) or not np.isfinite(det_y):
            (failed_calib if is_calib else failed_target).append(i)
            continue

        err_px = np.hypot(det_x - true_x, det_y - true_y)

        # 严格门控：质心精度统计
        if err_px < det_gate_px:
            centroid_det_px.append([det_x, det_y])
            centroid_true_px.append([true_x, true_y])

        # 门控判断
        gate = calib_gate_px if is_calib else det_gate_px
        if err_px >= gate:
            (failed_calib if is_calib else failed_target).append(i)
            continue

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

    # ── 检测率 ────────────────────────────────────────────────
    total_ok = calib_ok + target_ok
    success_rate_all    = total_ok / len(fiber_data) if fiber_data else 0.0
    success_rate_calib  = calib_ok / calib_total if calib_total > 0 else 0.0
    success_rate_target = target_ok / target_total if target_total > 0 else 0.0

    src = np.array(matched_calib_px, dtype=np.float64)
    dst = np.array(matched_calib_mm, dtype=np.float64)

    if len(src) >= 30:
        actual_order = min(order, 4)
    elif len(src) >= 15:
        actual_order = min(order, 3)
    else:
        actual_order = 2

    if len(src) < 6:
        raise RuntimeError(
            f"可用基准点过少，无法进行多项式拟合：len(src)={len(src)}\n"
            f"  基准光纤总数={calib_total}, 通过门控={calib_ok}\n"
            f"  建议：增大 calib_gate_px（当前={calib_gate_px}px）"
        )

    # ★ 关键修复：同时归一化 src（像素）和 dst（mm坐标）
    # 原问题：dst包含绝对焦面坐标（如X=-375mm），多项式无法预测常数偏移
    # 修复：对dst也做中心化，变成预测相对偏差而非绝对坐标
    src_center = src.mean(axis=0)
    src_scale  = src.std(axis=0).mean()
    dst_center = dst.mean(axis=0)   # ★ 新增：记录mm坐标均值

    if not np.isfinite(src_scale) or src_scale < 1e-8:
        raise RuntimeError(f"src_scale 非法：{src_scale}")

    src_norm = (src - src_center) / src_scale
    # ★ dst不归一化，但记录中心用于后续还原
    # 多项式直接拟合 归一化像素 → mm（含绝对偏移）
    # 这样允许多项式的常数项吸收场偏移

    tform = PolynomialTransform()
    if not tform.estimate(src_norm, dst, order=actual_order):
        raise RuntimeError("PolynomialTransform.estimate() 拟合失败")

    # ── 预测与评估 ────────────────────────────────────────────
    test_px        = np.array(matched_target_px, dtype=np.float64)
    true_target_mm = np.array(matched_target_mm, dtype=np.float64)

    if len(test_px) == 0:
        raise RuntimeError("没有可用于评估的待测光纤点")

    test_norm = (test_px - src_center) / src_scale
    pred_mm   = tform(test_norm)

    # ★ 调试输出：验证预测范围
    pred_range = np.max(pred_mm, axis=0) - np.min(pred_mm, axis=0)
    true_range = np.max(true_target_mm, axis=0) - np.min(true_target_mm, axis=0)
    print(f"  [调试] 预测mm范围: X={pred_mm[:,0].min():.1f}~{pred_mm[:,0].max():.1f}, "
          f"Y={pred_mm[:,1].min():.1f}~{pred_mm[:,1].max():.1f}")
    print(f"  [调试] 真值mm范围: X={true_target_mm[:,0].min():.1f}~{true_target_mm[:,0].max():.1f}, "
          f"Y={true_target_mm[:,1].min():.1f}~{true_target_mm[:,1].max():.1f}")

    # mm → μm 计算误差
    true_um = true_target_mm * 1000.0
    pred_um = pred_mm        * 1000.0

    transform_err = calculate_errors(true_um, pred_um)

    # ── 质心精度 ──────────────────────────────────────────────
    if len(centroid_det_px) > 0:
        centroid_err  = calculate_errors(
            np.array(centroid_true_px, dtype=float),
            np.array(centroid_det_px,  dtype=float)
        )
        centroid_rmse = centroid_err["rmse"]
    else:
        centroid_rmse = np.nan

    return {
        "centroid_rmse_px":   centroid_rmse,
        "centroid_count":     len(centroid_det_px),
        "transform_rmse_um":  transform_err["rmse"],
        "success_rate_all":   success_rate_all,
        "success_rate_calib": success_rate_calib,
        "success_rate_target":success_rate_target,
        "calib_used":         len(src),
        "target_tested":      len(test_px),
        "actual_order":       actual_order,
        "failed_calib":       failed_calib,
        "failed_target":      failed_target,
        "failed_calib_count": len(failed_calib),
        "failed_target_count":len(failed_target),
        "dst_center_mm":      dst_center.tolist(),
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
    print(f"[Config] 多项式阶数: 4")

    all_results = []

    for sample_name in sample_names:
        img_path = os.path.join(
            base_path, "dataset", "images", f"{sample_name}.npy"
        )
        lbl_path = os.path.join(
            base_path, "dataset", "labels", f"{sample_name}.json"
        )

        print(f"\n--- 正在测试 Baseline 2 (Poly): {sample_name} ---")
        try:
            res = run_poly_baseline(
                img_path, lbl_path,
                order=4,
                det_gate_px=1.0,    # 严格门控：质心精度统计
                calib_gate_px=3.0   # 宽松门控：确保基准点够用
            )

            print(f"总检测率: {res['success_rate_all']*100:.1f}%")
            print(f"基准光纤检测率: {res['success_rate_calib']*100:.1f}%")
            print(f"待测光纤检测率: {res['success_rate_target']*100:.1f}%")
            print(f"使用基准点: {res['calib_used']} (阶数: {res['actual_order']})")
            print(f"测试目标点: {res['target_tested']}")
            print(f"未通过检测/门控的基准光纤: "
                  f"{res['failed_calib_count']} -> {res['failed_calib']}")
            print(f"未通过检测/门控的待测光纤: "
                  f"{res['failed_target_count']} -> {res['failed_target']}")
            print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px "
                  f"(统计点数: {res['centroid_count']})")
            print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")

            all_results.append((sample_name, res))

        except Exception as e:
            print(f"运行出错: {e}")

    print("\n================ 汇总结果 ================")
    for sample_name, res in all_results:
        centroid_str = (f"{res['centroid_rmse_px']:.4f} px"
                        if np.isfinite(res['centroid_rmse_px'])
                        else "N/A")
        print(
            f"{sample_name:12s} | "
            f"all={res['success_rate_all']*100:5.1f}% | "
            f"calib={res['success_rate_calib']*100:5.1f}% | "
            f"target={res['success_rate_target']*100:5.1f}% | "
            f"used={res['calib_used']:2d} | "
            f"test={res['target_tested']:3d} | "
            f"order={res['actual_order']} | "
            f"centroid={centroid_str} | "
            f"transform={res['transform_rmse_um']:.2f} μm"
        )