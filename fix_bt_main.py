"""
修复验证脚本：确认局部坐标方案的效果
运行：python fix_bt_main.py
"""
import os, sys, json
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from coordinate_transform import FVCCalibrator


def run_fixed(image_path, label_path, max_det_error_px=1.0):
    image = np.load(image_path).astype(np.float32)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fiber_data = label["fibers"]
    img_cx = label.get("img_cx_px", image.shape[1] / 2)
    img_cy = label.get("img_cy_px", image.shape[0] / 2)
    scale_um_px = label.get("scale_um_per_px", 139.12)

    print(f"\n  图像中心: ({img_cx}, {img_cy}) px")
    print(f"  图像尺度: {scale_um_px} μm/px")
    print(f"  场点位置: ({label['field_x_mm']:.1f}, "
          f"{label['field_y_mm']:.1f}) mm")
    print(f"  覆盖范围: {label.get('coverage_mm', '?')} mm")

    # ── 检测 ────────────────────────────────────────────────
    seeds = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    detector = GaussianDetector()
    results_list, _ = detector.detect_all(image, seeds)

    # ── 分离基准/目标，用局部坐标 ─────────────────────────────
    calib_px, calib_um = [], []
    target_px, target_um = [], []
    true_px_all, det_px_all = [], []

    for i, (res, fib) in enumerate(zip(results_list, fiber_data)):
        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)
        true_x = fib["true_x_px"]
        true_y = fib["true_y_px"]

        ok = (res.get("success", False)
              and np.isfinite(det_x) and np.isfinite(det_y)
              and np.hypot(det_x - true_x, det_y - true_y) < max_det_error_px)

        if not ok:
            continue

        true_px_all.append([true_x, true_y])
        det_px_all.append([det_x, det_y])

        # ★ 局部坐标（相对图像中心）
        loc_x_um = (true_x - img_cx) * scale_um_px
        loc_y_um = (true_y - img_cy) * scale_um_px

        if fib["is_calib"]:
            calib_px.append([det_x, det_y])
            calib_um.append([loc_x_um, loc_y_um])
        else:
            target_px.append([det_x, det_y])
            target_um.append([loc_x_um, loc_y_um])

    # ── 检测精度 ─────────────────────────────────────────────
    if true_px_all:
        err_px = np.array(true_px_all) - np.array(det_px_all)
        rmse_det_px = np.sqrt(np.mean(err_px ** 2))
        rmse_det_um = rmse_det_px * scale_um_px
        print(f"\n  [检测精度] {rmse_det_px:.5f} px "
              f"= {rmse_det_um:.3f} μm")

    # ── 标定 ─────────────────────────────────────────────────
    src = np.array(calib_px)
    dst = np.array(calib_um)

    print(f"\n  基准点数: {len(src)}")
    print(f"  局部坐标范围: X=[{dst[:, 0].min():.0f}, "
          f"{dst[:, 0].max():.0f}] μm")

    n_calib = len(src)
    poly_deg = 3 if n_calib < 30 else (4 if n_calib < 80 else 5)

    cal = FVCCalibrator(poly_degree=poly_deg)
    rpt = cal.calibrate(src, dst, verbose=False)
    print(f"  标定残差: {rpt['final_rms_um']:.4f} μm")

    # ── 反演精度 ─────────────────────────────────────────────
    test_px = np.array(target_px)
    true_um = np.array(target_um)
    pred_um = cal.transform(test_px)

    err_um = true_um - pred_um
    err_r = np.sqrt(err_um[:, 0] ** 2 + err_um[:, 1] ** 2)
    rmse_um = float(np.sqrt(np.mean(err_r ** 2)))
    p95_um = float(np.percentile(err_r, 95))
    max_um = float(np.max(err_r))

    print(f"\n  [反演精度] RMSE={rmse_um:.3f} μm  "
          f"P95={p95_um:.3f} μm  "
          f"Max={max_um:.3f} μm")
    print(f"  目标3μm: {'✓ 达标' if rmse_um <= 3.0 else '✗ 未达标'}")

    return rmse_um


# ── 主流程 ──────────────────────────────────────────────────
if __name__ == "__main__":

    samples = ["sample_00900", "sample_00532", "sample_00608"]
    img_dir = os.path.join(base_dir, "dataset", "images")
    lbl_dir = os.path.join(base_dir, "dataset", "labels")

    print("=" * 60)
    print("修复版：使用局部焦面坐标")
    print("=" * 60)

    results = []
    for sname in samples:
        img_p = os.path.join(img_dir, f"{sname}.npy")
        lbl_p = os.path.join(lbl_dir, f"{sname}.json")

        if not (os.path.exists(img_p) and os.path.exists(lbl_p)):
            print(f"\n{sname}: 文件不存在，跳过")
            continue

        print(f"\n{'─' * 40}")
        print(f"样本: {sname}")
        try:
            rmse = run_fixed(img_p, lbl_p)
            results.append(rmse)
        except Exception as e:
            print(f"  ❌ 出错: {e}")
            import traceback

            traceback.print_exc()

    if results:
        print(f"\n{'=' * 60}")
        print(f"平均RMSE: {np.mean(results):.3f} μm")
        print(f"最差RMSE: {np.max(results):.3f} μm")
        print(f"{'=' * 60}")