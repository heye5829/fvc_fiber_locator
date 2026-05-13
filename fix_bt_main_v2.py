# 创建新文件：D:\fvc_fiber_locator\fix_bt_main_v2.py
# 内容如下（完整版）：

"""
修复验证脚本v2：使用局部坐标 + 修复路径问题
运行：python fix_bt_main_v2.py
"""
import os, sys, json
import numpy as np

# 项目根目录
PROJECT_ROOT = r"D:\fvc_fiber_locator"
sys.path.insert(0, PROJECT_ROOT)

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

    print(f"\n  图像尺寸: {image.shape}")
    print(f"  图像中心: ({img_cx:.1f}, {img_cy:.1f}) px")
    print(f"  图像尺度: {scale_um_px:.2f} μm/px")
    print(f"  场点位置: ({label['field_x_mm']:.1f}, {label['field_y_mm']:.1f}) mm")
    print(f"  光纤总数: {len(fiber_data)}")

    # 检测
    seeds = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]
    detector = GaussianDetector()
    results_list, _ = detector.detect_all(image, seeds)

    # 分离基准/目标
    calib_px, calib_um = [], []
    target_px, target_um = [], []
    true_px_all, det_px_all = [], []

    n_success = 0
    n_calib_ok = 0
    n_target_ok = 0

    for res, fib in zip(results_list, fiber_data):
        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)
        true_x = fib["true_x_px"]
        true_y = fib["true_y_px"]

        if res.get("success", False):
            n_success += 1

        ok = (res.get("success", False)
              and np.isfinite(det_x) and np.isfinite(det_y)
              and np.hypot(det_x - true_x, det_y - true_y) < max_det_error_px)

        if not ok:
            continue

        true_px_all.append([true_x, true_y])
        det_px_all.append([det_x, det_y])

        # 局部坐标
        loc_x_um = (true_x - img_cx) * scale_um_px
        loc_y_um = (true_y - img_cy) * scale_um_px

        if fib["is_calib"]:
            calib_px.append([det_x, det_y])
            calib_um.append([loc_x_um, loc_y_um])
            n_calib_ok += 1
        else:
            target_px.append([det_x, det_y])
            target_um.append([loc_x_um, loc_y_um])
            n_target_ok += 1

    print(f"  检测成功: {n_success}/{len(fiber_data)}")
    print(f"  门控通过: {len(true_px_all)}/{len(fiber_data)}")
    print(f"  基准点: {n_calib_ok}, 目标点: {n_target_ok}")

    # 检测精度
    if true_px_all:
        err_px = np.array(true_px_all) - np.array(det_px_all)
        rmse_det_px = np.sqrt(np.mean(np.sum(err_px ** 2, axis=1)))
        rmse_det_um = rmse_det_px * scale_um_px
        print(f"\n  [检测精度] {rmse_det_px:.5f} px = {rmse_det_um:.3f} μm")

    # 标定
    if len(calib_px) < 6:
        print(f"  ❌ 基准点不足（需要≥6，实际{len(calib_px)}）")
        return None

    src = np.array(calib_px)
    dst = np.array(calib_um)

    print(f"\n  局部坐标范围:")
    print(f"    X: [{dst[:, 0].min():.0f}, {dst[:, 0].max():.0f}] μm")
    print(f"    Y: [{dst[:, 1].min():.0f}, {dst[:, 1].max():.0f}] μm")

    n_calib = len(src)
    poly_deg = 3 if n_calib < 30 else (4 if n_calib < 80 else 5)

    cal = FVCCalibrator(poly_degree=poly_deg)
    rpt = cal.calibrate(src, dst, verbose=False)
    print(f"  多项式阶数: {poly_deg}")
    print(f"  标定残差: {rpt['final_rms_um']:.4f} μm")

    # 反演
    if len(target_px) == 0:
        print(f"  ⚠️  无目标点可测试")
        return rpt['final_rms_um']

    test_px = np.array(target_px)
    true_um = np.array(target_um)
    pred_um = cal.transform(test_px)

    err_um = true_um - pred_um
    err_r = np.sqrt(np.sum(err_um ** 2, axis=1))
    rmse_um = float(np.sqrt(np.mean(err_r ** 2)))
    p95_um = float(np.percentile(err_r, 95))
    max_um = float(np.max(err_r))

    print(f"\n  [反演精度]")
    print(f"    RMSE: {rmse_um:.3f} μm")
    print(f"    P95:  {p95_um:.3f} μm")
    print(f"    Max:  {max_um:.3f} μm")
    print(f"    目标3μm: {'✓ 达标' if rmse_um <= 3.0 else '✗ 未达标'}")

    return rmse_um


if __name__ == "__main__":

    samples = ["sample_00900", "sample_00532", "sample_00608"]
    img_dir = os.path.join(PROJECT_ROOT, "dataset", "images")
    lbl_dir = os.path.join(PROJECT_ROOT, "dataset", "labels")

    print("=" * 60)
    print("修复版v2：使用局部焦面坐标")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"图像目录: {img_dir}")
    print(f"标签目录: {lbl_dir}")

    if not os.path.exists(img_dir):
        print(f"\n❌ 图像目录不存在！")
        sys.exit(1)
    if not os.path.exists(lbl_dir):
        print(f"\n❌ 标签目录不存在！")
        sys.exit(1)

    results = []
    for sname in samples:
        img_p = os.path.join(img_dir, f"{sname}.npy")
        lbl_p = os.path.join(lbl_dir, f"{sname}.json")

        if not os.path.exists(img_p):
            print(f"\n{sname}: 图像文件不存在")
            continue
        if not os.path.exists(lbl_p):
            print(f"\n{sname}: 标签文件不存在")
            continue

        print(f"\n{'─' * 60}")
        print(f"样本: {sname}")
        try:
            rmse = run_fixed(img_p, lbl_p)
            if rmse is not None:
                results.append(rmse)
        except Exception as e:
            print(f"  ❌ 出错: {e}")
            import traceback

            traceback.print_exc()

    if results:
        print(f"\n{'=' * 60}")
        print(f"测试样本数: {len(results)}")
        print(f"平均RMSE: {np.mean(results):.3f} μm")
        print(f"最差RMSE: {np.max(results):.3f} μm")
        print(f"最佳RMSE: {np.min(results):.3f} μm")
        print(f"{'=' * 60}")