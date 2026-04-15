"""
diagnose_transform.py
专项诊断多项式反演精度问题
运行：python diagnose_transform.py
"""
import numpy as np
import json
import os
from skimage.transform import PolynomialTransform

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def diagnose_transform(sample_name):
    label_path = os.path.join(DATASET_DIR, "labels", f"{sample_name}.json")
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fibers = label["fibers"]
    print(f"\n{'='*60}")
    print(f"样本: {sample_name}")
    print(f"  场点位置: ({label['field_x_mm']:.1f}, {label['field_y_mm']:.1f}) mm")
    print(f"  SNR级别: {label['snr_level']}")
    print(f"  基准光纤数: {label['n_calib']}")

    # 提取坐标
    calib_fibers  = [f for f in fibers if f["is_calib"]]
    target_fibers = [f for f in fibers if not f["is_calib"]]

    src = np.array([[f["true_x_px"], f["true_y_px"]] for f in calib_fibers])
    dst = np.array([[f["true_x_mm"], f["true_y_mm"]] for f in calib_fibers])
    tgt_px = np.array([[f["true_x_px"], f["true_y_px"]] for f in target_fibers])
    tgt_mm = np.array([[f["true_x_mm"], f["true_y_mm"]] for f in target_fibers])

    print(f"\n  基准点像素范围: X=[{src[:,0].min():.0f}, {src[:,0].max():.0f}]")
    print(f"  基准点mm范围:   X=[{dst[:,0].min():.1f}, {dst[:,0].max():.1f}], "
          f"Y=[{dst[:,1].min():.1f}, {dst[:,1].max():.1f}]")
    print(f"  mm坐标中心:     ({dst[:,0].mean():.1f}, {dst[:,1].mean():.1f})")

    # ── 方法A：原始方法（不归一化dst）────────────────────────
    src_center = src.mean(axis=0)
    src_scale  = src.std(axis=0).mean()
    src_norm   = (src - src_center) / src_scale

    tform_A = PolynomialTransform()
    tform_A.estimate(src_norm, dst, order=2)

    tgt_norm  = (tgt_px - src_center) / src_scale
    pred_A    = tform_A(tgt_norm)
    err_A     = np.sqrt(((pred_A - tgt_mm)**2).sum(axis=1)) * 1000  # μm
    print(f"\n  方法A（原始，dst不归一化）:")
    print(f"    RMSE = {np.sqrt(np.mean(err_A**2)):.2f} μm, "
          f"中位 = {np.median(err_A):.2f} μm")

    # ── 方法B：dst也归一化（相对坐标）────────────────────────
    dst_center = dst.mean(axis=0)
    dst_scale  = dst.std(axis=0).mean()
    if dst_scale < 1e-8:
        dst_scale = 1.0
    dst_norm = (dst - dst_center) / dst_scale

    tform_B = PolynomialTransform()
    tform_B.estimate(src_norm, dst_norm, order=2)

    pred_B_norm = tform_B(tgt_norm)
    pred_B      = pred_B_norm * dst_scale + dst_center
    err_B       = np.sqrt(((pred_B - tgt_mm)**2).sum(axis=1)) * 1000
    print(f"\n  方法B（src和dst都归一化）:")
    print(f"    RMSE = {np.sqrt(np.mean(err_B**2)):.2f} μm, "
          f"中位 = {np.median(err_B):.2f} μm")

    # ── 方法C：理想变换（用真值验证理论精度上界）────────────
    # 直接用线性变换（仿射）近似，作为精度基准
    from numpy.linalg import lstsq

    # 仿射变换：mm = A * px_norm + b
    ones = np.ones((len(src_norm), 1))
    src_aug = np.hstack([src_norm, ones])   # (N, 3)

    # X方向
    Ax, _, _, _ = lstsq(src_aug, dst[:, 0], rcond=None)
    # Y方向
    Ay, _, _, _ = lstsq(src_aug, dst[:, 1], rcond=None)

    tgt_aug  = np.hstack([tgt_norm, np.ones((len(tgt_norm), 1))])
    pred_C   = np.column_stack([tgt_aug @ Ax, tgt_aug @ Ay])
    err_C    = np.sqrt(((pred_C - tgt_mm)**2).sum(axis=1)) * 1000
    print(f"\n  方法C（仿射变换，理论下界）:")
    print(f"    RMSE = {np.sqrt(np.mean(err_C**2)):.2f} μm, "
          f"中位 = {np.median(err_C):.2f} μm")

    # ── 分析：哪个方法最好？ ─────────────────────────────────
    best = min(
        ('A', np.sqrt(np.mean(err_A**2))),
        ('B', np.sqrt(np.mean(err_B**2))),
        ('C', np.sqrt(np.mean(err_C**2))),
        key=lambda x: x[1]
    )
    print(f"\n  ★ 最优方法: {best[0]}，RMSE={best[1]:.2f} μm")

    return {
        'sample': sample_name,
        'rmse_A_um': float(np.sqrt(np.mean(err_A**2))),
        'rmse_B_um': float(np.sqrt(np.mean(err_B**2))),
        'rmse_C_um': float(np.sqrt(np.mean(err_C**2))),
        'field_x_mm': label['field_x_mm'],
        'field_y_mm': label['field_y_mm'],
    }


if __name__ == "__main__":
    print("=== 多项式反演精度专项诊断 ===")
    results = []
    for s in ["sample_00900", "sample_00532", "sample_00608"]:
        r = diagnose_transform(s)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print("汇总：")
    print(f"{'样本':12s} | {'方法A':>10} | {'方法B':>10} | {'方法C':>10} | {'场点':>15}")
    for r in results:
        print(f"{r['sample']:12s} | "
              f"{r['rmse_A_um']:>8.2f}μm | "
              f"{r['rmse_B_um']:>8.2f}μm | "
              f"{r['rmse_C_um']:>8.2f}μm | "
              f"({r['field_x_mm']:5.0f},{r['field_y_mm']:5.0f})mm")