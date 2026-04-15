"""
detection_quality_analyzer.py
诊断检测失败原因，指导参数优化
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussian_detector import GaussianDetector, fit_gaussian, estimate_background, compute_snr
from config import SPOT_SIGMA_PX, MIN_SNR

# ★ 修复：直接构建数据路径，不依赖 DATA_DIR
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def analyze_failed_detections(sample_name):
    """
    分析检测失败的光斑，输出失败原因统计
    """
    img_path   = os.path.join(DATASET_DIR, "images", f"{sample_name}.npy")
    label_path = os.path.join(DATASET_DIR, "labels", f"{sample_name}.json")

    if not os.path.exists(img_path):
        print(f"[错误] 找不到图像: {img_path}")
        return
    if not os.path.exists(label_path):
        print(f"[错误] 找不到标签: {label_path}")
        return

    image = np.load(img_path)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fiber_data = label["fibers"]
    detector   = GaussianDetector()

    print(f"\n{'='*60}")
    print(f"=== 失败检测诊断: {sample_name} ===")
    print(f"{'='*60}")
    print(f"总光纤数: {len(fiber_data)}")
    print(f"检测窗口半径: {detector.half_win} px")

    fail_reasons = {
        'boundary':     0,
        'low_snr':      0,
        'bad_sigma':    0,
        'bad_position': 0,
        'fit_error':    0,
        'gate_fail':    0,
        'success':      0,
    }

    snr_list_success = []
    snr_list_fail    = []
    err_list_success = []
    err_list_fail    = []

    from spot_generator import extract_patch

    H_img, W_img = image.shape

    for fiber in fiber_data:
        seed_x = float(fiber.get('true_x_px', 0))
        seed_y = float(fiber.get('true_y_px', 0))
        true_x = seed_x
        true_y = seed_y

        half_win = detector.half_win

        # ── 边界检查 ──────────────────────────────────────────
        if (seed_x < half_win or seed_x >= W_img - half_win or
                seed_y < half_win or seed_y >= H_img - half_win):
            fail_reasons['boundary'] += 1
            continue

        patch, off_x, off_y = extract_patch(image, seed_x, seed_y, half_win)

        if patch.size == 0:
            fail_reasons['boundary'] += 1
            continue

        # ── 计算SNR（拟合前） ─────────────────────────────────
        bg  = estimate_background(patch)
        snr = compute_snr(patch, bg)

        # ── 拟合 ──────────────────────────────────────────────
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX)

        det_x = result.get('x0', np.nan) + off_x
        det_y = result.get('y0', np.nan) + off_y
        err   = np.hypot(det_x - true_x, det_y - true_y) if np.isfinite(det_x) else np.inf

        if not result.get('success', False):
            # 诊断失败原因
            x0   = result.get('x0', -1)
            y0   = result.get('y0', -1)
            H_p, W_p = patch.shape

            if snr < MIN_SNR * 0.3:
                fail_reasons['low_snr'] += 1
            elif not (0 <= x0 < W_p and 0 <= y0 < H_p):
                fail_reasons['bad_position'] += 1
            elif 'fit_error' in result:
                fail_reasons['fit_error'] += 1
            else:
                fail_reasons['bad_sigma'] += 1

            snr_list_fail.append(snr)
            err_list_fail.append(err)

        elif err >= 1.0:
            # 拟合成功但误差超过门控
            fail_reasons['gate_fail'] += 1
            snr_list_fail.append(snr)
            err_list_fail.append(err)

        else:
            fail_reasons['success'] += 1
            snr_list_success.append(snr)
            err_list_success.append(err)

    # ── 输出统计 ──────────────────────────────────────────────
    total = len(fiber_data)
    print(f"\n失败原因统计（门控阈值 1.0 px）：")
    print(f"  {'原因':<15} {'数量':>5} {'占比':>7}  {'图示'}")
    print(f"  {'-'*50}")
    for reason, count in fail_reasons.items():
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * max(1, int(pct / 2)) if count > 0 else ''
        print(f"  {reason:<15} {count:>5} ({pct:5.1f}%)  {bar}")

    print(f"\nSNR 统计：")
    if snr_list_success:
        print(f"  成功检测 SNR: 均值={np.mean(snr_list_success):6.1f}, "
              f"最小={np.min(snr_list_success):6.1f}, "
              f"中位={np.median(snr_list_success):6.1f}")
    if snr_list_fail:
        print(f"  失败检测 SNR: 均值={np.mean(snr_list_fail):6.1f}, "
              f"最小={np.min(snr_list_fail):6.1f}, "
              f"中位={np.median(snr_list_fail):6.1f}")

    print(f"\n定位误差统计（成功检测）：")
    if err_list_success:
        print(f"  误差分布: "
              f"均值={np.mean(err_list_success):.4f} px, "
              f"中位={np.median(err_list_success):.4f} px, "
              f"90%分位={np.percentile(err_list_success, 90):.4f} px")

    # ── 诊断建议 ──────────────────────────────────────────────
    print(f"\n诊断建议：")
    n_fail_total = total - fail_reasons['success']

    if fail_reasons['low_snr'] > n_fail_total * 0.3:
        print(f"  → [高优先] SNR失败({fail_reasons['low_snr']}个占主导): "
              f"降低 config.py 中的 MIN_SNR（当前={MIN_SNR}）")

    if fail_reasons['bad_sigma'] > n_fail_total * 0.3:
        print(f"  → [高优先] Sigma失败({fail_reasons['bad_sigma']}个占主导): "
              f"放宽 gaussian_detector.py 中 is_valid_fit_result 的 sigma 范围")

    if fail_reasons['gate_fail'] > n_fail_total * 0.3:
        print(f"  → [高优先] 门控失败({fail_reasons['gate_fail']}个占主导): "
              f"种子点偏差过大，检查数据集标注精度")

    if fail_reasons['boundary'] > total * 0.05:
        print(f"  → [中优先] 边界失败({fail_reasons['boundary']}个): "
              f"减小 FIT_WINDOW_SIGMA 或检查图像边缘光纤")

    if fail_reasons['fit_error'] > n_fail_total * 0.2:
        print(f"  → [中优先] 拟合异常({fail_reasons['fit_error']}个): "
              f"增大 maxfev 或改用 photutils")

    if fail_reasons['gate_fail'] == 0 and fail_reasons['low_snr'] == 0:
        print(f"  → 检测质量良好，失败主要来自边界/极低SNR光纤，无需优化")

    print()
    return fail_reasons


if __name__ == "__main__":
    samples = ['sample_00900', 'sample_00532', 'sample_00608']
    for sample in samples:
        analyze_failed_detections(sample)