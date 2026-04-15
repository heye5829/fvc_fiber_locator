"""
check_dataset.py
快速检查数据集各样本的图像质量
"""
import numpy as np
import json
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def check_sample(sample_name):
    img_path   = os.path.join(DATASET_DIR, "images", f"{sample_name}.npy")
    label_path = os.path.join(DATASET_DIR, "labels", f"{sample_name}.json")

    if not os.path.exists(img_path):
        print(f"  [错误] 图像不存在: {img_path}")
        return

    img = np.load(img_path).astype(np.float64)

    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    fibers = label["fibers"]

    # 采样检查前10个光纤的patch质量
    from spot_generator import extract_patch
    snr_list   = []
    sigma_list = []

    half_win = 9
    for fiber in fibers[:50]:
        x = fiber["true_x_px"]
        y = fiber["true_y_px"]
        patch, _, _ = extract_patch(img, x, y, half_win)
        if patch.size == 0:
            continue

        patch_f = patch.astype(np.float64)
        bg      = float(np.percentile(patch_f, 25))
        peak    = float(patch_f.max()) - bg

        flat   = patch_f.ravel()
        mad    = float(np.median(np.abs(flat - np.median(flat))))
        noise  = mad * 1.4826 + 1e-6
        snr    = peak / noise
        snr_list.append(snr)

        # 估计sigma（用二阶矩）
        data  = np.maximum(patch_f - bg, 0)
        total = data.sum()
        if total > 1e-6:
            H_p, W_p = patch.shape
            y_arr, x_arr = np.mgrid[0:H_p, 0:W_p]
            cx = (x_arr * data).sum() / total
            cy = (y_arr * data).sum() / total
            var = ((x_arr - cx)**2 * data + (y_arr - cy)**2 * data).sum() / total
            sigma = float(np.sqrt(var / 2.0))
            sigma_list.append(sigma)

    print(f"\n{sample_name}:")
    print(f"  图像shape: {img.shape}, dtype: {np.load(img_path).dtype}")
    print(f"  图像范围: [{img.min():.1f}, {img.max():.1f}]")
    print(f"  光纤总数: {len(fibers)}, "
          f"基准光纤: {sum(1 for f in fibers if f['is_calib'])}")

    if snr_list:
        print(f"  SNR统计(前50): 均值={np.mean(snr_list):.1f}, "
              f"最小={np.min(snr_list):.1f}, "
              f"中位={np.median(snr_list):.1f}")
    if sigma_list:
        print(f"  Sigma统计(前50): 均值={np.mean(sigma_list):.2f}, "
              f"最小={np.min(sigma_list):.2f}, "
              f"最大={np.max(sigma_list):.2f}")

    # 检查坐标系
    xs = [f["true_x_px"] for f in fibers]
    ys = [f["true_y_px"] for f in fibers]
    print(f"  坐标范围: X=[{min(xs):.0f}, {max(xs):.0f}], "
          f"Y=[{min(ys):.0f}, {max(ys):.0f}]")

    if "true_x_mm" in fibers[0]:
        xs_mm = [f["true_x_mm"] for f in fibers]
        ys_mm = [f["true_y_mm"] for f in fibers]
        print(f"  焦面坐标范围: X=[{min(xs_mm):.1f}, {max(xs_mm):.1f}]mm, "
              f"Y=[{min(ys_mm):.1f}, {max(ys_mm):.1f}]mm")

        # 验证尺度因子
        px_span = max(xs) - min(xs)
        mm_span = max(xs_mm) - min(xs_mm)
        if px_span > 0:
            scale = mm_span / px_span * 1000
            print(f"  实际尺度: {scale:.3f} μm/px "
                  f"(config: 139.12 μm/px) "
                  f"{'✓' if abs(scale-139.12)<5 else '✗ 不匹配！'}")


if __name__ == "__main__":
    print("=== 数据集质量检查 ===")
    for s in ["sample_00900", "sample_00532", "sample_00608"]:
        check_sample(s)