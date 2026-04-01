"""
baselines/bt_affine.py
Baseline 1: 全局仿射变换 (Global Affine Transform)
使用已有的 GaussianDetector 进行亚像素质心提取，并使用线性仿射变换进行标定。
"""

import os
import sys
import json
import numpy as np
from skimage.transform import AffineTransform

# 将根目录加入环境变量
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from evaluation.metrics import calculate_errors

def run_affine_baseline(image_path, label_path, max_det_error_px=1.0):
    # 1. 读取图像和真值标签
    image = np.load(image_path).astype(np.float32)
    with open(label_path, 'r', encoding='utf-8') as f:
        label = json.load(f)

    # 2. 准备检测：提取真值位置作为“种子点”
    # 真实场景下，这对应于光纤定位器的目标指令位置
    fiber_data = label["fibers"]
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]

    # 3. 运行高斯质心检测器 (调用 detect_all)
    detector = GaussianDetector()
    results_list, _ = detector.detect_all(image, seed_positions)

    # 4. 匹配与分类
    matched_calib_px = []     # 匹配上的基准点 (检测出的像素坐标)
    matched_calib_mm = []     # 对应的基准点 (真值物理坐标)
    matched_target_px = []    # 匹配上的待测点 (检测出的像素坐标)
    matched_target_mm = []    # 对应的待测点 (真值物理坐标)

    true_px_list = []         # 计算提取精度用
    detected_px_list = []     # 计算提取精度用

    for i, res in enumerate(results_list):
        if not res.get('success', False):
            continue

        fib = fiber_data[i]
        det_x, det_y = res['x_global'], res['y_global']
        true_x, true_y = fib["true_x_px"], fib["true_y_px"]

        # 过滤偏差过大的点
        if np.hypot(det_x - true_x, det_y - true_y) > max_det_error_px:
            continue

        # 记录提取精度数据
        true_px_list.append([true_x, true_y])
        detected_px_list.append([det_x, det_y])

        # 分离基准点和待测点
        if fib["is_calib"]:
            matched_calib_px.append([det_x, det_y])
            matched_calib_mm.append([fib["true_x_mm"], fib["true_y_mm"]])
        else:
            matched_target_px.append([det_x, det_y])
            matched_target_mm.append([fib["true_x_mm"], fib["true_y_mm"]])

    # --- 开始评价 ---

    # [指标 A] 质心提取误差 (Centroid Error) 像素级
    centroid_err = calculate_errors(true_px_list, detected_px_list)

    # 5. 拟合仿射变换 (仅使用基准光纤)
    src = np.array(matched_calib_px)
    dst = np.array(matched_calib_mm)

    if len(src) < 3:
        raise ValueError(f"基准点太少({len(src)})，无法拟合仿射变换！")

    tform = AffineTransform()
    tform.estimate(src, dst)

    # 6. 预测待测光纤的物理坐标
    test_px = np.array(matched_target_px)
    pred_mm = tform(test_px)

    # [指标 B] 坐标反演误差 (Transformation Error)
    # 计算 (预测mm - 真值mm) 并转为微米 (x1000)
    transform_err = calculate_errors(np.array(matched_target_mm) * 1000, pred_mm * 1000)

    return {
        "centroid_rmse_px": centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "transform_max_um": transform_err["max"],
        "calib_used": len(src),
        "target_tested": len(test_px),
        "success_rate": len(detected_px_list) / len(fiber_data)
    }

# ====== 多文件批量测试 ======
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sample_names = [
        "sample_00900",
        "sample_00532",
        "sample_00608",
    ]

    print(f"[Config] 焦面尺度: 139.12 μm/px")
    print(f"[Config] 目标精度: 3.0 μm = 0.0216 px")

    all_results = []

    for sample_name in sample_names:
        img_path = os.path.join(base_path, "dataset", "images", f"{sample_name}.npy")
        lbl_path = os.path.join(base_path, "dataset", "labels", f"{sample_name}.json")

        if not os.path.exists(img_path):
            print(f"\n--- {sample_name} ---")
            print(f"错误：找不到文件 {img_path}")
            continue

        print(f"\n--- 正在测试 Baseline 1 (Affine): {sample_name} ---")
        try:
            res = run_affine_baseline(img_path, lbl_path, max_det_error_px=1.0)

            print(f"成功检测率: {res['success_rate']*100:.1f}%")
            print(f"使用基准点: {res['calib_used']}")
            print(f"测试目标点: {res['target_tested']}")
            print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px")
            print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")
            print(f"[反演精度] Max : {res['transform_max_um']:.2f} μm")

            all_results.append((sample_name, res))

        except Exception as e:
            print(f"运行出错: {e}")

    print("\n================ 汇总结果 ================")
    for sample_name, res in all_results:
        print(
            f"{sample_name:12s} | "
            f"det={res['success_rate']*100:5.1f}% | "
            f"calib={res['calib_used']:2d} | "
            f"target={res['target_tested']:2d} | "
            f"centroid={res['centroid_rmse_px']:.4f} px | "
            f"transform={res['transform_rmse_um']:.2f} μm | "
            f"max={res['transform_max_um']:.2f} μm"
        )