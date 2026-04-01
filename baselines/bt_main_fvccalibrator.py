"""
baselines/bt_main_fvccalibrator.py

把主方法（GaussianDetector + FVCCalibrator）放到与各 bt_*.py 相同的数据集上做公平对比。

公平性原则：
1. 与 bt_affine.py / bt_poly.py / bt_weighted_rbf.py 使用同一批 dataset/images/*.npy
2. 与基线使用同样的 labels/*.json
3. 与基线使用同样的 seed_positions（标签真值粗定位）
4. 与基线使用同样的检测门控 max_det_error_px=1.0
5. 仅替换“坐标映射/标定模型”为主方法的 FVCCalibrator

输出指标尽量与 baseline 保持一致：
- centroid_rmse_px
- transform_rmse_um
- transform_max_um
- success_rate_all / calib / target
- calib_used / target_tested
"""

import os
import sys
import json
import numpy as np

# 将项目根目录加入环境变量
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from gaussian_detector import GaussianDetector
from evaluation.metrics import calculate_errors
from coordinate_transform import FVCCalibrator


def choose_poly_degree(n_calib_points: int) -> int:
    """
    参考 main_pipeline.py 的自适应策略：
    < 30   -> 3阶
    < 80   -> 4阶
    else   -> 5阶
    """
    if n_calib_points < 30:
        return 3
    elif n_calib_points < 80:
        return 4
    else:
        return 5


def run_main_method_on_dataset(image_path, label_path, max_det_error_px=1.0):
    """
    在真实/数据集样本上运行“主方法”的公平对比版本。

    流程：
    1. 读取 image 和 label
    2. 使用 GaussianDetector.detect_all() 做检测（与基线一致）
    3. 用与基线一致的门控规则筛选有效点
    4. 将有效点分成 calib / target
    5. 用主方法的 FVCCalibrator 做标定
    6. 用 calibrator.transform() 对 target 做反演
    7. 计算像素质心误差、物理坐标反演误差
    """
    # 1) 读取图像与标签
    image = np.load(image_path).astype(np.float32)
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    fiber_data = label["fibers"]

    # 2) 与 baseline 一样：使用标签真值像素位置作为 seed
    #    这样公平，因为 bt_affine / bt_poly / bt_weighted_rbf 都这么做
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]

    detector = GaussianDetector()
    results_list, _ = detector.detect_all(image, seed_positions)

    matched_calib_px = []
    matched_calib_mm = []
    matched_target_px = []
    matched_target_mm = []

    true_px_list = []
    detected_px_list = []

    valid_indices = []
    failed_calib = []
    failed_target = []

    calib_total = sum(1 for f in fiber_data if f["is_calib"])
    target_total = sum(1 for f in fiber_data if not f["is_calib"])
    calib_ok = 0
    target_ok = 0

    # 3) 统一门控，与 baseline 保持一致
    for i, res in enumerate(results_list):
        fib = fiber_data[i]

        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)
        true_x = fib["true_x_px"]
        true_y = fib["true_y_px"]
        is_calib = fib["is_calib"]

        ok = (
            res.get("success", False)
            and np.isfinite(det_x)
            and np.isfinite(det_y)
            and np.hypot(det_x - true_x, det_y - true_y) < max_det_error_px
        )

        if not ok:
            if is_calib:
                failed_calib.append(i)
            else:
                failed_target.append(i)
            continue

        valid_indices.append(i)

        # 记录像素级检测误差统计
        true_px_list.append([true_x, true_y])
        detected_px_list.append([det_x, det_y])

        # 分离基准光纤和待测光纤
        if is_calib:
            calib_ok += 1
            matched_calib_px.append([det_x, det_y])
            matched_calib_mm.append([fib["true_x_mm"], fib["true_y_mm"]])
        else:
            target_ok += 1
            matched_target_px.append([det_x, det_y])
            matched_target_mm.append([fib["true_x_mm"], fib["true_y_mm"]])

    success_rate_all = len(valid_indices) / len(fiber_data) if len(fiber_data) > 0 else 0.0
    success_rate_calib = calib_ok / calib_total if calib_total > 0 else 0.0
    success_rate_target = target_ok / target_total if target_total > 0 else 0.0

    # 4) 质心精度
    if len(true_px_list) == 0:
        raise RuntimeError("没有任何有效检测点，无法统计质心误差。")
    centroid_err = calculate_errors(np.array(true_px_list), np.array(detected_px_list))

    # 5) 准备标定数据
    src = np.array(matched_calib_px, dtype=float)   # pixel
    dst_mm = np.array(matched_calib_mm, dtype=float)  # mm
    dst_um = dst_mm * 1000.0

    if len(src) < 6:
        raise RuntimeError(f"可用基准点过少，无法运行主方法标定：len(src)={len(src)}")

    poly_degree = choose_poly_degree(len(src))

    calibrator = FVCCalibrator(poly_degree=poly_degree)

    # 主方法 calibrator 接收的是“物理坐标”，在 main_pipeline.py 里用的是 um
    calib_report = calibrator.calibrate(src, dst_um, verbose=False)

    # 6) 预测待测点物理坐标
    test_px = np.array(matched_target_px, dtype=float)
    true_target_mm = np.array(matched_target_mm, dtype=float)

    if len(test_px) == 0:
        raise RuntimeError("没有可用于评估的待测光纤点。")

    # main_pipeline.py 在 run_measurement() 中使用 calibrator.transform()
    pred_target_um = calibrator.transform(test_px)
    pred_target_mm = pred_target_um / 1000.0

    # 7) 坐标反演误差（单位转为 um）
    transform_err = calculate_errors(true_target_mm * 1000.0, pred_target_um)

    result = {
        "centroid_rmse_px": centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "transform_max_um": transform_err["max"],

        "success_rate_all": success_rate_all,
        "success_rate_calib": success_rate_calib,
        "success_rate_target": success_rate_target,

        "calib_used": len(src),
        "target_tested": len(test_px),

        "poly_degree": poly_degree,

        "failed_calib": failed_calib,
        "failed_target": failed_target,
        "failed_calib_count": len(failed_calib),
        "failed_target_count": len(failed_target),

        # 额外保留主方法内部标定信息，方便写论文
        "affine_rms_um": calib_report.get("affine_rms_um", None),
        "final_rms_um": calib_report.get("final_rms_um", None),
        "bias_x_um": calib_report.get("bias_x_um", None),
        "bias_y_um": calib_report.get("bias_y_um", None),
    }

    return result


def save_results_json(all_results, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    sample_names = [
        "sample_00900",
        "sample_00532",
        "sample_00608",
    ]

    dataset_image_dir = os.path.join(base_dir, "dataset", "images")
    dataset_label_dir = os.path.join(base_dir, "dataset", "labels")
    output_dir = os.path.join(base_dir, "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)

    print("[Config] 公平对比：主方法 on dataset")
    print("[Config] 检测器: GaussianDetector.detect_all()")
    print("[Config] 标定器: FVCCalibrator")
    print("[Config] 门控阈值: max_det_error_px = 1.0")

    all_results = []

    for sample_name in sample_names:
        img_path = os.path.join(dataset_image_dir, f"{sample_name}.npy")
        lbl_path = os.path.join(dataset_label_dir, f"{sample_name}.json")

        print(f"\n--- 正在测试 Main Method (FVCCalibrator): {sample_name} ---")

        if not os.path.exists(img_path):
            print(f"错误：找不到文件 {img_path}")
            continue
        if not os.path.exists(lbl_path):
            print(f"错误：找不到文件 {lbl_path}")
            continue

        try:
            res = run_main_method_on_dataset(
                img_path,
                lbl_path,
                max_det_error_px=1.0
            )

            print(f"总检测率: {res['success_rate_all']*100:.1f}%")
            print(f"基准光纤检测率: {res['success_rate_calib']*100:.1f}%")
            print(f"待测光纤检测率: {res['success_rate_target']*100:.1f}%")
            print(f"使用基准点: {res['calib_used']} (poly_degree: {res['poly_degree']})")
            print(f"测试目标点: {res['target_tested']}")
            print(f"未通过检测/门控的基准光纤: {res['failed_calib_count']} -> {res['failed_calib']}")
            print(f"未通过检测/门控的待测光纤: {res['failed_target_count']} -> {res['failed_target']}")
            print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px")
            print(f"[主方法标定] Affine RMS: {res['affine_rms_um']}")
            print(f"[主方法标定] Final  RMS: {res['final_rms_um']}")
            print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")
            print(f"[反演精度] Max : {res['transform_max_um']:.2f} μm")

            all_results.append((sample_name, res))

        except Exception as e:
            print(f"运行出错: {e}")

    print("\n================ 汇总结果 ================")
    packed_results = {}
    for sample_name, res in all_results:
        packed_results[sample_name] = res
        print(
            f"{sample_name:12s} | "
            f"all={res['success_rate_all']*100:5.1f}% | "
            f"calib={res['success_rate_calib']*100:5.1f}% | "
            f"target={res['success_rate_target']*100:5.1f}% | "
            f"used={res['calib_used']:2d} | "
            f"test={res['target_tested']:3d} | "
            f"poly={res['poly_degree']} | "
            f"centroid={res['centroid_rmse_px']:.4f} px | "
            f"transform={res['transform_rmse_um']:.2f} μm | "
            f"max={res['transform_max_um']:.2f} μm"
        )

    save_path = os.path.join(output_dir, "main_method_on_dataset_results.json")
    save_results_json(packed_results, save_path)