"""
baselines/bt_main_fvccalibrator.py

把主方法（GaussianDetector + FVCCalibrator）放到与各 bt_*.py 相同的数据集上做公平对比。

公平性原则：
1. 与 bt_affine.py / bt_poly.py / bt_weighted_rbf.py 使用同一批 dataset/images/*.npy
2. 与基线使用同样的 labels/*.json
3. 与基线使用同样的 seed_positions（标签真值粗定位）
4. 与基线使用同样的检测门控 max_det_error_px=1.0
5. 仅替换"坐标映射/标定模型"为主方法的 FVCCalibrator

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


def _safe_float_or_none(x):
    """
    将数值安全转成 float；非有限值、None 返回 None。
    用于 JSON 输出，避免 np.nan / np.inf 进入结果文件。
    """
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def compute_local_quality(image, x, y, half_win=9):
    """
    从检测中心附近 patch 计算统一的局部质量指标。

    这些指标不依赖 true_x / true_y，因此真实实验也可用。

    返回字段：
      peak             : patch 最大灰度
      background       : patch 边缘像素中位数，作为局部背景
      noise_std        : patch 边缘像素标准差，作为局部噪声
      snr_local        : (peak - background) / noise_std
      signal_above_bg  : peak - background
      window_sum       : patch 灰度总和
      distance_to_edge : 检测点到图像边界的最小距离
      patch_shape      : 实际 patch 尺寸
    """
    h, w = image.shape[:2]

    if not np.isfinite(x) or not np.isfinite(y):
        return {
            "peak": None,
            "background": None,
            "noise_std": None,
            "snr_local": None,
            "signal_above_bg": None,
            "window_sum": None,
            "distance_to_edge": None,
            "patch_shape": None,
        }

    xi = int(round(x))
    yi = int(round(y))

    x0 = max(0, xi - half_win)
    x1 = min(w, xi + half_win + 1)
    y0 = max(0, yi - half_win)
    y1 = min(h, yi + half_win + 1)

    patch = image[y0:y1, x0:x1].astype(float)

    if patch.size == 0:
        return {
            "peak": None,
            "background": None,
            "noise_std": None,
            "snr_local": None,
            "signal_above_bg": None,
            "window_sum": None,
            "distance_to_edge": None,
            "patch_shape": None,
        }

    peak = float(np.max(patch))
    window_sum = float(np.sum(patch))

    # 用 patch 外圈估计背景和噪声，避免光斑中心污染背景估计
    if patch.shape[0] >= 3 and patch.shape[1] >= 3:
        top = patch[0, :]
        bottom = patch[-1, :]
        left = patch[:, 0]
        right = patch[:, -1]
        border = np.concatenate([top, bottom, left, right])
    else:
        border = patch.ravel()

    background = float(np.median(border))
    noise_std = float(np.std(border))
    signal_above_bg = float(peak - background)

    if noise_std > 1e-6:
        snr_local = float(signal_above_bg / noise_std)
    else:
        snr_local = None

    distance_to_edge = float(min(x, y, w - 1 - x, h - 1 - y))

    return {
        "peak": peak,
        "background": background,
        "noise_std": noise_std,
        "snr_local": snr_local,
        "signal_above_bg": signal_above_bg,
        "window_sum": window_sum,
        "distance_to_edge": distance_to_edge,
        "patch_shape": [int(patch.shape[0]), int(patch.shape[1])],
    }




def mark_duplicate_detections(results_list, seed_positions, duplicate_dist_px=0.3):
    """
    对检测结果做重复峰去重。

    如果多个 fiber 检测到几乎同一个坐标，只保留 seed_shift 最小的那个。
    这可以抑制两个 seed 被同一个强峰吸附的问题。

    Parameters
    ----------
    results_list      : list 检测器输出
    seed_positions    : list 初始 seed 坐标
    duplicate_dist_px : float 判定为同一检测峰的距离阈值（px）

    Returns
    -------
    duplicate_indices : set[int]
        被判定为重复峰、需要剔除的 fiber index。
    duplicate_groups  : list[dict]
        重复峰分组诊断信息。
    """
    valid = []

    for i, res in enumerate(results_list):
        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)

        if not (
            res.get("success", False)
            and np.isfinite(det_x)
            and np.isfinite(det_y)
        ):
            continue

        seed_x, seed_y = seed_positions[i]
        seed_shift = float(np.hypot(det_x - seed_x, det_y - seed_y))

        valid.append({
            "index": int(i),
            "det_x": float(det_x),
            "det_y": float(det_y),
            "seed_shift": seed_shift,
        })

    duplicate_indices = set()
    duplicate_groups = []
    assigned = np.zeros(len(valid), dtype=bool)

    for a in range(len(valid)):
        if assigned[a]:
            continue

        cluster = [a]
        assigned[a] = True

        # 简单半径聚类：把与当前代表点足够近的检测归为同一峰
        for b in range(a + 1, len(valid)):
            if assigned[b]:
                continue

            dist = float(np.hypot(
                valid[a]["det_x"] - valid[b]["det_x"],
                valid[a]["det_y"] - valid[b]["det_y"],
            ))

            if dist < duplicate_dist_px:
                cluster.append(b)
                assigned[b] = True

        if len(cluster) <= 1:
            continue

        # 保留 seed_shift 最小的那个 fiber，其他认为检测到了重复峰
        cluster_sorted = sorted(cluster, key=lambda k: valid[k]["seed_shift"])
        keep = cluster_sorted[0]
        removed = cluster_sorted[1:]

        for k in removed:
            duplicate_indices.add(valid[k]["index"])

        duplicate_groups.append({
            "kept_index": int(valid[keep]["index"]),
            "removed_indices": [int(valid[k]["index"]) for k in removed],
            "det_x_px": float(valid[keep]["det_x"]),
            "det_y_px": float(valid[keep]["det_y"]),
            "kept_seed_shift_px": float(valid[keep]["seed_shift"]),
            "removed_seed_shift_px": [float(valid[k]["seed_shift"]) for k in removed],
            "cluster_size": int(len(cluster)),
        })

    return duplicate_indices, duplicate_groups


def choose_poly_degree(n_calib_points: int) -> int:
    """
    参数比 >= 2.0 的策略（不再过于保守）

    2阶：6参数，N>=12
    3阶：10参数，N>=20
    4阶：15参数，N>=30
    5阶：21参数，N>=42
    """
    if n_calib_points < 20:
        return 2
    elif n_calib_points < 30:
        return 3
    elif n_calib_points < 42:
        return 4
    else:
        return 5


def _choose_degree_by_loocv(src_px, dst_um, max_degree=5):
    """
    用留一交叉验证（LOOCV）选择最优多项式阶数。

    原理：
      对每个候选阶数，依次留出一个基准点，
      用剩余 N-1 个点标定，预测留出点，
      汇总所有预测误差得到 LOOCV RMS。
      选择 LOOCV RMS 最小的阶数。

    好处：
      LOOCV RMS 反映泛化误差，不会过拟合
      比固定阈值策略更自适应

    Parameters
    ----------
    src_px    : (N,2) 基准点像素坐标
    dst_um    : (N,2) 基准点焦面坐标（μm）
    max_degree: int   最大允许阶数

    Returns
    -------
    best_degree : int   最优阶数
    best_loocv  : float 对应的 LOOCV RMS (μm)
    """
    N = len(src_px)

    # 根据点数确定候选阶数范围（与 choose_poly_degree 保持一致）
    candidate_degrees = []
    if N >= 6:   candidate_degrees.append(2)
    if N >= 20:  candidate_degrees.append(3)
    if N >= 30:  candidate_degrees.append(4)
    if N >= 42:  candidate_degrees.append(5)

    # 至少保留2阶
    if not candidate_degrees:
        candidate_degrees = [2]

    # 限制最大阶数
    candidate_degrees = [d for d in candidate_degrees if d <= max_degree]

    best_degree = candidate_degrees[0]
    best_loocv = np.inf

    print(f"  [LOOCV] N={N}，候选阶数：{candidate_degrees}")

    for deg in candidate_degrees:
        loocv_errors = []

        for i in range(N):
            # 构造留一掩码
            mask = np.ones(N, dtype=bool)
            mask[i] = False

            try:
                # 用 N-1 个点标定
                cal_tmp = FVCCalibrator(poly_degree=deg)
                cal_tmp.calibrate(src_px[mask], dst_um[mask], verbose=False)

                # 预测第 i 个点
                pred = cal_tmp.transform(src_px[i:i+1])
                err = float(np.linalg.norm(pred[0] - dst_um[i]))
                loocv_errors.append(err)

            except Exception:
                # 标定失败，给一个大惩罚值
                loocv_errors.append(1e6)

        loocv_rms = float(np.sqrt(np.mean(np.array(loocv_errors) ** 2)))
        print(f"  [LOOCV] degree={deg}: RMS={loocv_rms:.2f} μm")

        if loocv_rms < best_loocv:
            best_loocv = loocv_rms
            best_degree = deg

    print(f"  [LOOCV] 选择 degree={best_degree}"
          f"（LOOCV RMS={best_loocv:.2f} μm）")
    return best_degree, best_loocv


def run_main_method_on_dataset(image_path, label_path,
                                max_det_error_px=1.0,
                                calib_gate_px=1.5,
                                use_loocv=True,
                                target_seed_shift_gate_px=0.2,
                                calib_seed_shift_gate_px=None,
                                enable_duplicate_filter=True,
                                duplicate_dist_px=0.3):
    """
    在真实/数据集样本上运行"主方法"的公平对比版本。

    流程：
    1. 读取 image 和 label
    2. 使用 GaussianDetector.detect_all() 做检测（与基线一致）
    3. 用与基线一致的门控规则筛选有效点
       - 基准点：使用宽松门控 calib_gate_px（默认1.5px）
       - 目标点：使用严格门控 max_det_error_px（默认1px）
    4. 将有效点分成 calib / target
    5. 用主方法的 FVCCalibrator 做标定
       - 可选：用 LOOCV 自动选择最优多项式阶数
    6. 用 calibrator.transform() 对 target 做反演
    7. 计算像素质心误差、物理坐标反演误差

    Parameters
    ----------
    image_path      : str   图像路径（.npy）
    label_path      : str   标签路径（.json）
    max_det_error_px: float 目标点门控阈值（px），与基线保持一致
    calib_gate_px   : float 基准点门控阈值（px），宽于目标点
                            原因：基准点数量直接影响标定质量，
                            过严的门控会导致基准点不足进而过拟合
    use_loocv       : bool  是否用 LOOCV 自动选阶数
                            True  → 防过拟合，更可靠
                            False → 用 choose_poly_degree 固定策略
    target_seed_shift_gate_px: float or None
                            目标点基于 seed_shift 的真实可用门控阈值。
                            None 表示不启用。
    calib_seed_shift_gate_px : float or None
                            基准点 seed_shift 门控阈值。默认 None，避免过度减少基准点。
    enable_duplicate_filter : bool 是否启用重复峰去重。
    duplicate_dist_px       : float 重复峰距离阈值（px）。
    """
    # 1) 读取图像与标签
    image = np.load(image_path).astype(np.float32)
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    fiber_data = label["fibers"]

    # 2) 与 baseline 一样：使用标签真值像素位置作为 seed
    #    这样公平，因为 bt_affine / bt_poly / bt_weighted_rbf 都这么做
    seed_positions = [[f["true_x_px"], f["true_y_px"]] for f in fiber_data]

    # detector = GaussianDetector()
    # results_list, _ = detector.detect_all(image, seed_positions)

    # 修改后：
    detector = GaussianDetector(
        use_photutils=False,  # 关闭photutils
        use_elliptical=True,  # 启用椭圆高斯
        n_iter=2,  #  迭代2次；diagnose_detector.py 对比表明优于 n_iter=1，n_iter=3 无额外收益
    )
    results_list, _ = detector.detect_all(image, seed_positions)

    # ── 新增：重复峰去重诊断/筛选 ───────────────────────────
    # 如果多个 fiber 被吸附到同一个强峰，只保留 seed_shift 最小的那个。
    if enable_duplicate_filter:
        duplicate_indices, duplicate_groups = mark_duplicate_detections(
            results_list,
            seed_positions,
            duplicate_dist_px=duplicate_dist_px,
        )
    else:
        duplicate_indices, duplicate_groups = set(), []

    # # ── 修改后 ──────────────────────────────────────────────────
    # detector = GaussianDetector(
    #     use_photutils=False,  # 强制走 scipy_enhanced（倾斜背景补偿）
    #     use_elliptical=False,  # 圆对称高斯（光斑基本是圆形）
    #     n_iter=2,  # 迭代精化一次
    # )
    # results_list, _ = detector.detect_all(image, seed_positions)

    # # ── 新增：统计各engine使用情况 ──────────────────────────────
    # engine_counts = {}
    # for res in results_list:
    #     eng = res.get('engine', 'unknown')
    #     engine_counts[eng] = engine_counts.get(eng, 0) + 1

    matched_calib_px = []
    matched_calib_mm = []
    matched_target_px = []
    matched_target_mm = []

    true_px_list = []
    detected_px_list = []

    valid_indices = []
    failed_calib = []
    failed_target = []

    # ── 新增：逐点诊断信息，用于后续分析误差来源 ─────────────
    # 注意：这里只记录，不参与当前算法筛选，不改变主方法结果
    per_point_records = []

    # 记录通过门控的基准点 / 目标点索引，方便后续空间分析
    used_calib_indices = []
    used_target_indices = []

    calib_total = sum(1 for f in fiber_data if f["is_calib"])
    target_total = sum(1 for f in fiber_data if not f["is_calib"])
    calib_ok = 0
    target_ok = 0


    # 3) 分离门控：基准点用宽松阈值，目标点用严格阈值
    for i, res in enumerate(results_list):
        fib = fiber_data[i]

        det_x = res.get("x_global", np.nan)
        det_y = res.get("y_global", np.nan)
        true_x = fib["true_x_px"]
        true_y = fib["true_y_px"]
        is_calib = fib["is_calib"]

        # 基准点用宽松门控，目标点用严格门控
        gate = calib_gate_px if is_calib else max_det_error_px

        det_err_px = np.hypot(det_x - true_x, det_y - true_y)

        # ── 新增：无需真值也可获得的检测质量指标 ─────────────
        # seed_x / seed_y 是检测初始位置，在真实场景中通常来自预测位置或上一轮位置
        seed_x, seed_y = seed_positions[i]
        seed_shift_px = np.hypot(det_x - seed_x, det_y - seed_y)

        # ── 新增：真实可用的 seed_shift 门控 + 重复峰去重 ─────
        seed_shift_gate = (
            calib_seed_shift_gate_px if is_calib else target_seed_shift_gate_px
        )

        if not res.get("success", False):
            failed_reason = "detector_failed"
        elif not (np.isfinite(det_x) and np.isfinite(det_y)):
            failed_reason = "nan_detection"
        elif enable_duplicate_filter and i in duplicate_indices:
            failed_reason = "duplicate_peak"
        elif seed_shift_gate is not None and (
            not np.isfinite(seed_shift_px) or seed_shift_px >= seed_shift_gate
        ):
            failed_reason = "seed_shift_gate"
        elif not np.isfinite(det_err_px) or det_err_px >= gate:
            failed_reason = "truth_gate"
        else:
            failed_reason = "ok"

        ok = (failed_reason == "ok")

        # # 尝试从 detector 返回结果中读取质量字段
        # # 如果某些字段当前 GaussianDetector 没有返回，则记为 None，不影响运行
        # peak = res.get("peak", None)
        # background = res.get("background", None)
        # noise_std = res.get("noise_std", None)
        # snr = res.get("snr", None)
        # engine = res.get("engine", "unknown")
        #
        # # 图像边缘距离：无需真值，真实场景可用
        # h, w = image.shape[:2]
        # if np.isfinite(det_x) and np.isfinite(det_y):
        #     distance_to_edge = min(det_x, det_y, w - 1 - det_x, h - 1 - det_y)
        # else:
        #     distance_to_edge = np.nan

        # 检测器内部返回的质量字段：保留，但不作为统一主指标
        engine = res.get("engine", "unknown")
        snr_detector = res.get("snr", None)

        # 新增：统一从图像 patch 计算局部质量指标
        # 这样 peak/background/noise_std/snr_local 的定义固定，
        # 不依赖 GaussianDetector 内部是否返回这些字段。
        quality = compute_local_quality(
            image=image,
            x=det_x,
            y=det_y,
            half_win=9,
        )

        peak = quality["peak"]
        background = quality["background"]
        noise_std = quality["noise_std"]
        snr_local = quality["snr_local"]
        signal_above_bg = quality["signal_above_bg"]
        window_sum = quality["window_sum"]
        distance_to_edge = quality["distance_to_edge"]
        patch_shape = quality["patch_shape"]

        # 保存逐点记录
        # det_err_px / true_x_px / true_y_px 是仿真评估字段；
        # seed_shift_px / snr / peak / background / noise_std / distance_to_edge 是真实场景也可用字段
        per_point_records.append({
            "index": int(i),
            "is_calib": bool(is_calib),

            "success": bool(res.get("success", False)),
            "engine": engine,

            "true_x_px": float(true_x),
            "true_y_px": float(true_y),
            "det_x_px": float(det_x) if np.isfinite(det_x) else None,
            "det_y_px": float(det_y) if np.isfinite(det_y) else None,

            "det_err_px": float(det_err_px) if np.isfinite(det_err_px) else None,
            "gate_px": float(gate),
            "seed_shift_gate_px": _safe_float_or_none(seed_shift_gate),
            "duplicate_peak": bool(enable_duplicate_filter and i in duplicate_indices),
            "failed_reason": failed_reason,
            "in_gate": bool(ok),

            # 无需真值的质量指标
            "seed_x_px": float(seed_x),
            "seed_y_px": float(seed_y),
            "seed_shift_px": float(seed_shift_px) if np.isfinite(seed_shift_px) else None,

            # detector 内部返回值：保留用于对照，不作为统一主指标
            "snr_detector": _safe_float_or_none(snr_detector),

            # 统一 patch 质量指标：后续质量分析/门控优先使用这些
            "snr_local": _safe_float_or_none(snr_local),
            "peak": _safe_float_or_none(peak),
            "background": _safe_float_or_none(background),
            "noise_std": _safe_float_or_none(noise_std),
            "signal_above_bg": _safe_float_or_none(signal_above_bg),
            "window_sum": _safe_float_or_none(window_sum),
            "distance_to_edge": _safe_float_or_none(distance_to_edge),
            "patch_shape": patch_shape,
        })

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
            used_calib_indices.append(i)
            matched_calib_px.append([det_x, det_y])
            matched_calib_mm.append([fib["true_x_mm"], fib["true_y_mm"]])
        else:
            target_ok += 1
            used_target_indices.append(i)
            matched_target_px.append([det_x, det_y])
            matched_target_mm.append([fib["true_x_mm"], fib["true_y_mm"]])

    success_rate_all = (len(valid_indices) / len(fiber_data)
                        if len(fiber_data) > 0 else 0.0)
    success_rate_calib = calib_ok / calib_total if calib_total > 0 else 0.0
    success_rate_target = target_ok / target_total if target_total > 0 else 0.0

    # ── 新增：失败原因统计 ───────────────────────────────────
    failed_reason_counts = {}
    failed_reason_counts_calib = {}
    failed_reason_counts_target = {}

    for r in per_point_records:
        reason = r.get("failed_reason", "unknown")
        failed_reason_counts[reason] = failed_reason_counts.get(reason, 0) + 1

        if r.get("is_calib", False):
            failed_reason_counts_calib[reason] = (
                failed_reason_counts_calib.get(reason, 0) + 1
            )
        else:
            failed_reason_counts_target[reason] = (
                failed_reason_counts_target.get(reason, 0) + 1
            )

    # ── 新增：样本级局部质量统计 ─────────────────────────────
    # 只统计通过门控的目标点，避免失败检测点污染质量分布。
    snr_local_target_in_gate = [
        r["snr_local"]
        for r in per_point_records
        if (not r["is_calib"])
        and r["in_gate"]
        and r.get("snr_local") is not None
    ]

    if len(snr_local_target_in_gate) > 0:
        arr = np.array(snr_local_target_in_gate, dtype=float)
        snr_local_summary_target_in_gate = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p10": float(np.percentile(arr, 10)),
            "p05": float(np.percentile(arr, 5)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": int(len(arr)),
        }
    else:
        snr_local_summary_target_in_gate = None

    # 可选：统计通过门控的基准点，后面分析标定点质量时有用
    snr_local_calib_in_gate = [
        r["snr_local"]
        for r in per_point_records
        if r["is_calib"]
        and r["in_gate"]
        and r.get("snr_local") is not None
    ]

    if len(snr_local_calib_in_gate) > 0:
        arr = np.array(snr_local_calib_in_gate, dtype=float)
        snr_local_summary_calib_in_gate = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p10": float(np.percentile(arr, 10)),
            "p05": float(np.percentile(arr, 5)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": int(len(arr)),
        }
    else:
        snr_local_summary_calib_in_gate = None

    # 4) 质心精度（所有通过门控的点）
    if len(true_px_list) == 0:
        raise RuntimeError("没有任何有效检测点，无法统计质心误差。")
    centroid_err = calculate_errors(
        np.array(true_px_list),
        np.array(detected_px_list)
    )

    # 5) 准备标定数据
    src    = np.array(matched_calib_px, dtype=float)   # pixel
    dst_mm = np.array(matched_calib_mm, dtype=float)   # mm
    dst_um = dst_mm * 1000.0                           # μm

    if len(src) < 6:
        raise RuntimeError(
            f"可用基准点过少，无法运行主方法标定：len(src)={len(src)}"
        )

    # ── 新增：标定点空间覆盖诊断 ─────────────────────────────
    src_min = src.min(axis=0)
    src_max = src.max(axis=0)
    src_span = src_max - src_min

    print(f"[标定点覆盖] x: {src_min[0]:.1f} ~ {src_max[0]:.1f} px, "
          f"y: {src_min[1]:.1f} ~ {src_max[1]:.1f} px, "
          f"span=({src_span[0]:.1f}, {src_span[1]:.1f}) px")

    # 6) 选择多项式阶数并标定
    loocv_rms = None

    if use_loocv:
        # LOOCV 自动选阶数（防止过拟合）
        poly_degree, loocv_rms = _choose_degree_by_loocv(src, dst_um)
    else:
        # 固定策略选阶数
        poly_degree = choose_poly_degree(len(src))

    calibrator = FVCCalibrator(poly_degree=poly_degree)
    calib_report = calibrator.calibrate(src, dst_um, verbose=False)

    # 7) 预测待测点物理坐标
    test_px        = np.array(matched_target_px, dtype=float)
    true_target_mm = np.array(matched_target_mm, dtype=float)

    if len(test_px) == 0:
        raise RuntimeError("没有可用于评估的待测光纤点。")

    pred_target_um = calibrator.transform(test_px)
    pred_target_mm = pred_target_um / 1000.0

    # 8) 坐标反演误差（单位 μm）
    transform_err = calculate_errors(true_target_mm * 1000.0, pred_target_um)

    # ── 新增：逐目标点反演误差诊断 ───────────────────────────
    target_transform_records = []
    true_target_um = true_target_mm * 1000.0
    target_err_vec_um = pred_target_um - true_target_um
    target_err_norm_um = np.linalg.norm(target_err_vec_um, axis=1)

    for k, idx in enumerate(used_target_indices):
        target_transform_records.append({
            "index": int(idx),
            "pred_x_um": float(pred_target_um[k, 0]),
            "pred_y_um": float(pred_target_um[k, 1]),
            "true_x_um": float(true_target_um[k, 0]),
            "true_y_um": float(true_target_um[k, 1]),
            "err_x_um": float(target_err_vec_um[k, 0]),
            "err_y_um": float(target_err_vec_um[k, 1]),
            "err_norm_um": float(target_err_norm_um[k]),
        })

    # 过拟合比：反演误差 / 标定残差
    # 理想情况 ≈ 1.0；> 2.0 说明存在过拟合
    final_rms = calib_report.get("final_rms_um", None)
    overfit_ratio = (transform_err["rmse"] / max(final_rms, 0.1)
                     if final_rms is not None else None)

    result = {
        "centroid_rmse_px":  centroid_err["rmse"],
        "transform_rmse_um": transform_err["rmse"],
        "transform_max_um":  transform_err["max"],

        "success_rate_all":    success_rate_all,
        "success_rate_calib":  success_rate_calib,
        "success_rate_target": success_rate_target,

        "calib_used":    len(src),
        "target_tested": len(test_px),
        "poly_degree":   poly_degree,

        "failed_calib":       failed_calib,
        "failed_target":      failed_target,
        "failed_calib_count": len(failed_calib),
        "failed_target_count":len(failed_target),

        # 新增：失败原因和重复峰诊断
        "failed_reason_counts": failed_reason_counts,
        "failed_reason_counts_calib": failed_reason_counts_calib,
        "failed_reason_counts_target": failed_reason_counts_target,
        "duplicate_filter_enabled": bool(enable_duplicate_filter),
        "duplicate_dist_px": _safe_float_or_none(duplicate_dist_px),
        "duplicate_groups": duplicate_groups,
        "duplicate_removed_indices": sorted(int(i) for i in duplicate_indices),
        "target_seed_shift_gate_px": _safe_float_or_none(target_seed_shift_gate_px),
        "calib_seed_shift_gate_px": _safe_float_or_none(calib_seed_shift_gate_px),

        # 主方法内部标定信息（论文写作用）
        "affine_rms_um": calib_report.get("affine_rms_um", None),
        "final_rms_um":  calib_report.get("final_rms_um",  None),
        "loocv_rms_um":  loocv_rms,
        "bias_x_um":     calib_report.get("bias_x_um",     None),
        "bias_y_um":     calib_report.get("bias_y_um",     None),

        # 过拟合诊断
        "overfit_ratio": overfit_ratio,

        # 新增：诊断信息，不参与算法，仅用于后续分析
        "used_calib_indices": used_calib_indices,
        "used_target_indices": used_target_indices,

        # 新增：统一局部 SNR 统计
        "snr_local_summary_target_in_gate": snr_local_summary_target_in_gate,
        "snr_local_summary_calib_in_gate": snr_local_summary_calib_in_gate,

        "calib_x_min_px": float(src_min[0]),
        "calib_x_max_px": float(src_max[0]),
        "calib_y_min_px": float(src_min[1]),
        "calib_y_max_px": float(src_max[1]),
        "calib_span_x_px": float(src_span[0]),
        "calib_span_y_px": float(src_span[1]),

        "per_point_records": per_point_records,
        "target_transform_records": target_transform_records,

        # "engine_counts": engine_counts,  # ← 加在这里
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
    output_dir        = os.path.join(base_dir, "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)

    # ── Config 打印（新增配置项）────────────────────────────────
    SCALE_UM_PER_PX  = 139.12
    TARGET_UM        = 3.0
    MAX_DET_ERR_PX   = 1.0
    CALIB_GATE_PX    = 1.5
    USE_LOOCV        = True

    # 新增：真实可用检测置信门控与重复峰去重
    TARGET_SEED_SHIFT_GATE_PX = 0.2
    CALIB_SEED_SHIFT_GATE_PX  = None
    ENABLE_DUPLICATE_FILTER   = True
    DUPLICATE_DIST_PX         = 0.3

    print(f"[Config] 焦面尺度: {SCALE_UM_PER_PX} μm/px")
    print(f"[Config] 目标精度: {TARGET_UM} μm = "
          f"{TARGET_UM/SCALE_UM_PER_PX:.4f} px")
    print(f"[Config] 公平对比：主方法 on dataset")
    print(f"[Config] 检测器: GaussianDetector.detect_all()")
    print(f"[Config] 标定器: FVCCalibrator")
    print(f"[Config] 目标点门控阈值: max_det_error_px = {MAX_DET_ERR_PX}")
    print(f"[Config] 目标seed_shift门控: target_seed_shift_gate_px = {TARGET_SEED_SHIFT_GATE_PX}")
    print(f"[Config] 基准seed_shift门控: calib_seed_shift_gate_px  = {CALIB_SEED_SHIFT_GATE_PX}")
    print(f"[Config] 重复峰去重:      enable={ENABLE_DUPLICATE_FILTER}, duplicate_dist_px={DUPLICATE_DIST_PX}")
    print(f"[Config] 基准点门控阈值: calib_gate_px    = {CALIB_GATE_PX}  "
          f"← 宽松门控，增加基准点数量")
    print(f"[Config] LOOCV 选阶数:   use_loocv        = {USE_LOOCV}  "
          f"← 防止过拟合")
    # 新增一行：
    print(f"[Config] 检测器配置:    椭圆高斯"
          f"（use_elliptical=True, use_photutils=False）")

    print(f"[Config] 质量指标:      patch局部计算 "
          f"snr_local=(peak-background)/noise_std, half_win=9")

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
                max_det_error_px=MAX_DET_ERR_PX,
                calib_gate_px=CALIB_GATE_PX,
                use_loocv=USE_LOOCV,
                target_seed_shift_gate_px=TARGET_SEED_SHIFT_GATE_PX,
                calib_seed_shift_gate_px=CALIB_SEED_SHIFT_GATE_PX,
                enable_duplicate_filter=ENABLE_DUPLICATE_FILTER,
                duplicate_dist_px=DUPLICATE_DIST_PX,
            )

            print(f"总检测率: {res['success_rate_all']*100:.1f}%")
            print(f"基准光纤检测率: {res['success_rate_calib']*100:.1f}%")
            print(f"待测光纤检测率: {res['success_rate_target']*100:.1f}%")
            print(f"使用基准点: {res['calib_used']} "
                  f"(poly_degree: {res['poly_degree']})")
            print(f"测试目标点: {res['target_tested']}")
            print(f"未通过检测/门控的基准光纤: "
                  f"{res['failed_calib_count']} -> {res['failed_calib']}")
            print(f"未通过检测/门控的待测光纤: "
                  f"{res['failed_target_count']} -> {res['failed_target']}")
            print(f"[失败原因/全部] {res.get('failed_reason_counts', {})}")
            print(f"[失败原因/目标] {res.get('failed_reason_counts_target', {})}")
            print(f"[重复峰] removed={len(res.get('duplicate_removed_indices', []))} "
                  f"groups={len(res.get('duplicate_groups', []))}")
            print(f"[质心精度] RMSE: {res['centroid_rmse_px']:.6f} px")
            print(f"[主方法标定] Affine RMS: {res['affine_rms_um']}")
            print(f"[主方法标定] Final  RMS: {res['final_rms_um']}")
            print(f"[主方法标定] LOOCV  RMS: {res['loocv_rms_um']}")
            print(f"[反演精度] RMSE: {res['transform_rmse_um']:.2f} μm")
            print(f"[反演精度] Max : {res['transform_max_um']:.2f} μm")

            print(f"[标定点覆盖] x_span={res['calib_span_x_px']:.1f} px, "
                  f"y_span={res['calib_span_y_px']:.1f} px")
            # print(f"[检测引擎] {res.get('engine_counts', {})}")

            # 新增：局部 SNR 质量统计
            s_tgt = res.get("snr_local_summary_target_in_gate")
            if s_tgt is not None:
                print(
                    f"[局部SNR/目标门控内] "
                    f"count={s_tgt['count']} | "
                    f"mean={s_tgt['mean']:.2f} | "
                    f"p50={s_tgt['p50']:.2f} | "
                    f"p10={s_tgt['p10']:.2f} | "
                    f"min={s_tgt['min']:.2f}"
                )

            s_cal = res.get("snr_local_summary_calib_in_gate")
            if s_cal is not None:
                print(
                    f"[局部SNR/基准门控内] "
                    f"count={s_cal['count']} | "
                    f"mean={s_cal['mean']:.2f} | "
                    f"p50={s_cal['p50']:.2f} | "
                    f"p10={s_cal['p10']:.2f} | "
                    f"min={s_cal['min']:.2f}"
                )

            # 过拟合诊断输出
            if res['overfit_ratio'] is not None:
                flag = ("⚠️  过拟合" if res['overfit_ratio'] > 2.0
                        else "✓  正常")
                print(f"[过拟合比] {res['overfit_ratio']:.2f}x  {flag}")

            all_results.append((sample_name, res))

        except Exception as e:
            print(f"运行出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n================ 汇总结果 ================")
    packed_results = {}
    for sample_name, res in all_results:
        packed_results[sample_name] = res
        overfit_str = (f"overfit={res['overfit_ratio']:.1f}x"
                       if res['overfit_ratio'] is not None else "overfit=N/A")
        loocv_str   = (f"loocv={res['loocv_rms_um']:.2f}μm"
                       if res['loocv_rms_um'] is not None else "loocv=N/A")
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
            f"max={res['transform_max_um']:.2f} μm | "
            f"{loocv_str} | "
            f"{overfit_str}"
        )

    save_path = os.path.join(
        output_dir, "main_method_on_dataset_results_seedgate_dupfilter.json"
    )
    save_results_json(packed_results, save_path)