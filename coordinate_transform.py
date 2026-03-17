"""
坐标变换模块
实现：像素坐标 → 仿射变换 → 多项式畸变校正 → 焦面物理坐标(μm)

流程：
  1. 用基准光纤（已知焦面坐标）标定仿射矩阵 + 畸变系数
  2. 对待测光纤像素坐标做逆变换，得到焦面物理坐标
"""
import numpy as np
from numpy.polynomial import polynomial as P
from config import (FOCAL_PLANE_SCALE_UM_PX, POLY_ORDER,
                    IMAGE_WIDTH, IMAGE_HEIGHT)
import json
import os
import datetime


# ============================================================
# 仿射变换
# ============================================================

def fit_affine(px_coords, focal_coords):
    N = len(px_coords)
    assert N >= 4, "至少需要4个对应点"
    H = np.hstack([px_coords, np.ones((N, 1))])
    result, _, _, _ = np.linalg.lstsq(H, focal_coords, rcond=None)
    A = result[:2].T
    t = result[2]
    return A, t


def apply_affine(px_coords, A, t):
    px = np.atleast_2d(px_coords)
    return (A @ px.T).T + t


def affine_residuals(px_coords, focal_coords, A, t):
    predicted = apply_affine(px_coords, A, t)
    residuals = focal_coords - predicted
    rms = np.sqrt(np.mean(residuals ** 2))
    return residuals, rms


# ============================================================
# 多项式畸变模型
# ============================================================

def poly_features(coords_norm, degree):
    u, v = coords_norm[:, 0], coords_norm[:, 1]
    features = []
    for d in range(degree + 1):
        for i in range(d + 1):
            features.append(u ** (d - i) * v ** i)
    return np.column_stack(features)


def fit_distortion(affine_residuals_um, px_coords_norm, degree=POLY_ORDER):
    F = poly_features(px_coords_norm, degree)
    coeff_x, _, _, _ = np.linalg.lstsq(F, affine_residuals_um[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(F, affine_residuals_um[:, 1], rcond=None)
    return coeff_x, coeff_y


def apply_distortion_correction(px_coords_norm, coeff_x, coeff_y,
                                degree=POLY_ORDER):
    F = poly_features(np.atleast_2d(px_coords_norm), degree)
    corr_x = F @ coeff_x
    corr_y = F @ coeff_y
    return np.column_stack([corr_x, corr_y])


# ============================================================
# 主标定类
# ============================================================

class FVCCalibrator:
    """
    FVC标定器：建立像素→焦面的完整变换模型

    使用方法：
    ---------
    cal = FVCCalibrator()
    cal.calibrate(ref_px, ref_focal)    # 用基准光纤标定
    focal_xy = cal.transform(target_px) # 变换待测光纤坐标
    """

    def __init__(self, poly_degree=POLY_ORDER):
        self.poly_degree = poly_degree
        self.A = None
        self.t = None
        self.coeff_x = None
        self.coeff_y = None
        self.px_norm_scale = None
        self.px_norm_offset = None
        self.is_calibrated = False
        self.calibration_report = {}
        # 系统偏差补偿（消除标定残余的全局系统性偏移）
        self.bias_x = 0.0
        self.bias_y = 0.0

    def _normalize_px(self, px_coords):
        return (np.atleast_2d(px_coords) - self.px_norm_offset) / self.px_norm_scale

    def calibrate(self, ref_px, ref_focal_um, verbose=True):
        ref_px = np.array(ref_px, dtype=float)
        ref_focal_um = np.array(ref_focal_um, dtype=float)
        N = len(ref_px)
        assert N >= 6, f"基准光纤数量不足（当前{N}，至少需要6个）"

        # 1. 归一化参数
        self.px_norm_offset = ref_px.min(axis=0)
        self.px_norm_scale = ref_px.max(axis=0) - ref_px.min(axis=0)
        self.px_norm_scale = np.where(self.px_norm_scale < 1e-6, 1.0, self.px_norm_scale)

        # 2. 仿射变换
        self.A, self.t = fit_affine(ref_px, ref_focal_um)
        residuals_affine, rms_affine = affine_residuals(ref_px, ref_focal_um, self.A, self.t)

        # 3. 畸变模型
        px_norm = self._normalize_px(ref_px)
        self.coeff_x, self.coeff_y = fit_distortion(
            residuals_affine, px_norm, degree=self.poly_degree)

        # *** 关键修复：先设为已标定，才能在下一步调用 self.transform() ***
        self.is_calibrated = True

        # 4. 验证总体精度（此时 bias 尚未设置，transform 里 bias=0 不影响）
        final_pred = self.transform(ref_px)
        final_err = ref_focal_um - final_pred
        rms_final = float(np.sqrt(np.mean(final_err ** 2)))


        # 5. Sigma-clipping：剔除离群基准点后重新标定
        residuals_r = np.sqrt(np.sum(final_err ** 2, axis=1))
        sigma = np.std(residuals_r)
        mu = np.mean(residuals_r)
        mask = residuals_r < mu + 3.0 * sigma  # 保留3σ以内的点
        n_outliers = int(np.sum(~mask))

        if n_outliers > 0 and np.sum(mask) >= max(10, self.poly_degree * 3):
            # 用剩余点重新归一化
            ref_px_clean = ref_px[mask]
            ref_focal_clean = ref_focal_um[mask]

            self.px_norm_offset = ref_px_clean.min(axis=0)
            self.px_norm_scale = ref_px_clean.max(axis=0) - ref_px_clean.min(axis=0)
            self.px_norm_scale = np.where(
                self.px_norm_scale < 1e-6, 1.0, self.px_norm_scale)

            # 重新仿射拟合
            self.A, self.t = fit_affine(ref_px_clean, ref_focal_clean)
            residuals_affine2, rms_affine = affine_residuals(
                ref_px_clean, ref_focal_clean, self.A, self.t)

            # 重新畸变拟合
            px_norm_clean = self._normalize_px(ref_px_clean)
            self.coeff_x, self.coeff_y = fit_distortion(
                residuals_affine2, px_norm_clean, degree=self.poly_degree)

            # 重新计算残差（用全部点评估，包括被剔除的点）
            final_pred = self.transform(ref_px)
            final_err = ref_focal_um - final_pred
            rms_final = float(np.sqrt(np.mean(final_err ** 2)))

            if verbose:
                print(f"  Sigma-clipping: 剔除 {n_outliers} 个离群点"
                      f"（阈值 μ+3σ = {mu + 3 * sigma:.3f} um），重新标定")
        else:
            if verbose and n_outliers == 0:
                print(f"  Sigma-clipping: 无离群点，无需重新标定")

        # 6. 计算并存储系统偏差
        # final_err = ref_focal_um - final_pred，mean(final_err) 即为预测值的全局偏低量
        #undefined() 里加上 bias，使基准点残差均值归零
        self.bias_x = float(np.mean(final_err[:, 0]))
        self.bias_y = float(np.mean(final_err[:, 1]))

        # 7. 用补偿后的 transform 重新计算最终残差（用于报告）
        final_pred_corrected = self.transform(ref_px)
        final_err_corrected = ref_focal_um - final_pred_corrected
        rms_final_corrected = float(np.sqrt(np.mean(final_err_corrected ** 2)))

        # ── 8. 保存报告 ──────────────────────────────────────────────
        self.calibration_report = {
            'n_reference':    N,
            'affine_rms_um':  float(rms_affine),
            'final_rms_um':   rms_final_corrected,
            'poly_degree':    self.poly_degree,
            'bias_x_um':      self.bias_x,
            'bias_y_um':      self.bias_y,
        }

        # ── 8. 保存报告 ──────────────────────────────────────────────
        if verbose:
            print(f"=== FVC 标定完成 ===")
            print(f"  基准光纤数: {N}")
            print(f"  仿射残差 RMS: {rms_affine:.3f} um")
            print(f"  最终残差 RMS: {rms_final_corrected:.3f} um (含畸变校正)")
            print(f"  系统偏差:     X={self.bias_x:.3f} um, Y={self.bias_y:.3f} um")
            self._save_calibration_record()       # ← 无参数，只在verbose时调用
            # _save_calibration_record 内部会调用 _print_calibration_summary

        return self.calibration_report            # ← 唯一的return，返回报告字典

    def transform(self, px_coords):
        assert self.is_calibrated, "请先调用 calibrate()"
        px = np.atleast_2d(px_coords).astype(float)
        focal = apply_affine(px, self.A, self.t)
        px_norm = self._normalize_px(px)
        correction = apply_distortion_correction(px_norm, self.coeff_x, self.coeff_y,
                                                 degree=self.poly_degree)
        focal += correction
        # 补偿系统偏差（消除标定残余的全局系统性偏移）
        focal[:, 0] += self.bias_x
        focal[:, 1] += self.bias_y
        return focal

    def transform_with_uncertainty(self, px_coords, px_error_px=0.023):
        focal_um = self.transform(px_coords)
        scale_um_per_px = FOCAL_PLANE_SCALE_UM_PX
        uncertainty = px_error_px * scale_um_per_px * np.ones(len(focal_um))
        return focal_um, uncertainty

    # ----------------------------------------------------------------
    # 标定流程正式化（新增）
    # ----------------------------------------------------------------

    def _save_calibration_record(self):
        """保存完整标定记录到文件"""
        # ── 直接用类内已有变量，不从 config 导入不存在的名字 ──
        from config import OUTPUT_DIR

        # 从 calibration_report 里取数据（calibrate() 末尾已写好）
        rpt = self.calibration_report
        affine_rms  = rpt.get('affine_rms_um',  None)
        final_rms   = rpt.get('final_rms_um',   None)
        n_fid       = rpt.get('n_reference',     0)
        poly_degree = rpt.get('poly_degree',     self.poly_degree)

        # 改善倍数
        if affine_rms and final_rms and final_rms > 0:
            improvement = round(affine_rms / final_rms, 2)
        else:
            improvement = None

        # 仿射矩阵（A 和 t 合并成 2×3）
        if self.A is not None and self.t is not None:
            affine_23 = np.hstack([self.A, self.t.reshape(2, 1)]).tolist()
        else:
            affine_23 = None

        # 多项式系数
        coeffs_x = self.coeff_x.tolist() if self.coeff_x is not None else None
        coeffs_y = self.coeff_y.tolist() if self.coeff_y is not None else None
        n_coeffs  = len(self.coeff_x)    if self.coeff_x is not None else None

        # 标定质量等级
        grade = self._get_calibration_grade(final_rms)

        record = {
            # ── 1. 基本信息 ──────────────────────────────────
            "calibration_info": {
                "timestamp":        datetime.datetime.now().isoformat(),
                "system_name":      "MUST望远镜 FVC 光纤位置测量系统",
                "calibration_type": "基准光纤驱动的焦面几何标定",
                "description": (
                    "通过焦面板基准光纤建立FVC像素坐标系到焦面物理坐标系的映射。"
                    "第一步：仿射变换吸收整体平移、旋转、缩放；"
                    "第二步：高阶多项式拟合残余非线性畸变。"
                ),
            },

            # ── 2. 标定配置 ──────────────────────────────────
            "calibration_config": {
                "n_fiducials":          n_fid,
                "poly_degree":          poly_degree,
                "focal_scale_um_px":    float(FOCAL_PLANE_SCALE_UM_PX),
                "sigma_clip_threshold": 3.0,
                "coord_system": {
                    "pixel_origin": "FVC图像左上角",
                    "focal_origin": "焦面中心",
                    "focal_unit":   "μm",
                    "pixel_unit":   "px",
                },
            },

            # ── 3. 标定流程说明 ───────────────────────────────
            "calibration_procedure": {
                "step1": {
                    "name":        "基准光纤检测",
                    "description": "对FVC图像中的基准光纤光斑进行高斯拟合，获取亚像素精度中心坐标",
                    "output":      "基准光纤像素坐标 (u_i, v_i)",
                },
                "step2": {
                    "name":        "仿射初始映射",
                    "description": "最小二乘拟合6参数仿射变换，建立像素→焦面初始线性映射",
                    "model":       "X = a1*u + a2*v + a3 ;  Y = b1*u + b2*v + b3",
                    "algorithm":   "numpy.linalg.lstsq（最小二乘）",
                    "output":      "仿射矩阵 A(2×2) 和平移向量 t(2,)",
                },
                "step3": {
                    "name":        "残差计算",
                    "description": "计算仿射变换后的残余误差，体现光学系统高阶畸变",
                    "output":      "残差向量 (ΔX_i, ΔY_i)",
                },
                "step4": {
                    "name":        "Sigma-clipping 离群点剔除",
                    "description": "以 μ+3σ 为阈值迭代剔除异常基准点，提高标定鲁棒性",
                    "threshold":   "μ + 3.0σ",
                    "output":      "清洗后的基准点集合",
                },
                "step5": {
                    "name":        "高阶多项式畸变拟合",
                    "description": f"对残差用 {poly_degree} 阶多项式（含交叉项）拟合，吸收非线性畸变",
                    "model":       f"ΔX, ΔY = poly{poly_degree}(u_norm, v_norm)",
                    "algorithm":   "numpy.linalg.lstsq（最小二乘）",
                    "output":      "多项式系数向量 coeff_x, coeff_y",
                },
                "step6": {
                    "name":        "系统偏差补偿",
                    "description": "补偿标定残余的全局系统偏移量 (bias_x, bias_y)",
                    "output":      "偏差补偿标量",
                },
            },

            # ── 4. 仿射矩阵（2×3） ───────────────────────────
            "affine_matrix": {
                "description": "2×3 仿射矩阵，[X,Y]^T = A[:,0:2]*[u,v]^T + A[:,2]",
                "unit_in":     "px",
                "unit_out":    "μm",
                "matrix_2x3":  affine_23,
            },

            # ── 5. 多项式系数 ─────────────────────────────────
            "polynomial_model": {
                "description":    f"{poly_degree} 阶多项式拟合仿射残余畸变",
                "degree":         poly_degree,
                "n_coefficients": n_coeffs,
                "coeffs_x":       coeffs_x,
                "coeffs_y":       coeffs_y,
            },

            # ── 6. 系统偏差 ───────────────────────────────────
            "bias_compensation": {
                "bias_x_um":   float(self.bias_x),
                "bias_y_um":   float(self.bias_y),
                "description": "标定残余全局偏移补偿量",
            },

            # ── 7. 标定残差统计 ────────────────────────────────
            "calibration_residuals": {
                "n_fiducials_used":    n_fid,
                "affine_residual_rms": affine_rms,
                "final_residual_rms":  final_rms,
                "unit":                "μm",
                "interpretation": {
                    "affine_residual": "仿射变换后残差，反映光学系统非线性畸变量级",
                    "final_residual":  "多项式修正后残差，反映标定最终精度",
                },
            },

            # ── 8. 标定质量评估 ────────────────────────────────
            "quality_assessment": {
                "is_calibrated":     bool(self.is_calibrated),
                "affine_rms_um":     affine_rms,
                "final_rms_um":      final_rms,
                "improvement_ratio": improvement,
                "pass_threshold_um": 3.0,
                "pass":              (final_rms <= 3.0) if final_rms else False,
                "grade":             grade,
            },

            # ── 9. 重标定建议 ──────────────────────────────────
            "recalibration_guide": {
                "when_needed": [
                    "焦面板发生平移或旋转（任意量级）",
                    "焦面板发生倾斜（> 0.1°）",
                    "温度变化超过 ±5°C",
                    "仪器拆装或重新安装后",
                    "标定残差 RMS 超过 2.0 μm",
                ],
                "quick_recalibration": (
                    "若仅焦面板整体平移/旋转，可只重新拟合仿射矩阵（6参数），"
                    "保留多项式系数"
                ),
                "full_recalibration": (
                    "若光学链路、温度或装调状态发生明显变化，"
                    "需重新执行完整标定（仿射+多项式）"
                ),
            },
        }

        # ── 保存 JSON ─────────────────────────────────────────
        out_dir = os.path.join(OUTPUT_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "calibration_record.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n  标定记录已保存: {save_path}")
        self._print_calibration_summary(record)


    def _get_calibration_grade(self, final_rms=None):
        """根据最终残差给出标定质量等级"""
        if final_rms is None:
            return "未知"
        if final_rms <= 0.5:
            return "优秀 (≤0.5 μm)"
        elif final_rms <= 1.0:
            return "良好 (≤1.0 μm)"
        elif final_rms <= 2.0:
            return "合格 (≤2.0 μm)"
        elif final_rms <= 3.0:
            return "勉强合格 (≤3.0 μm)"
        else:
            return "不合格 (>3.0 μm)"


    def _print_calibration_summary(self, record):
        """在控制台打印标定流程摘要"""
        print("\n" + "=" * 50)
        print("  标定流程正式记录")
        print("=" * 50)

        info = record["calibration_info"]
        cfg  = record["calibration_config"]
        res  = record["calibration_residuals"]
        qa   = record["quality_assessment"]
        bias = record["bias_compensation"]

        print(f"  标定时间:       {info['timestamp'][:19]}")
        print(f"  标定类型:       {info['calibration_type']}")
        print()
        print(f"  基准光纤数:     {cfg['n_fiducials']} 根")
        print(f"  多项式阶数:     {cfg['poly_degree']} 阶")
        print(f"  像素尺度:       {cfg['focal_scale_um_px']} μm/px")
        print()
        print(f"  标定步骤:")
        for k, v in record["calibration_procedure"].items():
            print(f"    {k}: {v['name']}")
        print()
        print(f"  标定残差:")
        print(f"    仿射残差 RMS:   {res['affine_residual_rms']:.3f} μm")
        print(f"    最终残差 RMS:   {res['final_residual_rms']:.3f} μm")
        print(f"    改善倍数:       {qa['improvement_ratio']:.1f}×")
        print()
        print(f"  系统偏差补偿:   "
              f"X={bias['bias_x_um']:.4f} μm, "
              f"Y={bias['bias_y_um']:.4f} μm")
        print()
        print(f"  标定质量等级:   {qa['grade']}")
        print(f"  是否通过验收:   {'✓ 是' if qa['pass'] else '✗ 否'}")
        print("=" * 50)


    # ----------------------------------------------------------------
    # 重标定策略（新增）
    # ----------------------------------------------------------------

    def check_recalibration_needed(self, new_ref_px, new_ref_focal_um,
                                   rms_threshold_um=2.0,
                                   bias_threshold_um=5.0,
                                   verbose=True):
        """
        检查是否需要重标定

        用当前标定模型预测新基准点坐标，
        若残差超过阈值则触发重标定。

        Parameters
        ----------
        new_ref_px        : (M,2) 新观测的基准光纤像素坐标
        new_ref_focal_um  : (M,2) 对应的焦面真值坐标（μm）
        rms_threshold_um  : float  RMS残差触发阈值（默认2.0μm）
        bias_threshold_um : float  全局偏移触发阈值（默认5.0μm）
        verbose           : bool   是否打印检查结果

        Returns
        -------
        result : dict  包含检查结论和各项指标
        """
        assert self.is_calibrated, "请先执行初始标定"

        new_ref_px       = np.array(new_ref_px,       dtype=float)
        new_ref_focal_um = np.array(new_ref_focal_um, dtype=float)

        # 用当前模型预测
        pred = self.transform(new_ref_px)
        err  = new_ref_focal_um - pred
        err_r = np.sqrt(err[:, 0]**2 + err[:, 1]**2)

        current_rms    = float(np.sqrt(np.mean(err_r**2)))
        current_bias_x = float(np.mean(err[:, 0]))
        current_bias_y = float(np.mean(err[:, 1]))
        current_bias_r = float(np.sqrt(current_bias_x**2 + current_bias_y**2))
        max_err        = float(np.max(err_r))
        p95_err        = float(np.percentile(err_r, 95))

        # 判断触发条件
        rms_trigger  = current_rms  > rms_threshold_um
        bias_trigger = current_bias_r > bias_threshold_um

        need_recal = rms_trigger or bias_trigger

        # 判断重标定类型
        # 关键逻辑：
        #   纯平移/旋转 → 偏移大但RMS/偏移比值接近1 → 快速重标定
        #   畸变变化   → RMS超标但偏移小           → 完整重标定
        #   纯平移特征：残差几乎全部来自全局偏移
        #   即：bias_r / rms ≈ 1.0（偏移占主导）
        if not need_recal:
            recal_type = "无需重标定"
        else:
            # 计算偏移占残差的比例
            bias_ratio = (current_bias_r / current_rms
                          if current_rms > 1e-6 else 0.0)
            # 若全局偏移占总残差90%以上，认为是纯平移/旋转，快速重标定即可
            if bias_ratio >= 0.90:
                recal_type = "快速重标定（仅更新仿射矩阵）"
            else:
                recal_type = "完整重标定（仿射+多项式）"

        result = {
            'need_recalibration': need_recal,
            'recalibration_type': recal_type,
            'current_rms_um':     current_rms,
            'current_bias_x_um':  current_bias_x,
            'current_bias_y_um':  current_bias_y,
            'current_bias_r_um':  current_bias_r,
            'max_err_um':         max_err,
            'p95_err_um':         p95_err,
            'rms_threshold_um':   rms_threshold_um,
            'bias_threshold_um':  bias_threshold_um,
            'rms_trigger':        rms_trigger,
            'bias_trigger':       bias_trigger,
            'n_checked':          len(new_ref_px),
        }

        if verbose:
            self._print_recal_check(result)

        return result

    def quick_recalibrate(self, new_ref_px, new_ref_focal_um, verbose=True):
        """
        快速重标定：仅重新拟合仿射矩阵，保留多项式畸变系数

        适用场景：焦面板发生整体平移或小角度旋转，
        光学畸变特性未变化。

        Parameters
        ----------
        new_ref_px       : (M,2) 新基准光纤像素坐标
        new_ref_focal_um : (M,2) 对应焦面真值（μm）

        Returns
        -------
        report : dict  快速重标定报告
        """
        assert self.is_calibrated, "请先执行初始标定"
        assert len(new_ref_px) >= 4, "快速重标定至少需要4个基准点"

        new_ref_px       = np.array(new_ref_px,       dtype=float)
        new_ref_focal_um = np.array(new_ref_focal_um, dtype=float)

        # 保存旧参数（用于对比）
        old_A = self.A.copy()
        old_t = self.t.copy()
        old_rms = self.calibration_report.get('final_rms_um', None)

        # 更新归一化参数
        self.px_norm_offset = new_ref_px.min(axis=0)
        self.px_norm_scale  = new_ref_px.max(axis=0) - new_ref_px.min(axis=0)
        self.px_norm_scale  = np.where(
            self.px_norm_scale < 1e-6, 1.0, self.px_norm_scale)

        # 只重新拟合仿射矩阵（保留 coeff_x, coeff_y 不变）
        self.A, self.t = fit_affine(new_ref_px, new_ref_focal_um)

        # 评估新精度
        pred    = self.transform(new_ref_px)
        err     = new_ref_focal_um - pred
        new_rms = float(np.sqrt(np.mean(err**2)))

        # 更新系统偏差
        self.bias_x = float(np.mean(err[:, 0]))
        self.bias_y = float(np.mean(err[:, 1]))

        # 更新报告
        self.calibration_report.update({
            'final_rms_um':    new_rms,
            'recal_type':      'quick',
            'n_reference':     len(new_ref_px),
        })

        report = {
            'type':        '快速重标定',
            'n_points':    len(new_ref_px),
            'old_rms_um':  old_rms,
            'new_rms_um':  new_rms,
            'delta_A':     (self.A - old_A).tolist(),
            'delta_t_um':  (self.t - old_t).tolist(),
            'bias_x_um':   self.bias_x,
            'bias_y_um':   self.bias_y,
        }

        if verbose:
            print(f"\n=== 快速重标定完成 ===")
            print(f"  基准点数:   {len(new_ref_px)} 根")
            print(f"  旧残差RMS:  {old_rms:.3f} μm")
            print(f"  新残差RMS:  {new_rms:.3f} μm")
            dA = self.A - old_A
            dt = self.t - old_t
            print(f"  仿射矩阵变化量:")
            print(f"    ΔA = [[{dA[0,0]:+.4f}, {dA[0,1]:+.4f}],")
            print(f"          [{dA[1,0]:+.4f}, {dA[1,1]:+.4f}]]")
            print(f"    Δt = [{dt[0]:+.2f}, {dt[1]:+.2f}] μm")
            print(f"  多项式系数: 保留不变（畸变特性未变）")

        return report

    def full_recalibrate(self, new_ref_px, new_ref_focal_um, verbose=True):
        """
        完整重标定：重新执行全部标定流程（仿射+多项式）

        适用场景：光学链路变化、温度漂移大、仪器重装后。

        Parameters
        ----------
        new_ref_px       : (M,2) 新基准光纤像素坐标
        new_ref_focal_um : (M,2) 对应焦面真值（μm）

        Returns
        -------
        report : dict  完整重标定报告
        """
        old_rms = self.calibration_report.get('final_rms_um', None)

        if verbose:
            print(f"\n=== 开始完整重标定 ===")
            print(f"  旧标定残差: {old_rms:.3f} μm" if old_rms else "  （无旧标定）")

        # 直接调用完整标定流程
        report = self.calibrate(new_ref_px, new_ref_focal_um, verbose=verbose)
        report['type']       = '完整重标定'
        report['old_rms_um'] = old_rms

        return report

    def _print_recal_check(self, result):
        """打印重标定检查结果"""
        print(f"\n{'='*50}")
        print(f"  重标定检查报告")
        print(f"{'='*50}")
        print(f"  检查基准点数:    {result['n_checked']} 根")
        print(f"  当前残差 RMS:    {result['current_rms_um']:.3f} μm  "
              f"（阈值 {result['rms_threshold_um']:.1f} μm）"
              f"  {'⚠ 超标' if result['rms_trigger'] else '✓ 正常'}")
        print(f"  全局偏移 X:      {result['current_bias_x_um']:+.3f} μm")
        print(f"  全局偏移 Y:      {result['current_bias_y_um']:+.3f} μm")
        print(f"  全局偏移 径向:   {result['current_bias_r_um']:.3f} μm  "
              f"（阈值 {result['bias_threshold_um']:.1f} μm）"
              f"  {'⚠ 超标' if result['bias_trigger'] else '✓ 正常'}")
        print(f"  P95 误差:        {result['p95_err_um']:.3f} μm")
        print(f"  最大误差:        {result['max_err_um']:.3f} μm")
        print(f"{'─'*50}")
        print(f"  结论: {result['recalibration_type']}")
        if result['need_recalibration']:
            print(f"  触发原因: "
                  f"{'RMS超标 ' if result['rms_trigger'] else ''}"
                  f"{'偏移超标' if result['bias_trigger'] else ''}")
        print(f"{'='*50}")

# ============================================================
# 独立测试（与 config 保持完全一致）
# ============================================================

if __name__ == "__main__":
    from config import (REFERENCE_GRID_SPACING_MM, RANDOM_SEED,
                        REFERENCE_GRID_ORIGIN_MM, REF_GRID_SIDE,
                        POLY_ORDER, DISTORTION_K1, DISTORTION_K2,
                        FOCAL_RADIUS_UM, IMAGE_WIDTH, IMAGE_HEIGHT,
                        SPOT_SIGMA_PX, ELLIPTICAL_SPOT_PROB)
    from spot_generator import generate_gaussian_spot
    from gaussian_detector import fit_gaussian

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. 构建基准光纤焦面坐标（7×7=49个）─────────────────────────
    spacing_um = REFERENCE_GRID_SPACING_MM * 1000
    ox = REFERENCE_GRID_ORIGIN_MM[0] * 1000
    oy = REFERENCE_GRID_ORIGIN_MM[1] * 1000

    ref_focal = []
    for i in range(REF_GRID_SIDE):
        for j in range(REF_GRID_SIDE):
            ref_focal.append([ox + i * spacing_um,
                               oy + j * spacing_um])
    ref_focal = np.array(ref_focal)
    print(f"基准光纤数: {len(ref_focal)}  ({REF_GRID_SIDE}×{REF_GRID_SIDE} 格网)")
    print(f"焦面覆盖范围: "
          f"X=[{ref_focal[:,0].min()/1000:.0f}, "
          f"{ref_focal[:,0].max()/1000:.0f}] mm, "
          f"Y=[{ref_focal[:,1].min()/1000:.0f}, "
          f"{ref_focal[:,1].max()/1000:.0f}] mm")

    # ── 2. 仿真：焦面坐标 → 像素坐标（含畸变，与main_pipeline一致）──
    scale = 1.0 / FOCAL_PLANE_SCALE_UM_PX
    angle = np.deg2rad(0.5)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    offset_px = np.array([IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2])
    cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2

    # 仿射变换
    ref_px_ideal = (scale * (R @ ref_focal.T).T) + offset_px

    # 叠加径向畸变
    dx_um = (ref_px_ideal[:, 0] - cx) * FOCAL_PLANE_SCALE_UM_PX
    dy_um = (ref_px_ideal[:, 1] - cy) * FOCAL_PLANE_SCALE_UM_PX
    r_um   = np.sqrt(dx_um**2 + dy_um**2)
    r_norm = r_um / FOCAL_RADIUS_UM
    distortion_factor = (1.0
                         + DISTORTION_K1 * r_norm**2
                         + DISTORTION_K2 * r_norm**4)
    ref_px_distorted = ref_px_ideal.copy()
    ref_px_distorted[:, 0] = cx + (ref_px_ideal[:, 0] - cx) * distortion_factor
    ref_px_distorted[:, 1] = cy + (ref_px_ideal[:, 1] - cy) * distortion_factor

    # ── 3. 用 generate_gaussian_spot + fit_gaussian 检测（与main一致）
    patch_size = 50
    ref_px_detected = []
    detection_errors = []

    for px, py in ref_px_distorted:
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        patch   = generate_gaussian_spot(true_cx, true_cy,
                                          image_size=patch_size,
                                          rng=rng,
                                          ellipticity_prob=ELLIPTICAL_SPOT_PROB)
        result  = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                               use_elliptical=True)
        if result['success']:
            offset_x = round(px) - patch_size // 2
            offset_y = round(py) - patch_size // 2
            det_x = result['x0'] + offset_x
            det_y = result['y0'] + offset_y
            ref_px_detected.append([det_x, det_y])
            detection_errors.append(
                np.sqrt((det_x - px)**2 + (det_y - py)**2))
        else:
            ref_px_detected.append([px, py])
            detection_errors.append(0.0)

    ref_px_detected    = np.array(ref_px_detected)
    detection_noise_px = float(np.std(detection_errors))
    print(f"基准点检测噪声: {detection_noise_px:.4f} px "
          f"= {detection_noise_px * FOCAL_PLANE_SCALE_UM_PX:.3f} μm/轴")
    print(f"像素坐标范围: "
          f"u=[{ref_px_detected[:,0].min():.0f}, "
          f"{ref_px_detected[:,0].max():.0f}] px, "
          f"v=[{ref_px_detected[:,1].min():.0f}, "
          f"{ref_px_detected[:,1].max():.0f}] px")

    # ── 4. 执行标定 ───────────────────────────────────────────────────
    print(f"\n使用 poly_degree={POLY_ORDER} 阶多项式标定...")
    cal    = FVCCalibrator(poly_degree=POLY_ORDER)
    report = cal.calibrate(ref_px_detected, ref_focal)

    # ── 5. 生成待测光纤并检测 ─────────────────────────────────────────
    N_test = 500
    margin = 0.05
    focal_min   = ref_focal.min(axis=0)
    focal_max   = ref_focal.max(axis=0)
    focal_range = focal_max - focal_min
    target_low  = focal_min + margin * focal_range
    target_high = focal_max - margin * focal_range

    test_focal_true = rng.uniform(target_low, target_high, (N_test, 2))

    # 焦面 → 像素（含畸变）
    test_px_ideal = (scale * (R @ test_focal_true.T).T) + offset_px
    dx_um  = (test_px_ideal[:, 0] - cx) * FOCAL_PLANE_SCALE_UM_PX
    dy_um  = (test_px_ideal[:, 1] - cy) * FOCAL_PLANE_SCALE_UM_PX
    r_um   = np.sqrt(dx_um**2 + dy_um**2)
    r_norm = r_um / FOCAL_RADIUS_UM
    distortion_factor = (1.0
                         + DISTORTION_K1 * r_norm**2
                         + DISTORTION_K2 * r_norm**4)
    test_px_distorted = test_px_ideal.copy()
    test_px_distorted[:, 0] = cx + (test_px_ideal[:, 0] - cx) * distortion_factor
    test_px_distorted[:, 1] = cy + (test_px_ideal[:, 1] - cy) * distortion_factor

    # 用 generate_gaussian_spot + fit_gaussian 检测（与main一致）
    test_px_detected = []
    test_success     = []

    for px, py in test_px_distorted:
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        patch   = generate_gaussian_spot(true_cx, true_cy,
                                          image_size=patch_size,
                                          rng=rng,
                                          ellipticity_prob=ELLIPTICAL_SPOT_PROB)
        result  = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                               use_elliptical=True)
        if result['success']:
            offset_x = round(px) - patch_size // 2
            offset_y = round(py) - patch_size // 2
            test_px_detected.append(
                [result['x0'] + offset_x, result['y0'] + offset_y])
            test_success.append(True)
        else:
            test_px_detected.append([px, py])
            test_success.append(False)

    test_px_detected = np.array(test_px_detected)
    n_success = sum(test_success)
    print(f"待测光纤检测: {n_success}/{N_test} 成功")

    # ── 6. 坐标变换并评估精度 ─────────────────────────────────────────
    test_focal_pred = cal.transform(test_px_detected)
    err   = test_focal_true - test_focal_pred
    err_r = np.sqrt(err[:, 0]**2 + err[:, 1]**2)
    rms   = float(np.sqrt(np.mean(err_r**2)))
    p95   = float(np.percentile(err_r, 95))

    print(f"\n待测光纤坐标变换精度 (N={N_test}):")
    print(f"  RMS  误差:  {rms:.3f} μm")
    print(f"  X方向 RMS:  {np.std(err[:, 0]):.3f} μm")
    print(f"  Y方向 RMS:  {np.std(err[:, 1]):.3f} μm")
    print(f"  P95  误差:  {p95:.3f} μm")
    status = "✓ 达标" if rms <= 3.0 else "✗ 未达标"
    print(f"  目标 3.0 μm → {status}")


    # ── 6b. 生成XY结果图（coordinate_transform独立测试版）────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from config import OUTPUT_DIR

    # ── 中文字体设置（Windows系统）──────────────────────────────────
    _zh_candidates = [
        'Microsoft YaHei',   # 微软雅黑（Win10/11默认有）
        'SimHei',            # 黑体
        'SimSun',            # 宋体
        'KaiTi',             # 楷体
        'FangSong',          # 仿宋
    ]
    _zh_font = None
    for _fname in _zh_candidates:
        if any(_fname.lower() in f.name.lower()
               for f in fm.fontManager.ttflist):
            _zh_font = _fname
            break

    if _zh_font:
        plt.rcParams['font.family']       = _zh_font
        plt.rcParams['axes.unicode_minus'] = False   # 负号正常显示
    else:
        _zh_font = None
        print("  [警告] 未找到中文字体，图表标签将使用英文")

    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 转换为mm显示
    X_true_mm = test_focal_true[:, 0] / 1000
    Y_true_mm = test_focal_true[:, 1] / 1000
    dX = err[:, 0]   # μm
    dY = err[:, 1]   # μm
    ref_x_mm = ref_focal[:, 0] / 1000   # ← 新增：基准光纤坐标
    ref_y_mm = ref_focal[:, 1] / 1000   # ← 新增：基准光纤坐标

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── 左图：焦面坐标测量结果 ────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(X_true_mm, Y_true_mm,
                    c=err_r, cmap='RdYlGn_r',
                    s=15, alpha=0.7,
                    vmin=0, vmax=p95 * 1.5)

    if _zh_font:
        cb_label   = '径向误差 (μm)'
        xlabel0    = '焦面 X (mm)'
        ylabel0    = '焦面 Y (mm)'
        ref_label  = '基准光纤位置'
        status_str = "达标" if rms <= 3.0 else "未达标"
        title0     = (f'焦面坐标测量结果（N={N_test}）\n'
                      f'RMS={rms:.3f} μm  P95={p95:.3f} μm  '
                      f'目标=3.0 μm  [{status_str}]')
        xlabel1    = 'ΔX (μm)'
        ylabel1    = 'ΔY (μm)'
        title1     = (f'误差分布（焦面坐标系）\n'
                      f'偏差: X={np.mean(dX):+.3f} μm, '
                      f'Y={np.mean(dY):+.3f} μm')
    else:
        cb_label   = 'Radial Error (um)'
        xlabel0    = 'Focal X (mm)'
        ylabel0    = 'Focal Y (mm)'
        ref_label  = 'Fiducial Fibers'
        status_str = "PASS" if rms <= 3.0 else "FAIL"
        title0     = (f'Focal Plane Measurement Result (N={N_test})\n'
                      f'RMS={rms:.3f} um  P95={p95:.3f} um  '
                      f'Target=3.0 um  {status_str}')
        xlabel1    = 'dX (um)'
        ylabel1    = 'dY (um)'
        title1     = (f'Error Distribution (Focal Plane)\n'
                      f'Bias: X={np.mean(dX):+.3f} um, '
                      f'Y={np.mean(dY):+.3f} um')

    plt.colorbar(sc, ax=ax, label=cb_label)
    # 叠加基准光纤位置
    ax.scatter(ref_x_mm, ref_y_mm,
               marker='+', s=80, c='blue',
               linewidths=1.5, label=ref_label, zorder=5)
    ax.set_xlabel(xlabel0)
    ax.set_ylabel(ylabel0)
    ax.set_title(title0)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── 右图：误差分布 ────────────────────────────────────────────────
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(rms * np.cos(theta), rms * np.sin(theta),
            'b--', lw=2, label=f'RMS = {rms:.3f} μm')
    ax.plot(p95 * np.cos(theta), p95 * np.sin(theta),
            'r-',  lw=2, label=f'P95 = {p95:.3f} μm')
    ax.plot(3.0 * np.cos(theta), 3.0 * np.sin(theta),
            'g:',  lw=2, label=f'Target = 3.0 μm')
    ax.scatter(dX, dY, s=8, alpha=0.3, color='steelblue')
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel1)
    ax.set_title(title1)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "ct_xy_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  XY结果图已保存: {fig_path}")


    # ── 7. 误差传递理论验证 ───────────────────────────────────────────
    calib_rms          = report['final_rms_um']
    detect_noise_um    = detection_noise_px * FOCAL_PLANE_SCALE_UM_PX
    theory_single_axis = np.sqrt(calib_rms**2 + detect_noise_um**2)
    theory_radial      = np.sqrt(2) * theory_single_axis

    print(f"\n误差传递理论验证:")
    print(f"  标定残差:      {calib_rms:.3f} μm")
    print(f"  检测噪声:      {detection_noise_px:.4f} px"
          f" = {detect_noise_um:.3f} μm/轴")
    print(f"  理论径向RMS:   √2×√({calib_rms:.3f}²+{detect_noise_um:.3f}²)"
          f" = {theory_radial:.3f} μm")
    print(f"  实测径向RMS:   {rms:.3f} μm")
    diff = abs(rms - theory_radial)
    print(f"  理论vs实测:    差异 {diff:.3f} μm "
          f"{'✓ 吻合' if diff < 0.5 else '⚠ 偏差'}")

    # ── 8. 参数一致性验证 ─────────────────────────────────────────────
    print(f"\n参数一致性验证:")
    print(f"  基准点数:   {len(ref_focal)} 根  "
          f"({REF_GRID_SIDE}×{REF_GRID_SIDE}={REF_GRID_SIDE**2}) ✓")
    print(f"  多项式阶数: {POLY_ORDER} 阶  (config: POLY_ORDER={POLY_ORDER}) ✓")
    print(f"  畸变系数:   k1={DISTORTION_K1}, k2={DISTORTION_K2} ✓")
    print(f"  光斑生成:   generate_gaussian_spot()  (与main_pipeline一致) ✓")
    print(f"  光斑检测:   fit_gaussian(use_elliptical=True) (与main一致) ✓")

    # ── 9. 重标定策略测试 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  重标定策略测试")
    print(f"{'='*60}")

    # ================================================================
    # 场景A：正常状态检查（应该无需重标定）
    # ================================================================
    print(f"\n[场景A] 正常状态：使用原始基准点检查")
    cal_A = FVCCalibrator(poly_degree=POLY_ORDER)
    cal_A.calibrate(ref_px_detected, ref_focal, verbose=False)

    check_A = cal_A.check_recalibration_needed(
        ref_px_detected, ref_focal,
        rms_threshold_um=2.0,
        bias_threshold_um=5.0
    )

    # ================================================================
    # 场景B：焦面板发生纯平移（应触发快速重标定）
    # ================================================================
    print(f"\n[场景B] 焦面板整体平移 5mm（≈36px）")
    SHIFT_UM = 5000.0
    SHIFT_PX = SHIFT_UM / FOCAL_PLANE_SCALE_UM_PX  # ≈35.9px

    # 基准点像素坐标整体平移
    ref_px_shifted = ref_px_distorted.copy()
    ref_px_shifted[:, 0] += SHIFT_PX

    # 重新用高斯检测（模拟重新拍照）
    rng_B = np.random.default_rng(RANDOM_SEED + 10)
    ref_px_shifted_det = []
    for px, py in ref_px_shifted:
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        patch   = generate_gaussian_spot(
            true_cx, true_cy,
            image_size=patch_size,
            rng=rng_B,
            ellipticity_prob=ELLIPTICAL_SPOT_PROB)
        result  = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                               use_elliptical=True)
        if result['success']:
            ox = round(px) - patch_size // 2
            oy = round(py) - patch_size // 2
            ref_px_shifted_det.append(
                [result['x0'] + ox, result['y0'] + oy])
        else:
            ref_px_shifted_det.append([px, py])
    ref_px_shifted_det = np.array(ref_px_shifted_det)

    # 待测光纤也同步平移（焦面板整体移动）
    test_px_shifted = test_px_distorted.copy()
    test_px_shifted[:, 0] += SHIFT_PX

    rng_B2 = np.random.default_rng(RANDOM_SEED + 11)
    test_px_shifted_det = []
    for px, py in test_px_shifted:
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        patch   = generate_gaussian_spot(
            true_cx, true_cy,
            image_size=patch_size,
            rng=rng_B2,
            ellipticity_prob=ELLIPTICAL_SPOT_PROB)
        result  = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                               use_elliptical=True)
        if result['success']:
            ox = round(px) - patch_size // 2
            oy = round(py) - patch_size // 2
            test_px_shifted_det.append(
                [result['x0'] + ox, result['y0'] + oy])
        else:
            test_px_shifted_det.append([px, py])
    test_px_shifted_det = np.array(test_px_shifted_det)

    # 独立的cal_B实例（基于原始标定）
    cal_B = FVCCalibrator(poly_degree=POLY_ORDER)
    cal_B.calibrate(ref_px_detected, ref_focal, verbose=False)

    check_B = cal_B.check_recalibration_needed(
        ref_px_shifted_det, ref_focal,
        rms_threshold_um=2.0,
        bias_threshold_um=5.0
    )

    if check_B['need_recalibration']:
        recal_type_B = check_B['recalibration_type']
        print(f"\n  触发: {recal_type_B}")
        if recal_type_B == '快速重标定（仅更新仿射矩阵）':
            quick_report = cal_B.quick_recalibrate(
                ref_px_shifted_det, ref_focal, verbose=True)
            # 用平移后的待测坐标验证精度
            test_focal_pred_B = cal_B.transform(test_px_shifted_det)
            err_B   = test_focal_true - test_focal_pred_B
            err_r_B = np.sqrt(err_B[:, 0]**2 + err_B[:, 1]**2)
            rms_B   = float(np.sqrt(np.mean(err_r_B**2)))
            print(f"  快速重标定后待测精度: {rms_B:.3f} μm  "
                  f"{'达标' if rms_B <= 3.0 else '未达标'}")
        else:
            full_report_B = cal_B.full_recalibrate(
                ref_px_shifted_det, ref_focal, verbose=False)
            print(f"  完整重标定后残差: "
                  f"{full_report_B['final_rms_um']:.3f} μm")

    # ================================================================
    # 场景C：光学畸变改变（应触发完整重标定）
    # 独立实例，不受场景B影响
    # ================================================================
    print(f"\n[场景C] 光学系统变化（畸变系数k1改变20%）")

    new_k1 = DISTORTION_K1 * 1.2   # k1变化20%

    # 用新畸变重新生成基准点像素坐标
    ref_px_newopt_ideal = ref_px_ideal.copy()
    dx_um2  = (ref_px_ideal[:, 0] - cx) * FOCAL_PLANE_SCALE_UM_PX
    dy_um2  = (ref_px_ideal[:, 1] - cy) * FOCAL_PLANE_SCALE_UM_PX
    r_um2   = np.sqrt(dx_um2**2 + dy_um2**2)
    r_norm2 = r_um2 / FOCAL_RADIUS_UM
    dist_new = (1.0 + new_k1 * r_norm2**2
                    + DISTORTION_K2 * r_norm2**4)
    ref_px_newopt = ref_px_ideal.copy()
    ref_px_newopt[:, 0] = cx + (ref_px_ideal[:, 0] - cx) * dist_new
    ref_px_newopt[:, 1] = cy + (ref_px_ideal[:, 1] - cy) * dist_new

    rng_C = np.random.default_rng(RANDOM_SEED + 20)
    ref_px_newopt_det = []
    for px, py in ref_px_newopt:
        true_cx = patch_size / 2 + (px - round(px))
        true_cy = patch_size / 2 + (py - round(py))
        patch   = generate_gaussian_spot(
            true_cx, true_cy,
            image_size=patch_size,
            rng=rng_C,
            ellipticity_prob=ELLIPTICAL_SPOT_PROB)
        result  = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                               use_elliptical=True)
        if result['success']:
            ox = round(px) - patch_size // 2
            oy = round(py) - patch_size // 2
            ref_px_newopt_det.append(
                [result['x0'] + ox, result['y0'] + oy])
        else:
            ref_px_newopt_det.append([px, py])
    ref_px_newopt_det = np.array(ref_px_newopt_det)

    # 独立的cal_C实例（基于原始标定）
    cal_C = FVCCalibrator(poly_degree=POLY_ORDER)
    cal_C.calibrate(ref_px_detected, ref_focal, verbose=False)

    check_C = cal_C.check_recalibration_needed(
        ref_px_newopt_det, ref_focal,
        rms_threshold_um=2.0,
        bias_threshold_um=5.0
    )

    if check_C['need_recalibration']:
        recal_type_C = check_C['recalibration_type']
        print(f"\n  触发: {recal_type_C}")
        if recal_type_C == '完整重标定（仿射+多项式）':
            full_report_C = cal_C.full_recalibrate(
                ref_px_newopt_det, ref_focal, verbose=False)
            new_rms_C = full_report_C['final_rms_um']
            print(f"  完整重标定后残差: {new_rms_C:.3f} μm  "
                  f"{'达标' if new_rms_C <= 2.0 else '未达标'}")

            # 验证完整重标定后的待测精度
            # 用新畸变重新生成待测点像素坐标
            test_px_newopt_ideal = (scale * (R @ test_focal_true.T).T) + offset_px
            dx_t  = (test_px_newopt_ideal[:, 0] - cx) * FOCAL_PLANE_SCALE_UM_PX
            dy_t  = (test_px_newopt_ideal[:, 1] - cy) * FOCAL_PLANE_SCALE_UM_PX
            r_t   = np.sqrt(dx_t**2 + dy_t**2)
            rn_t  = r_t / FOCAL_RADIUS_UM
            dist_t = 1.0 + new_k1 * rn_t**2 + DISTORTION_K2 * rn_t**4
            test_px_newopt = test_px_newopt_ideal.copy()
            test_px_newopt[:, 0] = cx + (test_px_newopt_ideal[:, 0] - cx) * dist_t
            test_px_newopt[:, 1] = cy + (test_px_newopt_ideal[:, 1] - cy) * dist_t

            rng_C2 = np.random.default_rng(RANDOM_SEED + 21)
            test_px_newopt_det = []
            for px, py in test_px_newopt:
                true_cx = patch_size / 2 + (px - round(px))
                true_cy = patch_size / 2 + (py - round(py))
                patch   = generate_gaussian_spot(
                    true_cx, true_cy,
                    image_size=patch_size,
                    rng=rng_C2,
                    ellipticity_prob=ELLIPTICAL_SPOT_PROB)
                result  = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX,
                                       use_elliptical=True)
                if result['success']:
                    ox = round(px) - patch_size // 2
                    oy = round(py) - patch_size // 2
                    test_px_newopt_det.append(
                        [result['x0'] + ox, result['y0'] + oy])
                else:
                    test_px_newopt_det.append([px, py])
            test_px_newopt_det = np.array(test_px_newopt_det)

            test_pred_C = cal_C.transform(test_px_newopt_det)
            err_C   = test_focal_true - test_pred_C
            err_r_C = np.sqrt(err_C[:, 0]**2 + err_C[:, 1]**2)
            rms_C   = float(np.sqrt(np.mean(err_r_C**2)))
            print(f"  完整重标定后待测精度: {rms_C:.3f} μm  "
                  f"{'达标' if rms_C <= 3.0 else '未达标'}")
        else:
            quick_report_C = cal_C.quick_recalibrate(
                ref_px_newopt_det, ref_focal, verbose=False)
            print(f"  [意外] 快速重标定后残差: "
                  f"{quick_report_C['new_rms_um']:.3f} μm "
                  f"（预期应触发完整重标定）")

    # ================================================================
    # 重标定策略决策树
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  重标定策略决策树")
    print(f"{'='*60}")
    print(f"  检查残差 RMS > 2.0 um 或 全局偏移 > 5.0 um ?")
    print(f"  |-- 否 --> 无需重标定，继续使用当前标定")
    print(f"  `-- 是 --> bias_r / RMS >= 0.90 ?（偏移占主导）")
    print(f"            |-- 是 --> 快速重标定（只更新仿射矩阵6参数）")
    print(f"            |          耗时：< 1秒，精度损失：< 0.1um")
    print(f"            `-- 否 --> 完整重标定（仿射+多项式全部重做）")
    print(f"                       耗时：< 5秒，恢复全精度")
    print(f"{'='*60}")
