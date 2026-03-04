"""
坐标变换模块
实现：像素坐标 → 仿射变换 → 多项式畸变校正 → 焦面物理坐标(μm)

流程：
  1. 用基准光纤（已知焦面坐标）标定仿射矩阵 + 畸变系数
  2. 对待测光纤像素坐标做逆变换，得到焦面物理坐标
"""
import numpy as np
from numpy.polynomial import polynomial as P
from config import (FOCAL_PLANE_SCALE_UM_PX, DISTORTION_POLY_DEGREE,
                    PIXEL_SIZE_UM, DEMAGNIFICATION)


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


def fit_distortion(affine_residuals_um, px_coords_norm, degree=DISTORTION_POLY_DEGREE):
    F = poly_features(px_coords_norm, degree)
    coeff_x, _, _, _ = np.linalg.lstsq(F, affine_residuals_um[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(F, affine_residuals_um[:, 1], rcond=None)
    return coeff_x, coeff_y


def apply_distortion_correction(px_coords_norm, coeff_x, coeff_y,
                                degree=DISTORTION_POLY_DEGREE):
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

    def __init__(self, poly_degree=DISTORTION_POLY_DEGREE):
        self.poly_degree = poly_degree
        self.A = None
        self.t = None
        self.coeff_x = None
        self.coeff_y = None
        self.px_norm_scale = None
        self.px_norm_offset = None
        self.is_calibrated = False
        self.calibration_report = {}

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

        # 4. 验证总体精度
        final_pred = self.transform(ref_px)
        final_err = ref_focal_um - final_pred
        rms_final = float(np.sqrt(np.mean(final_err ** 2)))

        self.calibration_report = {
            'n_reference': N,
            'affine_rms_um': float(rms_affine),
            'final_rms_um': float(rms_final),
            'poly_degree': self.poly_degree,
        }

        if verbose:
            print(f"=== FVC 标定完成 ===")
            print(f"  基准光纤数: {N}")
            print(f"  仿射残差 RMS: {rms_affine:.3f} um")
            print(f"  最终残差 RMS: {rms_final:.3f} um (含畸变校正)")

        return self.calibration_report

    def transform(self, px_coords):
        assert self.is_calibrated, "请先调用 calibrate()"
        px = np.atleast_2d(px_coords).astype(float)
        focal = apply_affine(px, self.A, self.t)
        px_norm = self._normalize_px(px)
        correction = apply_distortion_correction(px_norm, self.coeff_x, self.coeff_y,
                                                 degree=self.poly_degree)
        focal += correction
        return focal

    def transform_with_uncertainty(self, px_coords, px_error_px=0.023):
        focal_um = self.transform(px_coords)
        scale_um_per_px = FOCAL_PLANE_SCALE_UM_PX
        uncertainty = px_error_px * scale_um_per_px * np.ones(len(focal_um))
        return focal_um, uncertainty


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    from config import REFERENCE_GRID_SPACING_MM, RANDOM_SEED

    rng = np.random.default_rng(RANDOM_SEED)

    spacing_um = REFERENCE_GRID_SPACING_MM * 1000
    ref_focal = []
    for i in range(3):
        for j in range(3):
            ref_focal.append([i * spacing_um, j * spacing_um])
    ref_focal = np.array(ref_focal)

    scale = 1.0 / FOCAL_PLANE_SCALE_UM_PX
    angle = np.deg2rad(0.5)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    ref_px = (scale * (R @ ref_focal.T).T
              + np.array([1000.0, 800.0])
              + rng.normal(0, 0.01, ref_focal.shape))

    cal = FVCCalibrator(poly_degree=2)
    report = cal.calibrate(ref_px, ref_focal)

    N_test = 50
    test_focal_true = rng.uniform([0, 0], [2 * spacing_um, 2 * spacing_um], (N_test, 2))
    test_px = (scale * (R @ test_focal_true.T).T
               + np.array([1000.0, 800.0])
               + rng.normal(0, 0.05, test_focal_true.shape))

    test_focal_pred = cal.transform(test_px)
    err = test_focal_true - test_focal_pred
    rms = float(np.sqrt(np.mean(err ** 2)))

    print(f"\n待测光纤坐标变换精度 (N={N_test}):")
    print(f"  RMS误差: {rms:.3f} um")
    print(f"  X方向RMS: {np.std(err[:, 0]):.3f} um")
    print(f"  Y方向RMS: {np.std(err[:, 1]):.3f} um")

