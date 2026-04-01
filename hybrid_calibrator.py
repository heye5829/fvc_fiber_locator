"""
hybrid_calibrator.py
混合标定器：全局多项式 + 局部RBF修正
适用于大视场（> 500 mm）的高精度标定
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.interpolate import RBFInterpolator


class HybridCalibrator:
    """
    混合标定器：Polynomial + RBF

    策略：
    1. 全局多项式拟合主要畸变
    2. RBF插值拟合残差（局部修正）
    """

    def __init__(self, poly_degree=5, rbf_kernel='thin_plate_spline',
                 rbf_smoothing=0.1, rbf_epsilon=None):
        """
        Parameters
        ----------
        poly_degree : int
            全局多项式阶数（推荐4-5阶）
        rbf_kernel : str
            RBF核函数，可选：
            - 'thin_plate_spline' (推荐，DESI使用)
            - 'multiquadric'
            - 'gaussian'
        rbf_smoothing : float
            RBF平滑参数（0=精确插值，>0=平滑拟合）
        rbf_epsilon : float or None
            RBF形状参数（None=自动选择）
        """
        self.poly_degree = poly_degree
        self.rbf_kernel = rbf_kernel
        self.rbf_smoothing = rbf_smoothing
        self.rbf_epsilon = rbf_epsilon

        # 模型组件
        self.poly_x = None
        self.poly_y = None
        self.rbf_x = None
        self.rbf_y = None
        self.poly_features = None

        # 标定状态
        self.is_calibrated = False
        self.calib_residual_rms = None
        self.poly_residual_rms = None

    def calibrate(self, ref_px, ref_focal_um, verbose=True):
        """
        执行混合标定

        Parameters
        ----------
        ref_px : array (N, 2)
            基准点像素坐标 [u, v]
        ref_focal_um : array (N, 2)
            基准点焦面坐标 [X, Y] (μm)
        verbose : bool
            是否打印详细信息

        Returns
        -------
        report : dict
            标定报告
        """
        N = len(ref_px)

        if verbose:
            print(f"\n{'=' * 60}")
            print("混合标定器：Polynomial + RBF")
            print(f"{'=' * 60}")
            print(f"  基准点数: {N}")
            print(f"  多项式阶数: {self.poly_degree}")
            print(f"  RBF核函数: {self.rbf_kernel}")

        # ============================================================
        # 步骤1：全局多项式拟合
        # ============================================================
        self.poly_features = PolynomialFeatures(degree=self.poly_degree)
        X_poly = self.poly_features.fit_transform(ref_px)

        # 使用Ridge回归（带正则化，防止过拟合）
        self.poly_x = Ridge(alpha=1.0).fit(X_poly, ref_focal_um[:, 0])
        self.poly_y = Ridge(alpha=1.0).fit(X_poly, ref_focal_um[:, 1])

        # 计算多项式残差
        focal_poly = np.column_stack([
            self.poly_x.predict(X_poly),
            self.poly_y.predict(X_poly)
        ])
        residuals_poly = ref_focal_um - focal_poly
        self.poly_residual_rms = np.sqrt(np.mean(residuals_poly ** 2))

        if verbose:
            print(f"  多项式残差 RMS: {self.poly_residual_rms:.3f} μm")

        # ============================================================
        # 步骤2：RBF拟合残差（局部修正）
        # ============================================================
        # 只有当残差 > 1.0 μm 时才使用RBF修正
        if self.poly_residual_rms > 1.0:
            self.rbf_x = RBFInterpolator(
                ref_px,
                residuals_poly[:, 0],
                kernel=self.rbf_kernel,
                smoothing=self.rbf_smoothing,
                epsilon=self.rbf_epsilon
            )
            self.rbf_y = RBFInterpolator(
                ref_px,
                residuals_poly[:, 1],
                kernel=self.rbf_kernel,
                smoothing=self.rbf_smoothing,
                epsilon=self.rbf_epsilon
            )

            # 计算混合模型残差
            focal_hybrid = focal_poly + np.column_stack([
                self.rbf_x(ref_px),
                self.rbf_y(ref_px)
            ])
            residuals_hybrid = ref_focal_um - focal_hybrid
            self.calib_residual_rms = np.sqrt(np.mean(residuals_hybrid ** 2))

            if verbose:
                improvement = (1 - self.calib_residual_rms / self.poly_residual_rms) * 100
                print(f"  RBF修正后 RMS: {self.calib_residual_rms:.3f} μm")
                print(f"  精度提升: {improvement:.1f}%")
        else:
            # 残差已经很小，不需要RBF修正
            self.rbf_x = None
            self.rbf_y = None
            self.calib_residual_rms = self.poly_residual_rms
            if verbose:
                print(f"  多项式精度已足够（{self.poly_residual_rms:.3f} μm），跳过RBF修正")

        self.is_calibrated = True

        return {
            'n_calib': N,
            'poly_degree': self.poly_degree,
            'poly_residual_rms': self.poly_residual_rms,
            'final_residual_rms': self.calib_residual_rms,
            'use_rbf': self.rbf_x is not None
        }

    def pixel_to_focal(self, px):
        """
        像素坐标 → 焦面坐标

        Parameters
        ----------
        px : array (..., 2)
            像素坐标 [u, v]

        Returns
        -------
        focal_um : array (..., 2)
            焦面坐标 [X, Y] (μm)
        """
        if not self.is_calibrated:
            raise RuntimeError("标定器未初始化，请先调用 calibrate()")

        px = np.atleast_2d(px)

        # 步骤1：多项式变换
        X_poly = self.poly_features.transform(px)
        focal_x = self.poly_x.predict(X_poly)
        focal_y = self.poly_y.predict(X_poly)

        # 步骤2：RBF修正（如果有）
        if self.rbf_x is not None:
            focal_x += self.rbf_x(px)
            focal_y += self.rbf_y(px)

        return np.column_stack([focal_x, focal_y])