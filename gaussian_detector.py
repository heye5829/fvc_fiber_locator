"""
高斯拟合质心检测器
使用2D高斯拟合实现亚像素精度质心定位
目标：在焦面坐标系下达到 3μm 测量精度
"""
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from config import SPOT_SIGMA_PX, FIT_WINDOW_SIGMA, MIN_SNR

# photutils可选导入（安装后自动启用，未安装则降级到scipy）
try:
    from photutils.centroids import centroid_2dg
    HAS_PHOTUTILS = True
except ImportError:
    HAS_PHOTUTILS = False


# ============================================================
# 高斯模型
# ============================================================

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, background):
    """
    椭圆高斯模型（含旋转角theta）
    xy : (2, N) — meshgrid展平后的坐标
    """
    x, y = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = cos_t ** 2 / (2 * sigma_x ** 2) + sin_t ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = sin_t ** 2 / (2 * sigma_x ** 2) + cos_t ** 2 / (2 * sigma_y ** 2)
    dx, dy = x - x0, y - y0
    z = background + amplitude * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
    return z


def gaussian_2d_sym(xy, amplitude, x0, y0, sigma, background):
    """
    圆对称高斯模型（参数更少，更稳健）
    """
    x, y = xy
    z = background + amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return z


# ============================================================
# 预处理
# ============================================================

def estimate_background(patch, border=3):
    """用patch边缘像素估计背景"""
    mask = np.ones(patch.shape, dtype=bool)
    mask[border:-border, border:-border] = False
    bg = np.median(patch[mask])
    return float(bg)


def compute_snr(patch, background):
    """计算峰值SNR"""
    peak = float(np.max(patch) - background)
    noise = float(np.std(patch[:3, :3]))  # 用角落估计噪声
    if noise < 1e-6:
        noise = np.sqrt(background + 1)
    return peak / noise


def centroid_initial_guess(patch, background):
    """矩方法给高斯拟合提供初始值"""
    data = np.maximum(patch - background, 0)
    total = data.sum()
    if total < 1e-6:
        h, w = patch.shape
        return w / 2, h / 2
    y_idx, x_idx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = (x_idx * data).sum() / total
    cy = (y_idx * data).sum() / total
    return float(cx), float(cy)


# ============================================================
# 主拟合函数
# ============================================================

def fit_gaussian(patch, sigma_init=None, use_elliptical=False, use_photutils=True):
    """
    对patch进行2D高斯拟合（增强版：加入倾斜平面背景补偿，冲击 0.02 px 精度）

    Parameters
    ----------
    patch          : np.ndarray (H, W)  光斑图像patch
    sigma_init     : float  初始sigma估计（像素），None则用config值
    use_elliptical : bool   是否使用椭圆高斯（参数更多但更精确）
    use_photutils  : bool   是否优先使用photutils（已安装时生效）

    Returns
    -------
    result : dict  包含 x0, y0, sigma_x, sigma_y, amplitude, background, snr, success

    【修改说明】:
    1. 修复了 warnings 作用域导致的 UnboundLocalError
    2. 在 scipy 拟合分支中加入了倾斜平面背景模型 (bg_x*x + bg_y*y + bg_const)
    3. 保持了与原代码完全一致的函数签名和返回格式，确保兼容性
    """
    import warnings  # 局部导入，彻底避免作用域问题

    if sigma_init is None:
        sigma_init = SPOT_SIGMA_PX

    H, W = patch.shape

    # 背景估计
    bg = estimate_background(patch)
    snr = compute_snr(patch, bg)
    amplitude_init = float(np.max(patch) - bg)

    # 初始质心
    cx_init, cy_init = centroid_initial_guess(patch, bg)

    result = {
        'x0': cx_init, 'y0': cy_init,
        'sigma_x': sigma_init, 'sigma_y': sigma_init,
        'amplitude': amplitude_init, 'background': bg,
        'snr': snr, 'success': False, 'residual_rms': np.inf
    }

    if snr < MIN_SNR * 0.5:
        # SNR太低，直接返回矩估计
        result['x0'] = cx_init
        result['y0'] = cy_init
        return result

    # ── 优先尝试 photutils ────────────────────────────────
    if use_photutils and HAS_PHOTUTILS and not use_elliptical:
        try:
            img = patch.astype(float)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The fit may not have converged.*")
                x0, y0 = centroid_2dg(img)

            if np.isfinite(x0) and np.isfinite(y0):
                peak = float(img.max())
                bg_val = float(np.median(img))
                amp_val = peak - bg_val

                ok = is_valid_centroid_result(
                    x0, y0, img.shape,
                    amplitude=amp_val,
                    snr=snr,
                    sigma=sigma_init
                )

                result.update({
                    'x0': float(x0),
                    'y0': float(y0),
                    'sigma_x': sigma_init,
                    'sigma_y': sigma_init,
                    'amplitude': amp_val,
                    'background': bg_val,
                    'snr': snr,
                    'residual_rms': 0.0,
                    'engine': 'photutils'
                })
                result['success'] = is_valid_fit_result(result, patch.shape)
                return result
        except Exception:
            pass

    # ── scipy实现（增强版：加入倾斜平面背景补偿）─────────────────────
    y_arr, x_arr = np.mgrid[0:H, 0:W]
    xy = (x_arr.ravel().astype(float), y_arr.ravel().astype(float))
    z = patch.ravel().astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)

            if use_elliptical:
                # 椭圆高斯（保持原逻辑不变）
                p0 = [amplitude_init, cx_init, cy_init,
                      sigma_init, sigma_init, 0.0, bg]
                bounds = (
                    [0, 0, 0, 0.5, 0.5, -np.pi / 4, 0],
                    [amplitude_init * 3, W, H, sigma_init * 5, sigma_init * 5, np.pi / 4, bg * 3 + 100]
                )
                popt, pcov = curve_fit(gaussian_2d, xy, z, p0=p0,
                                       bounds=bounds, maxfev=2000)
                perr = np.sqrt(np.diag(pcov))
                result.update({
                    'amplitude': popt[0], 'x0': popt[1], 'y0': popt[2],
                    'sigma_x': abs(popt[3]), 'sigma_y': abs(popt[4]),
                    'theta': popt[5], 'background': popt[6],
                    'x0_err': perr[1], 'y0_err': perr[2],
                    'engine': 'scipy_elliptical'
                })
                result['success'] = is_valid_fit_result(result, patch.shape)

            else:
                # ★★★ 圆对称高斯 + 倾斜平面背景（核心改进）★★★
                # 新模型: A*exp(...) + bg_x*x + bg_y*y + bg_const
                def gaussian_2d_sym_with_plane(coords, amplitude, xo, yo, sigma, bg_x, bg_y, bg_const):
                    x_c, y_c = coords
                    g = amplitude * np.exp(-((x_c - xo) ** 2 + (y_c - yo) ** 2) / (2 * sigma ** 2))
                    plane = bg_x * x_c + bg_y * y_c + bg_const
                    return g + plane

                # 初值：[振幅, x中心, y中心, sigma, X梯度, Y梯度, 常数背景]
                p0 = [amplitude_init, cx_init, cy_init, sigma_init, 0.0, 0.0, bg]

                # 边界约束
                bounds = (
                    [0, 0, 0, 0.5, -np.inf, -np.inf, 0],
                    [amplitude_init * 3, W, H, sigma_init * 5, np.inf, np.inf, bg * 3 + 100]
                )

                popt, pcov = curve_fit(gaussian_2d_sym_with_plane, xy, z, p0=p0,
                                       bounds=bounds, maxfev=2000)
                perr = np.sqrt(np.diag(pcov))

                # 拟合残差
                z_fit = gaussian_2d_sym_with_plane(xy, *popt)
                residual_rms = float(np.sqrt(np.mean((z - z_fit) ** 2)))

                result.update({
                    'amplitude': popt[0],
                    'x0': popt[1],
                    'y0': popt[2],
                    'sigma_x': popt[3],
                    'sigma_y': popt[3],
                    'background': popt[6],  # 使用常数项作为背景
                    'x0_err': perr[1],
                    'y0_err': perr[2],
                    'residual_rms': residual_rms,
                    'engine': 'scipy_enhanced'
                })
                result['success'] = is_valid_fit_result(result, patch.shape)

    except (RuntimeError, ValueError) as e:
        # 拟合失败，回退到质心法
        result['x0'] = cx_init
        result['y0'] = cy_init
        result['fit_error'] = str(e)

    return result


# ============================================================
# 批量检测器
# ============================================================

class GaussianDetector:
    """
    批量高斯拟合质心检测器

    Usage
    -----
    detector = GaussianDetector()
    positions = detector.detect_all(image, seed_positions)
    """

    def __init__(self, window_sigma=FIT_WINDOW_SIGMA,
                 spot_sigma=SPOT_SIGMA_PX,
                 # use_elliptical=False):
                 use_elliptical=False,
                 use_photutils=False):  #  新增参数，默认关闭 photutils
        self.half_win = int(np.ceil(window_sigma * spot_sigma)) + 1
        self.spot_sigma = spot_sigma
        self.use_elliptical = use_elliptical
        self.use_photutils = use_photutils  #  保存为实例变量

    def detect_single(self, image, seed_x, seed_y):
        """
        检测单个光斑

        Parameters
        ----------
        image          : np.ndarray (H, W)
        seed_x, seed_y : float  粗略质心位置（像素）

        Returns
        -------
        result : dict  含 'x_global', 'y_global' 全图坐标
        """
        from spot_generator import extract_patch

        patch, off_x, off_y = extract_patch(image, seed_x, seed_y, self.half_win)

        if patch.size == 0:
            return {'x_global': seed_x, 'y_global': seed_y, 'success': False}

        result = fit_gaussian(patch, sigma_init=self.spot_sigma,
                              # use_elliptical=self.use_elliptical)
                              use_elliptical=self.use_elliptical,
                              use_photutils=self.use_photutils)  #  传入参数

        # 转回全图坐标
        result['x_global'] = result['x0'] + off_x
        result['y_global'] = result['y0'] + off_y

        return result

    def detect_all(self, image, seed_positions):
        """
        批量检测

        Parameters
        ----------
        seed_positions : list of (x, y)  种子坐标列表

        Returns
        -------
        results        : list of dict
        detected_xy    : np.ndarray (N, 2)  仅成功检测的坐标
        """
        results = []
        for sx, sy in seed_positions:
            r = self.detect_single(image, sx, sy)
            results.append(r)

        detected_xy = np.array([
            [r['x_global'], r['y_global']]
            for r in results if r.get('success', False)
        ])

        return results, detected_xy


# ============================================================
# 质心法 + 降级策略
# ============================================================

def centroid_weighted(patch, threshold_factor=0.3):
    """
    加权质心法（重心法）

    精度低于高斯拟合（约差3-5倍），但速度快5-10倍，
    在高斯拟合失败时作为降级备选。

    Parameters
    ----------
    patch            : 2D ndarray  光斑图像patch
    threshold_factor : float       0~1，去背景阈值比例

    Returns
    -------
    dict 包含 x0, y0, success, method
    """
    img = patch.astype(float)

    img_min, img_max = img.min(), img.max()
    if img_max <= img_min:
        return {
            'x0': img.shape[1] / 2.0,
            'y0': img.shape[0] / 2.0,
            'success': False,
            'method': 'centroid'
        }

    threshold = img_min + threshold_factor * (img_max - img_min)
    weights = np.maximum(img - threshold, 0.0)
    total = weights.sum()

    if total < 1e-10:
        return {
            'x0': img.shape[1] / 2.0,
            'y0': img.shape[0] / 2.0,
            'success': False,
            'method': 'centroid'
        }

    y_arr, x_arr = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    x0 = float((weights * x_arr).sum() / total)
    y0 = float((weights * y_arr).sum() / total)

    return {
        'x0': x0,
        'y0': y0,
        'success': True,
        'method': 'centroid'
    }


def fit_with_fallback(patch, sigma_init=None):
    """
    带降级策略的质心检测

    优先使用高斯拟合，失败时自动降级为加权质心法。

    Parameters
    ----------
    patch      : 2D ndarray  光斑图像patch
    sigma_init : float       高斯拟合初始sigma，None时从config读取

    Returns
    -------
    dict 包含 x0, y0, success, method（'gaussian' 或 'centroid'）
    """
    from config import SPOT_SIGMA_PX as DEFAULT_SIGMA
    if sigma_init is None:
        sigma_init = DEFAULT_SIGMA

    result = fit_gaussian(patch, sigma_init=sigma_init)
    if result['success']:
        result['method'] = 'gaussian'
        return result

    fallback = centroid_weighted(patch)
    if fallback['success']:
        print("  [降级] 高斯拟合失败，使用加权质心法")
    return fallback


def is_valid_centroid_result(x0, y0, patch_shape, amplitude=None, snr=None, sigma=None):
    H, W = patch_shape

    if not (np.isfinite(x0) and np.isfinite(y0)):
        return False

    # 质心必须落在patch内部
    if not (0.0 <= x0 < W and 0.0 <= y0 < H):
        return False

    # 振幅必须合理
    if amplitude is not None and amplitude <= 0:
        return False

    # SNR太低不认为是可靠拟合
    if snr is not None and snr < MIN_SNR * 0.5:
        return False

    # sigma如果给了，也做个宽松检查
    if sigma is not None and not (0.5 <= sigma <= 5.0 * SPOT_SIGMA_PX):
        return False

    return True


def is_valid_fit_result(result, patch_shape):
    H, W = patch_shape
    x0 = result.get("x0", np.nan)
    y0 = result.get("y0", np.nan)
    amp = result.get("amplitude", np.nan)
    snr = result.get("snr", np.nan)
    sigma_x = result.get("sigma_x", np.nan)
    sigma_y = result.get("sigma_y", np.nan)

    # 1) 坐标必须有效
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return False

    # 2) 质心必须落在 patch 内
    if not (0.0 <= x0 < W and 0.0 <= y0 < H):
        return False

    # 3) 振幅必须为正
    if not np.isfinite(amp) or amp <= 0:
        return False

    # 4) SNR 太低不通过
    if not np.isfinite(snr) or snr < MIN_SNR * 0.5:
        return False

    # 5) sigma 太离谱不通过
    if np.isfinite(sigma_x) and not (0.3 <= sigma_x <= 5.0 * SPOT_SIGMA_PX):
        return False
    if np.isfinite(sigma_y) and not (0.3 <= sigma_y <= 5.0 * SPOT_SIGMA_PX):
        return False

    return True
# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    from spot_generator import generate_gaussian_spot
    import numpy as np

    print(f"photutils 可用: {HAS_PHOTUTILS}")

    rng = np.random.default_rng(42)

    N = 200
    errors_x, errors_y = [], []
    snrs = []
    engine_counts = {}

    for _ in range(N):
        true_x = 24.0 + rng.uniform(-2, 2)
        true_y = 24.0 + rng.uniform(-2, 2)
        patch = generate_gaussian_spot(true_x, true_y, image_size=50, rng=rng)
        result = fit_gaussian(patch, sigma_init=SPOT_SIGMA_PX)

        if result['success']:
            errors_x.append(result['x0'] - true_x)
            errors_y.append(result['y0'] - true_y)
            snrs.append(result['snr'])
            engine = result.get('engine', 'scipy')
            engine_counts[engine] = engine_counts.get(engine, 0) + 1

    errors_x = np.array(errors_x)
    errors_y = np.array(errors_y)
    rms_x = np.std(errors_x)
    rms_y = np.std(errors_y)
    rms_r = np.sqrt(rms_x ** 2 + rms_y ** 2)

    from config import FOCAL_PLANE_SCALE_UM_PX, TARGET_ACCURACY_UM

    rms_um = rms_r * FOCAL_PLANE_SCALE_UM_PX

    print(f"=== 高斯拟合精度测试 (N={len(errors_x)}) ===")
    print(f"  X方向 RMS: {rms_x:.4f} px")
    print(f"  Y方向 RMS: {rms_y:.4f} px")
    print(f"  合成 RMS:  {rms_r:.4f} px = {rms_um:.2f} μm (焦面)")
    print(f"  平均 SNR:  {np.mean(snrs):.1f}")
    print(f"  引擎统计:  {engine_counts}")
    print(f"  目标精度:  {TARGET_ACCURACY_UM} μm → {'✓ 达标' if rms_um <= TARGET_ACCURACY_UM else '✗ 未达标'}")

