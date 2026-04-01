"""
仿真光斑生成器
生成带噪声的高斯光斑，模拟FVC实际观测图像
支持圆形和椭圆光斑
"""

import numpy as np
from config import *


def generate_gaussian_spot(center_x, center_y, image_size,
                           sigma=SPOT_SIGMA_PX,
                           peak=SPOT_PEAK_COUNTS,
                           background=BACKGROUND_COUNTS,
                           read_noise=READ_NOISE_E,
                           add_noise=True,
                           rng=None,
                           sigma_x=None,
                           sigma_y=None,
                           theta=None,
                           ellipticity_prob=None):
    """生成单个高斯光斑图像patch（支持椭圆）"""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    if ellipticity_prob is None:
        ellipticity_prob = ELLIPTICAL_SPOT_PROB

    is_elliptical = rng.random() < ellipticity_prob

    if sigma_x is None:
        sigma_x = sigma

    if sigma_y is None:
        if is_elliptical:
            ellipticity = rng.uniform(*ELLIPTICITY_RANGE)
            sigma_y = sigma_x * ellipticity
        else:
            sigma_y = sigma_x

    if theta is None:
        theta = rng.uniform(0, np.pi) if is_elliptical else 0.0

    y, x = np.mgrid[0:image_size, 0:image_size].astype(float)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = (cos_t ** 2) / (2 * sigma_x ** 2) + (sin_t ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (sin_t ** 2) / (2 * sigma_x ** 2) + (cos_t ** 2) / (2 * sigma_y ** 2)

    dx = x - center_x
    dy = y - center_y
    exponent = -(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2)
    signal = peak * np.exp(exponent) + background

    if add_noise:
        noisy = rng.poisson(np.maximum(signal, 0)).astype(float)
        noisy += rng.normal(0, read_noise, signal.shape)
    else:
        noisy = signal.copy()

    return noisy


def generate_scene(fiber_positions_px, image_shape=(512, 512),
                   sigma=SPOT_SIGMA_PX, peak=SPOT_PEAK_COUNTS,
                   background=BACKGROUND_COUNTS, rng=None,
                   # 椭圆光斑参数（新增）
                   sigma_x=None, sigma_y=None,
                   ellipticity_prob=None,
                   theta=None):          # 新增 theta 参数
    """
    在完整图像上生成多个光斑（支持椭圆高斯）

    Parameters
    ----------
    fiber_positions_px : list of (x, y)
    image_shape        : (H, W)
    sigma              : float  圆形光斑 sigma（椭圆时作为 sigma_x 基准）
    sigma_x            : float  椭圆长轴 sigma，None 时用 sigma
    sigma_y            : float  椭圆短轴 sigma，None 时按 ellipticity_prob 随机决定
    ellipticity_prob   : float  椭圆光斑概率，None 时用 config 里的值
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    if ellipticity_prob is None:
        ellipticity_prob = ELLIPTICAL_SPOT_PROB

    H, W = image_shape
    y, x = np.mgrid[0:H, 0:W].astype(float)
    image = np.full((H, W), float(background))

    _sigma_x = sigma_x if sigma_x is not None else sigma

    for cx, cy in fiber_positions_px:
        # 每根光纤独立决定是否椭圆及旋转角
        is_elliptical = rng.random() < ellipticity_prob

        if sigma_y is not None:
            _sigma_y = sigma_y
            # theta 优先用传入值（场依赖径向方向），否则按随机椭圆逻辑
            _theta = theta if theta is not None else (
                rng.uniform(0, np.pi) if is_elliptical else 0.0
            )
        elif is_elliptical:
            ellipticity = rng.uniform(*ELLIPTICITY_RANGE)
            _sigma_y    = _sigma_x * ellipticity
            _theta      = rng.uniform(0, np.pi)
        else:
            _sigma_y = _sigma_x
            _theta   = 0.0

        dx = x - cx
        dy = y - cy

        if _theta == 0.0 and _sigma_x == _sigma_y:
            # 圆形：用简化公式，速度更快
            exponent = -(dx ** 2 + dy ** 2) / (2 * _sigma_x ** 2)
        else:
            # 椭圆高斯
            cos_t = np.cos(_theta)
            sin_t = np.sin(_theta)
            a = cos_t**2 / (2*_sigma_x**2) + sin_t**2 / (2*_sigma_y**2)
            b = (-np.sin(2*_theta) / (4*_sigma_x**2)
                 + np.sin(2*_theta) / (4*_sigma_y**2))
            c = sin_t**2 / (2*_sigma_x**2) + cos_t**2 / (2*_sigma_y**2)
            exponent = -(a*dx**2 + 2*b*dx*dy + c*dy**2)

        image += peak * np.exp(exponent)

    # 泊松 + 读出噪声
    image = rng.poisson(np.maximum(image, 0)).astype(float)
    image += rng.normal(0, READ_NOISE_E, image.shape)

    return image


def extract_patch(image, cx, cy, half_size):
    """从图像中提取以(cx,cy)为中心的patch"""
    H, W = image.shape
    x0 = int(round(cx)) - half_size
    y0 = int(round(cy)) - half_size
    x1 = x0 + 2 * half_size
    y1 = y0 + 2 * half_size

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)

    patch = image[y0c:y1c, x0c:x1c]
    return patch, float(x0c), float(y0c)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rng = np.random.default_rng(RANDOM_SEED)
    img = generate_gaussian_spot(24.3, 25.7, image_size=50, rng=rng)
    print(f"光斑图像: shape={img.shape}, max={img.max():.1f}, min={img.min():.1f}")
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='hot', origin='lower')
    plt.colorbar(label='Counts')
    plt.title(
        f'模拟光斑 (sigma={SPOT_SIGMA_PX}px, '
        f'SNR={SPOT_PEAK_COUNTS/np.sqrt(SPOT_PEAK_COUNTS+BACKGROUND_COUNTS):.1f})')
    plt.tight_layout()
    plt.savefig('test_spot.png', dpi=150)
    print("已保存 test_spot.png")
