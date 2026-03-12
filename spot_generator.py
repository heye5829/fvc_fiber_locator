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
                           # 椭圆光斑参数（新增）
                           sigma_x=None,
                           sigma_y=None,
                           theta=None,
                           ellipticity_prob=None):
    """
    生成单个高斯光斑图像patch（支持椭圆）

    Parameters
    ----------
    center_x, center_y : float  光斑中心（像素，相对patch左上角）
    image_size         : int    patch边长（像素）
    sigma              : float  高斯sigma（像素，圆形光斑时使用）
    peak               : float  峰值计数
    background         : float  背景计数
    read_noise         : float  读出噪声 e-
    add_noise          : bool   是否添加噪声
    rng                : np.random.Generator  随机数生成器
    sigma_x            : float  X方向sigma（椭圆光斑，None时自动决定）
    sigma_y            : float  Y方向sigma（椭圆光斑，None时自动决定）
    theta              : float  椭圆旋转角（弧度，None时随机）
    ellipticity_prob   : float  生成椭圆光斑的概率（0-1，None时使用config）

    Returns
    -------
    image : np.ndarray (image_size, image_size)  模拟图像（float64）
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    if ellipticity_prob is None:
        ellipticity_prob = ELLIPTICAL_SPOT_PROB

    # 决定是否生成椭圆光斑
    is_elliptical = rng.random() < ellipticity_prob

    if sigma_x is None:
        sigma_x = sigma

    if sigma_y is None:
        if is_elliptical:
            # 椭圆率在配置范围内随机
            ellipticity = rng.uniform(*ELLIPTICITY_RANGE)
            sigma_y = sigma_x * ellipticity
        else:
            sigma_y = sigma_x  # 圆形

    if theta is None:
        if is_elliptical:
            theta = rng.uniform(0, np.pi)
        else:
            theta = 0.0  # 圆形时theta无意义

    y, x = np.mgrid[0:image_size, 0:image_size].astype(float)

    # 椭圆高斯公式
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
        # 泊松噪声（光子统计）
        noisy = rng.poisson(np.maximum(signal, 0)).astype(float)
        # 读出噪声（高斯）
        noisy += rng.normal(0, read_noise, signal.shape)
    else:
        noisy = signal.copy()

    return noisy


def generate_scene(fiber_positions_px, image_shape=(512, 512),
                   sigma=SPOT_SIGMA_PX, peak=SPOT_PEAK_COUNTS,
                   background=BACKGROUND_COUNTS, rng=None):
    """
    在完整图像上生成多个光斑

    Parameters
    ----------
    fiber_positions_px : list of (x, y)  光纤中心像素坐标
    image_shape        : (H, W)

    Returns
    -------
    image : np.ndarray (H, W)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    H, W = image_shape
    y, x = np.mgrid[0:H, 0:W].astype(float)
    image = np.full((H, W), float(background))

    for cx, cy in fiber_positions_px:
        exponent = -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)
        image += peak * np.exp(exponent)

    # 泊松 + 读出噪声
    image = rng.poisson(np.maximum(image, 0)).astype(float)
    image += rng.normal(0, READ_NOISE_E, image.shape)

    return image


def extract_patch(image, cx, cy, half_size):
    """
    从图像中提取以(cx,cy)为中心的patch（整数坐标）

    Returns
    -------
    patch     : np.ndarray
    offset_x  : float  patch左上角在原图中的x坐标
    offset_y  : float  patch左上角在原图中的y坐标
    """
    H, W = image.shape
    x0 = int(round(cx)) - half_size
    y0 = int(round(cy)) - half_size
    x1 = x0 + 2 * half_size
    y1 = y0 + 2 * half_size

    # 边界裁剪
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)

    patch = image[y0c:y1c, x0c:x1c]
    return patch, float(x0c), float(y0c)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rng = np.random.default_rng(RANDOM_SEED)
    # 测试：生成单个光斑
    img = generate_gaussian_spot(24.3, 25.7, image_size=50, rng=rng)
    print(f"光斑图像: shape={img.shape}, max={img.max():.1f}, min={img.min():.1f}")
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='hot', origin='lower')
    plt.colorbar(label='Counts')
    plt.title(
        f'模拟光斑 (sigma={SPOT_SIGMA_PX}px, SNR={SPOT_PEAK_COUNTS / np.sqrt(SPOT_PEAK_COUNTS + BACKGROUND_COUNTS):.1f})')
    plt.tight_layout()
    plt.savefig('test_spot.png', dpi=150)
    print("已保存 test_spot.png")