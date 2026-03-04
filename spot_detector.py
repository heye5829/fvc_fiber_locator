"""
MUST望远镜 FVC - 全图光斑粗检测模块

功能：从完整FVC图像中找到所有光斑的大致像素位置
     作为 gaussian_detector.py 精确拟合的前置步骤

使用场景：
    仿真模式：不需要此模块（光纤位置已知）
    真实图像模式：必须先调用此模块找到所有光斑位置

两种检测方法：
    threshold : 自适应阈值 + 连通域分析（推荐，速度快）
    log       : 高斯拉普拉斯斑点检测（适合sigma变化大的情况）
"""

import numpy as np
from scipy import ndimage
from config import SPOT_SIGMA_PX


# ============================================================
# 背景估计
# ============================================================

def estimate_background(image, box_size=64):
    """
    分块中位数背景估计，处理空间不均匀背景

    原理：将图像分成 box_size×box_size 的小块，
         每块取中位数作为背景值，再双线性插值得到全图背景。
         中位数对光斑亮点不敏感，适合星场/光纤场景。

    Parameters
    ----------
    image    : 2D ndarray  输入图像
    box_size : int         分块大小（像素）

    Returns
    -------
    bg : 2D ndarray  与输入同形状的背景估计图
    """
    h, w = image.shape
    # 计算每个块的中位数
    y_centers, x_centers, medians = [], [], []
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            block = image[y:min(y + box_size, h),
                          x:min(x + box_size, w)]
            y_centers.append(y + block.shape[0] / 2)
            x_centers.append(x + block.shape[1] / 2)
            medians.append(float(np.median(block)))

    # 双线性插值回全图尺寸
    from scipy.interpolate import griddata
    yi, xi = np.mgrid[0:h, 0:w]
    points = np.column_stack([y_centers, x_centers])
    bg = griddata(points, medians, (yi, xi),
                  method='linear', fill_value=np.median(medians))
    return bg.astype(float)


# ============================================================
# 方法A：自适应阈值 + 连通域分析
# ============================================================

def detect_spots_threshold(image, sigma_factor=5.0,
                           min_area=3, max_area=500):
    """
    自适应阈值法检测光斑（推荐方法）

    流程：
        1. 估计并减除背景
        2. 计算背景噪声标准差
        3. 阈值 = sigma_factor × noise_std
        4. 二值化 → 连通域标记 → 提取质心

    Parameters
    ----------
    image        : 2D ndarray  输入图像（灰度，float或uint16）
    sigma_factor : float       阈值倍数（越大越严格，漏检越多）
    min_area     : int         最小连通域面积（像素数），过滤噪声点
    max_area     : int         最大连通域面积（像素数），过滤合并光斑

    Returns
    -------
    positions : (N, 2) ndarray  每行为 [x, y] 像素坐标（粗定位，精度~1-2px）
    """
    img = image.astype(float)

    # 步骤1：背景减除
    bg = estimate_background(img)
    img_sub = img - bg

    # 步骤2：估计噪声（用背景减除后的负值区域，代表纯噪声）
    negative_pixels = img_sub[img_sub < 0]
    if len(negative_pixels) > 100:
        # 负值的标准差近似等于噪声标准差
        noise_std = float(np.std(negative_pixels))
    else:
        # 备用：用全图标准差
        noise_std = float(np.std(img_sub))
    noise_std = max(noise_std, 1.0)  # 防止除零

    # 步骤3：阈值分割
    threshold = sigma_factor * noise_std
    binary = (img_sub > threshold).astype(np.uint8)

    # 步骤4：连通域标记
    struct = ndimage.generate_binary_structure(2, 2)  # 8连通
    labeled, n_features = ndimage.label(binary, structure=struct)

    if n_features == 0:
        return np.zeros((0, 2))

    # 步骤5：提取每个连通域的质心（加权）
    positions = []
    for label_id in range(1, n_features + 1):
        region_mask = (labeled == label_id)
        area = int(region_mask.sum())

        # 面积过滤
        if area < min_area or area > max_area:
            continue

        # 用原始图像强度加权质心
        region_intensity = img_sub * region_mask
        total = float(region_intensity.sum())
        if total <= 0:
            continue

        yi, xi = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        cx = float((region_intensity * xi).sum() / total)
        cy = float((region_intensity * yi).sum() / total)
        positions.append([cx, cy])

    return np.array(positions) if positions else np.zeros((0, 2))


# ============================================================
# 方法B：LoG斑点检测
# ============================================================

def detect_spots_log(image, min_sigma=1.0, max_sigma=8.0,
                     num_sigma=10, threshold=0.02):
    """
    高斯拉普拉斯（LoG）斑点检测

    适用场景：光斑sigma在不同位置变化较大时（如边缘畸变导致光斑变形）
    缺点：比阈值法慢约5倍

    Parameters
    ----------
    image     : 2D ndarray  输入图像
    min_sigma : float       最小光斑sigma（像素）
    max_sigma : float       最大光斑sigma（像素）
    num_sigma : int         sigma搜索步数
    threshold : float       响应阈值（0~1归一化后）

    Returns
    -------
    positions : (N, 2) ndarray  每行为 [x, y] 像素坐标
    """
    try:
        from skimage.feature import blob_log
    except ImportError:
        print("  [警告] scikit-image未安装，LoG检测不可用，改用阈值法")
        return detect_spots_threshold(image)

    # 归一化到 [0, 1]
    img_min, img_max = image.min(), image.max()
    if img_max <= img_min:
        return np.zeros((0, 2))
    img_norm = (image.astype(float) - img_min) / (img_max - img_min)

    blobs = blob_log(img_norm,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     num_sigma=num_sigma,
                     threshold=threshold)

    if len(blobs) == 0:
        return np.zeros((0, 2))

    # blobs 每行：[row, col, sigma] → 转为 [x, y]
    return blobs[:, [1, 0]].copy()


# ============================================================
# 统一接口
# ============================================================

def detect_all_spots(image, method='threshold', verbose=True, **kwargs):
    """
    统一光斑粗检测接口

    Parameters
    ----------
    image   : 2D ndarray  完整FVC图像
    method  : str         'threshold'（推荐）或 'log'
    verbose : bool        是否打印检测结果统计
    **kwargs              传递给具体检测函数的额外参数

    Returns
    -------
    positions : (N, 2) ndarray  粗定位坐标 [x, y]，精度约1-2px
    """
    if method == 'threshold':
        positions = detect_spots_threshold(image, **kwargs)
    elif method == 'log':
        positions = detect_spots_log(image, **kwargs)
    else:
        raise ValueError(f"未知检测方法: {method}，可选: 'threshold', 'log'")

    if verbose:
        print(f"  [粗检测] 方法={method}，找到 {len(positions)} 个光斑候选")

    return positions
