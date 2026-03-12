"""
MUST望远镜 FVC - 全图光斑粗检测模块

功能：从完整FVC图像中找到所有光斑的大致像素位置
     作为 gaussian_detector.py 精确拟合的前置步骤

使用场景：
    仿真模式：不需要此模块（光纤位置已知）
    真实图像模式：必须先调用此模块找到所有光斑位置

三种检测方法：
    threshold : 自适应阈值 + 连通域分析（推荐，速度快）
    log       : 高斯拉普拉斯斑点检测（适合sigma变化大的情况）
    opencv    : OpenCV SimpleBlobDetector（安装opencv后可用）
"""

import numpy as np
from scipy import ndimage
from config import SPOT_SIGMA_PX

# OpenCV可选导入
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ============================================================
# 图像读取
# ============================================================

def load_fvc_image(image_path):
    """
    读取FVC图像，支持FITS、PNG、TIFF、BMP等格式

    Parameters
    ----------
    image_path : str  图像文件路径

    Returns
    -------
    image : 2D ndarray  灰度图像（float64）
    """
    path = str(image_path).lower()

    if path.endswith('.fits') or path.endswith('.fit'):
        # 天文FITS格式
        try:
            import astropy.io.fits as fits
            data = fits.getdata(image_path)
            print(f"  [读取] FITS图像: {data.shape}, dtype={data.dtype}")
            return data.astype(np.float64)
        except ImportError:
            raise ImportError("读取FITS需要安装astropy: pip install astropy")

    elif HAS_OPENCV:
        # 用OpenCV读取普通图像（支持16位）
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        # 彩色图转灰度
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"  [读取] 图像: {img.shape}, dtype={img.dtype}")
        return img.astype(np.float64)

    else:
        # OpenCV未安装，用PIL读取
        try:
            from PIL import Image
            img = np.array(Image.open(image_path).convert('L'), dtype=np.float64)
            print(f"  [读取] 图像(PIL): {img.shape}")
            return img
        except ImportError:
            raise ImportError("读取图像需要安装opencv或Pillow: pip install opencv-python")


# ============================================================
# 图像预处理
# ============================================================

def preprocess_image(image, apply_denoise=True, apply_bg_subtract=True):
    """
    FVC图像预处理流水线

    步骤：
        1. 高斯去噪（可选）—— 抑制读出噪声，使用3×3小核避免模糊光斑
        2. 背景减除（可选）—— 消除不均匀背景

    Parameters
    ----------
    image             : 2D ndarray  原始图像
    apply_denoise     : bool        是否高斯去噪
    apply_bg_subtract : bool        是否背景减除

    Returns
    -------
    processed : 2D ndarray  预处理后图像（float64）
    """
    img = image.astype(np.float64)

    # 步骤1：高斯去噪
    # 使用3×3小核，平滑强度低，避免模糊光斑影响后续定位精度
    if apply_denoise:
        if HAS_OPENCV:
            img_uint16 = np.clip(img, 0, 65535).astype(np.uint16)
            img_denoised = cv2.GaussianBlur(img_uint16, (3, 3), sigmaX=1)
            img = img_denoised.astype(np.float64)
        else:
            # OpenCV未安装，用scipy做高斯滤波
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma=0.5)

    # 步骤2：背景减除
    if apply_bg_subtract:
        bg = estimate_background(img)
        img = np.maximum(img - bg, 0.0)  # 不允许负值

    return img


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
    y_centers, x_centers, medians = [], [], []
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            block = image[y:min(y + box_size, h),
                          x:min(x + box_size, w)]
            y_centers.append(y + block.shape[0] / 2)
            x_centers.append(x + block.shape[1] / 2)
            medians.append(float(np.median(block)))

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

    # 步骤2：估计噪声
    negative_pixels = img_sub[img_sub < 0]
    if len(negative_pixels) > 100:
        noise_std = float(np.std(negative_pixels))
    else:
        noise_std = float(np.std(img_sub))
    noise_std = max(noise_std, 1.0)

    # 步骤3：阈值分割
    threshold = sigma_factor * noise_std
    binary = (img_sub > threshold).astype(np.uint8)

    # 步骤4：连通域标记
    struct = ndimage.generate_binary_structure(2, 2)  # 8连通
    labeled, n_features = ndimage.label(binary, structure=struct)

    if n_features == 0:
        return np.zeros((0, 2))

    # 步骤5：提取每个连通域的加权质心
    positions = []
    for label_id in range(1, n_features + 1):
        region_mask = (labeled == label_id)
        area = int(region_mask.sum())

        if area < min_area or area > max_area:
            continue

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

    return blobs[:, [1, 0]].copy()


# ============================================================
# 方法C：OpenCV SimpleBlobDetector
# ============================================================

def detect_spots_opencv(image, min_area=5, max_area=500,
                        min_circularity=0.5):
    """
    用OpenCV SimpleBlobDetector检测光斑

    比手写连通域分析更健壮，支持按面积、圆度过滤。
    需要安装opencv-python: pip install opencv-python

    Parameters
    ----------
    image           : 2D ndarray  预处理后的图像
    min_area        : float       最小光斑面积（像素数）
    max_area        : float       最大光斑面积（像素数）
    min_circularity : float       最小圆度（0~1，1为完美圆形）

    Returns
    -------
    positions : (N, 2) ndarray  [x, y] 粗定位坐标
    """
    if not HAS_OPENCV:
        print("  [警告] OpenCV未安装，改用阈值法。安装命令: pip install opencv-python")
        return detect_spots_threshold(image)

    # SimpleBlobDetector需要8位图像，归一化到0-255
    img_norm = cv2.normalize(image, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    # 反转：SimpleBlobDetector默认检测暗斑，反转后检测亮斑
    img_inv = 255 - img_norm

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_inv)

    if not keypoints:
        return np.zeros((0, 2))

    positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    return positions


# ============================================================
# 统一接口（更新版，支持路径输入、预处理、opencv）
# ============================================================

def detect_all_spots(image, method='threshold',
                     preprocess=False, verbose=True, **kwargs):
    """
    统一光斑粗检测接口

    Parameters
    ----------
    image      : str 或 2D ndarray
                 可以传文件路径字符串，也可以传图像数组
    method     : str  'threshold'（推荐）/ 'log' / 'opencv'
    preprocess : bool 是否做预处理（去噪+背景减除）
                 仿真模式下通常不需要（False）
                 真实图像模式下建议开启（True）
    verbose    : bool 是否打印检测结果统计
    **kwargs         传递给具体检测函数的额外参数

    Returns
    -------
    positions : (N, 2) ndarray  粗定位坐标 [x, y]，精度约1-2px
    """
    # 支持直接传路径字符串
    if isinstance(image, str):
        image = load_fvc_image(image)

    # 预处理（真实图像时建议开启）
    if preprocess:
        image = preprocess_image(image)

    if method == 'threshold':
        positions = detect_spots_threshold(image, **kwargs)
    elif method == 'log':
        positions = detect_spots_log(image, **kwargs)
    elif method == 'opencv':
        positions = detect_spots_opencv(image, **kwargs)
    else:
        raise ValueError(f"未知检测方法: {method}，可选: 'threshold' / 'log' / 'opencv'")

    if verbose:
        print(f"  [粗检测] 方法={method}，找到 {len(positions)} 个光斑候选")

    return positions

