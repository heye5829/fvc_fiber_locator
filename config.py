"""
MUST望远镜 FVC 光纤位置测量系统 - 参数配置
目标精度：3μm
"""

# ============ 相机传感器参数 ============
PIXEL_SIZE_UM = 3.76          # 像素尺寸 μm (IMX411: 3.76μm)
SENSOR_WIDTH_PX = 14208       # 传感器宽度（像素）
SENSOR_HEIGHT_PX = 10656      # 传感器高度（像素）
IMAGE_WIDTH = SENSOR_WIDTH_PX   # 图像宽度（用于畸变计算）
IMAGE_HEIGHT = SENSOR_HEIGHT_PX # 图像高度（用于畸变计算）

# ============ 光学系统参数 ============
FOCAL_LENGTH_MM = 230.0       # FVC镜头焦距 mm
DEMAGNIFICATION = 37.0        # 缩放比（焦面→像面）
# 焦面每像素物理尺寸 = PIXEL_SIZE_UM * DEMAGNIFICATION
FOCAL_PLANE_SCALE_UM_PX = PIXEL_SIZE_UM * DEMAGNIFICATION  # 3.76×37 = 139.12 μm/px

# ============ 标定参数 ============
# 基准光纤焦面坐标范围 mm
REFERENCE_GRID_SPACING_MM = 50.0   # 基准光纤间距 mm
# 基准格网原点：让7×7格网中心落在焦面坐标原点
# 7×7格网，间距50mm，总跨度300mm，中心在150mm处
# 原点设为 -150mm，使格网中心在(0, 0)
REFERENCE_GRID_ORIGIN_MM = (-150.0, -150.0)
REF_GRID_SIDE = 7   # 基准格网边长，共 7×7=49 个基准点

# 畸变模型阶数（修改为4阶，适应大视场）
POLY_ORDER = 4

# ============ 光纤参数 ============
FIBER_CORE_DIAMETER_UM = 140.0          # 光纤芯径 μm
NUM_REFERENCE_FIBERS = REF_GRID_SIDE ** 2  # 自动计算，当前49
NUM_TARGET_FIBERS = 500                 # 待测光纤数量（仿真用）

# ============ 仿真光斑参数 ============
SPOT_SIGMA_PX = 2              # 光斑高斯sigma（像素），含光学扩散           仿真用2px（实际光斑~0.53px，放大以保证采样质量）
SPOT_PEAK_COUNTS = 5000        # 光斑峰值计数
BACKGROUND_COUNTS = 50         # 背景计数
READ_NOISE_E = 10.0            # 工业CMOS典型值 读出噪声 e-
DARK_CURRENT_E_S = 0.1         # 暗电流 e-/s
EXPOSURE_TIME_S = 0.1          # 曝光时间 s

# 椭圆光斑仿真参数（新增）
ELLIPTICAL_SPOT_PROB = 0.2      # 椭圆光斑比例（0-1），0=全部圆形，1=全部椭圆     0.1
ELLIPTICITY_RANGE = (1.1, 1.4)  # 椭圆率范围（长短轴比 sigma_y/sigma_x）      1.05, 1.2

# ============ 精度目标 ============
TARGET_ACCURACY_UM = 3.0       # 目标测量精度 μm (RMS)
TARGET_ACCURACY_PX = TARGET_ACCURACY_UM / FOCAL_PLANE_SCALE_UM_PX  # 对应像素精度自动计算：3.0 / 139.12 = 0.0216 px（约1/46像素）

# ============ 高斯拟合参数 ============
FIT_WINDOW_SIGMA = 4           # 拟合窗口半径 = FIT_WINDOW_SIGMA * SPOT_SIGMA_PX
MIN_SNR = 20                   # 最低SNR要求
MAX_FIT_ITERATIONS = 1000      # 最大迭代次数

# ============ 径向畸变仿真参数 ============
# 针对超大视场系统（焦面直径1250mm）
DISTORTION_K1 = -0.005         # 径向畸变一阶系数（负值=桶形畸变）
DISTORTION_K2 = 0.001          # 径向畸变二阶系数
FOCAL_RADIUS_UM = 625000.0     # 焦面半径 = 1250mm / 2 = 625mm = 625000μm

# ============ 仿真随机种子 ============
RANDOM_SEED = 42               # None 表示随机

# ============ 输出路径 ============
OUTPUT_DIR = "outputs"
RESULTS_FILE = "results/accuracy_report.json"

print(f"[Config] 焦面尺度: {FOCAL_PLANE_SCALE_UM_PX:.2f} μm/px")
print(f"[Config] 目标精度: {TARGET_ACCURACY_UM} μm = {TARGET_ACCURACY_PX:.4f} px")
print(f"[Config] 多项式阶数: {POLY_ORDER}")


# ============ 数据集生成协议 ============
# 仅供 dataset_generator.py 使用，不影响现有任何文件

DATASET_CONFIG = {

    # 场点采样（中等视场 0~0.8R，第四阶段扩全场时加入 1.0）
    "sample_radii_norm": [0.0, 0.3, 0.6, 0.8],
    "azimuths_per_radius": 4,       # r=0 时自动只取1个点   8 → 4
    "field_radius_mm": 625.0,       # 焦面半径，对应1250mm直径

    # SNR 三档，用 peak 值控制
    # SNR ≈ peak / sqrt(peak + background)
    "snr_levels": {
        "high": 5000,   # SNR ≈ 70，与现有 SPOT_PEAK_COUNTS 一致
        "mid":  1500,   # SNR ≈ 38
        "low":   500,   # SNR ≈ 22
    },

    # 离焦三档，在基础 sigma 上叠加额外展宽
    # 安装公差 ±100μm / 139.12μm/px ≈ 0.72px，取 0.3/0.6 作为轻微/中等
    "defocus_levels": {
        "none":   0.0,
        "medium": 0.6,          # 删掉 slight
    },

    # 基准光纤数量（同一张图里已知位置的光纤数，用于标定资源敏感性实验）
    "n_calib_fibers_list": [16, 32, 64],

    # 基准光纤分布方式
    "calib_distributions": ["uniform", "random"],     # 删掉 ring

    # 前向/反演模型脱钩：生成器加真实扰动，求解器仍用理想高斯
    "fiber_brightness_variation": 0.15,    # 亮度随机扰动 ±15%
    "background_gradient": True,            # 背景带轻微梯度
    "background_gradient_strength": 10.0,  # 梯度强度（counts，跨全图）
    "pixel_response_nonuniformity": 0.02,  # 像元响应不均匀度 2%

    # 场依赖 PSF（边缘 sigma 比中心大，模拟大视场成像质量下降）
    "psf_sigma_center_px": 2.0,   # 与现有 SPOT_SIGMA_PX 一致
    "psf_sigma_edge_px":   2.6,   # 边缘展宽
    "ellipticity_edge":    1.20,  # 边缘椭圆率上限        # 0.20 → 1.20

    # 每种条件重复次数（不同随机子像素偏移，保证统计稳定性）
    "n_repeat": 2,              # 3 → 2

    # 输出路径（代码自动创建，无需手动建文件夹）
    "save_dir": "dataset",

    # 固定随机种子（保证可复现）
    "seed": 42,
}