"""
MUST望远镜 FVC 光纤位置测量系统 - 参数配置
目标精度：3μm
"""

# ============ 相机传感器参数 ============
PIXEL_SIZE_UM = 3.76          # 像素尺寸 μm (IMX411: 3.76μm)
SENSOR_WIDTH_PX = 14208       # 传感器宽度（像素）
SENSOR_HEIGHT_PX = 10656      # 传感器高度（像素）

# ============ 光学系统参数 ============
FOCAL_LENGTH_MM = 230.0       # FVC镜头焦距 mm
DEMAGNIFICATION = 37.0        # 缩放比（焦面→像面）
# 焦面每像素物理尺寸 = PIXEL_SIZE_UM * DEMAGNIFICATION
FOCAL_PLANE_SCALE_UM_PX = PIXEL_SIZE_UM * DEMAGNIFICATION  # 3.76×37 = 139.12 μm/px

# ============ 光纤参数 ============
FIBER_CORE_DIAMETER_UM = 140.0   # 光纤芯径 μm
NUM_REFERENCE_FIBERS = 9         # 基准光纤数量（3x3格网）
NUM_TARGET_FIBERS = 100          # 待测光纤数量（仿真用）

# ============ 仿真光斑参数 ============
SPOT_SIGMA_PX = 2              # 光斑高斯sigma（像素），含光学扩散         仿真用2px（实际光斑~0.53px，放大以保证采样质量）
SPOT_PEAK_COUNTS = 5000          # 光斑峰值计数
BACKGROUND_COUNTS = 50           # 背景计数
READ_NOISE_E = 1.5               # 读出噪声 e-
DARK_CURRENT_E_S = 0.1           # 暗电流 e-/s
EXPOSURE_TIME_S = 0.1            # 曝光时间 s

# ============ 精度目标 ============
TARGET_ACCURACY_UM = 3.0         # 目标测量精度 μm (RMS)
TARGET_ACCURACY_PX = TARGET_ACCURACY_UM / FOCAL_PLANE_SCALE_UM_PX  ## 对应像素精度自动计算：3.0 / 139.12 = 0.0216 px（约1/46像素）

# ============ 高斯拟合参数 ============
FIT_WINDOW_SIGMA = 4             # 拟合窗口半径 = FIT_WINDOW_SIGMA * SPOT_SIGMA_PX
MIN_SNR = 20                     # 最低SNR要求
MAX_FIT_ITERATIONS = 1000        # 最大迭代次数

# ============ 标定参数 ============
# 基准光纤焦面坐标范围 mm（示例：焦面板300x300mm区域中心9点）
REFERENCE_GRID_SPACING_MM = 50.0   # 基准光纤间距 mm
REFERENCE_GRID_ORIGIN_MM = (-100.0, -100.0)  # 基准网格原点

# 畸变模型阶数
DISTORTION_POLY_DEGREE = 1       # 多项式畸变模型阶数          改为1阶！9个基准点用3阶会严重过拟

# ============ 仿真随机种子 ============
RANDOM_SEED =42 #None

# ============ 输出路径 ============
OUTPUT_DIR = "outputs"
RESULTS_FILE = "results/accuracy_report.json"

print(f"[Config] 焦面尺度: {FOCAL_PLANE_SCALE_UM_PX:.2f} μm/px")
print(f"[Config] 目标精度: {TARGET_ACCURACY_UM} μm = {TARGET_ACCURACY_PX:.4f} px")

