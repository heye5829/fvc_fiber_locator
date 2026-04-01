"""
dataset_generator.py — 标准测试数据集生成器
位置：fvc_fiber_locator/ 根目录

图像尺寸：1024×1024 px，覆盖焦面约 142mm × 142mm
可验证：centroid精度、SNR/离焦鲁棒性、坐标变换精度、畸变校正

运行方式：
    python dataset_generator.py
"""

import os
import json
import numpy as np

from config import (
    DATASET_CONFIG,
    FOCAL_PLANE_SCALE_UM_PX,
    BACKGROUND_COUNTS,
    READ_NOISE_E,
    DISTORTION_K1,
    DISTORTION_K2,
    #FOCAL_RADIUS_UM,
    ELLIPTICAL_SPOT_PROB,
    ELLIPTICITY_RANGE,
)
from spot_generator import generate_scene


# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

IMG_H, IMG_W = 1024, 1024   # 覆盖焦面约 142mm × 142mm
N_FIBERS     = 400           # 每张图光纤数
MARGIN_PX    = 20            # 边界留白
IMG_RADIUS_PX = np.sqrt((IMG_W / 2)**2 + (IMG_H / 2)**2)  # ≈ 724px，畸变归一化半径

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _field_points(cfg):
    """生成所有场点物理坐标，返回 list of (x_mm, y_mm, r_norm)"""
    R = cfg["field_radius_mm"]
    pts = []
    for r in cfg["sample_radii_norm"]:
        if r == 0.0:
            pts.append((0.0, 0.0, 0.0))
        else:
            for i in range(cfg["azimuths_per_radius"]):
                a = 2 * np.pi * i / cfg["azimuths_per_radius"]
                pts.append((r * R * np.cos(a), r * R * np.sin(a), r))
    return pts


def _sigma_at(x_mm, y_mm, cfg):
    """场依赖 PSF sigma：中心到边缘线性插值"""
    r = np.sqrt(x_mm**2 + y_mm**2) / cfg["field_radius_mm"]
    return (cfg["psf_sigma_center_px"]
            + (cfg["psf_sigma_edge_px"] - cfg["psf_sigma_center_px"]) * r)


def _ellipticity_at(x_mm, y_mm, cfg):
    """
    场依赖椭圆率：中心=1.0（圆形），边缘=ellipticity_edge。
    返回 sigma_y / sigma_x，即长短轴比。
    """
    r = np.sqrt(x_mm**2 + y_mm**2) / cfg["field_radius_mm"]
    return 1.0 + (cfg["ellipticity_edge"] - 1.0) * r


def _apply_distortion(fx_px, fy_px, cx, cy):
    """
    对图像坐标施加径向畸变。
    归一化半径用图像对角线半径（而非焦面半径），
    确保图像边缘畸变量约 2px，对 0.02px 精度目标有足够区分度。
    """
    dx_px  = fx_px - cx
    dy_px  = fy_px - cy
    r_px   = np.sqrt(dx_px**2 + dy_px**2)
    r_norm = r_px / IMG_RADIUS_PX
    factor = 1.0 + DISTORTION_K1 * r_norm**2 + DISTORTION_K2 * r_norm**4
    return cx + dx_px * factor, cy + dy_px * factor


def _calib_idx(n_total, n_calib, dist, rng):
    """按分布方式选取基准光纤索引"""
    n = min(n_calib, n_total)
    if dist == "uniform":
        step = n_total / n
        return [int(i * step) for i in range(n)]
    elif dist == "ring":
        return np.linspace(0, n_total - 1, n, dtype=int).tolist()
    else:  # random
        return rng.choice(n_total, size=n, replace=False).tolist()


def _bg_map(shape, base, use_gradient, strength, rng):
    """生成背景图（均匀或带梯度）"""
    if not use_gradient:
        return np.full(shape, float(base))
    H, W = shape
    angle = rng.uniform(0, 2 * np.pi)
    yg, xg = np.mgrid[0:H, 0:W]
    grad = np.cos(angle) * xg / W + np.sin(angle) * yg / H
    return base + grad * strength


# ─────────────────────────────────────────────
# 单样本生成
# ─────────────────────────────────────────────

def _generate_one(field_x, field_y, r_norm,
                  snr_key, defocus_key,
                  n_calib, calib_dist,
                  cfg, rng):
    """
    生成一个样本：1024×1024 仿真图像 + 真值标注。

    坐标流程：
      1. 在图像坐标系内均匀随机生成光纤像素坐标（无畸变位置）
      2. 像素坐标 → 焦面物理坐标 mm（真值，存入标注）
      3. 对像素坐标施加径向畸变 → 畸变后像素坐标（用于渲染图像）
      4. baseline 需要从畸变后的图像坐标反解出物理坐标，
         这正是坐标变换方法要解决的问题
    """
    cx0 = IMG_W / 2.0
    cy0 = IMG_H / 2.0
    scale       = FOCAL_PLANE_SCALE_UM_PX
    peak        = cfg["snr_levels"][snr_key]
    defocus_add = cfg["defocus_levels"][defocus_key]

    # ── 1. 生成无畸变光纤像素坐标（均匀随机）──
    fx_px_ideal = rng.uniform(MARGIN_PX, IMG_W - MARGIN_PX, N_FIBERS)
    fy_px_ideal = rng.uniform(MARGIN_PX, IMG_H - MARGIN_PX, N_FIBERS)

    # ── 2. 无畸变像素坐标 → 焦面物理坐标 mm（真值）──
    fx_mm = field_x + (fx_px_ideal - cx0) * scale / 1000.0
    fy_mm = field_y + (fy_px_ideal - cy0) * scale / 1000.0

    # ── 3. 施加径向畸变 → 畸变后像素坐标（用于渲染）──
    fx_px_dist, fy_px_dist = _apply_distortion(fx_px_ideal, fy_px_ideal, cx0, cy0)

    # 过滤畸变后超出边界的光纤
    valid = ((fx_px_dist > MARGIN_PX) & (fx_px_dist < IMG_W - MARGIN_PX) &
             (fy_px_dist > MARGIN_PX) & (fy_px_dist < IMG_H - MARGIN_PX))
    fx_px_dist = fx_px_dist[valid]
    fy_px_dist = fy_px_dist[valid]
    fx_px_ideal = fx_px_ideal[valid]
    fy_px_ideal = fy_px_ideal[valid]
    fx_mm = fx_mm[valid]
    fy_mm = fy_mm[valid]
    n_valid = len(fx_px_dist)
    if n_valid == 0:
        return None, None

    # ── 4. 场依赖 PSF 参数 ──
    sigma_base   = _sigma_at(field_x, field_y, cfg) + defocus_add
    ellip        = _ellipticity_at(field_x, field_y, cfg)
    # sigma_x = sigma_base * ellip（长轴，边缘更大）
    # sigma_y = sigma_base（短轴，保持基准值）
    # 边缘场点 ellip > 1，光斑变椭圆
    sigma_x_val = sigma_base * ellip  # 长轴
    sigma_y_val = sigma_base  # 短轴

    # ── 5. 亮度独立扰动（每根光纤独立 peak）──
    bv         = cfg["fiber_brightness_variation"]
    peaks_arr  = peak * (1.0 + rng.uniform(-bv, bv, n_valid))

    # ── 6. 背景图 ──
    bg_image = _bg_map(
        (IMG_H, IMG_W),
        BACKGROUND_COUNTS,
        cfg["background_gradient"],
        cfg["background_gradient_strength"],
        rng
    )

    # ── 7. 逐根光纤渲染（每根用独立 peak，支持椭圆）──
    theta_field = np.arctan2(field_y, field_x)
    yg, xg = np.mgrid[0:IMG_H, 0:IMG_W].astype(float)

    signal = bg_image.copy()
    for i in range(n_valid):
        cx, cy = fx_px_dist[i], fy_px_dist[i]
        dx = xg - cx
        dy = yg - cy

        if sigma_x_val == sigma_y_val:
            # 圆形（中心场点）
            exponent = -(dx**2 + dy**2) / (2 * sigma_x_val**2)
        else:
            # 椭圆（边缘场点，旋转角=场点径向方向）
            cos_t = np.cos(theta_field)
            sin_t = np.sin(theta_field)
            a = cos_t**2 / (2*sigma_x_val**2) + sin_t**2 / (2*sigma_y_val**2)
            b = (-np.sin(2*theta_field) / (4*sigma_x_val**2)
                 + np.sin(2*theta_field) / (4*sigma_y_val**2))
            c = sin_t**2 / (2*sigma_x_val**2) + cos_t**2 / (2*sigma_y_val**2)
            exponent = -(a*dx**2 + 2*b*dx*dy + c*dy**2)

        signal += peaks_arr[i] * np.exp(exponent)

    # 泊松噪声 + 读出噪声 + 像元响应不均匀
    image = rng.poisson(np.maximum(signal, 0)).astype(float)
    image += rng.normal(0, READ_NOISE_E, image.shape)
    prnu   = 1.0 + rng.normal(0, cfg["pixel_response_nonuniformity"], (IMG_H, IMG_W))
    image  = image * prnu

    # ── 8. 选取基准光纤（加跳过保护）──
    calib_list = _calib_idx(n_valid, n_calib, calib_dist, rng)
    if len(calib_list) >= n_valid:
        return None, None
    calib_set = set(calib_list)

    # ── 9. 构建真值标注 ──
    # 注意：true_x_px / true_y_px 存的是畸变后坐标（图像里实际的光斑位置）
    #       true_x_mm / true_y_mm 存的是焦面物理坐标（从无畸变坐标换算）
    #       baseline 的任务就是：从畸变后像素坐标 → 焦面物理坐标
    fibers = [
        {
            "fiber_id":        i,
            "is_calib":        i in calib_set,
            "peak_counts": float(peaks_arr[i]),
            "true_x_px":       float(fx_px_dist[i]),   # 图像中实际光斑位置（含畸变）
            "true_y_px":       float(fy_px_dist[i]),
            "true_x_px_ideal": float(fx_px_ideal[i]),  # 无畸变像素坐标（供分析用）
            "true_y_px_ideal": float(fy_px_ideal[i]),
            "true_x_mm":       float(fx_mm[i]),         # 焦面物理坐标真值 mm
            "true_y_mm":       float(fy_mm[i]),
            "true_x_um":       float(fx_mm[i] * 1000), # 焦面物理坐标真值 μm
            "true_y_um":       float(fy_mm[i] * 1000),
        }
        for i in range(n_valid)
    ]

    label = {
        "field_x_mm":      float(field_x),
        "field_y_mm":      float(field_y),
        "r_norm":           float(r_norm),
        "snr_level":        snr_key,
        "defocus_level":    defocus_key,
        "n_calib":          len(calib_list),
        "calib_dist":       calib_dist,
        "sigma_px":         float(sigma_base),
        "sigma_x_px":       float(sigma_x_val),
        "sigma_y_px":       float(sigma_y_val),
        "ellipticity":      float(ellip),
        "distortion_k1":    DISTORTION_K1,
        "distortion_k2":    DISTORTION_K2,
        "n_fibers":         n_valid,
        "calib_indices":    calib_list,
        "fibers":           fibers,
        "img_cx_px":        cx0,
        "img_cy_px":        cy0,
        "scale_um_per_px":  scale,
        "coverage_mm":      round(IMG_W * scale / 1000.0, 1),
    }
    return image.astype(np.float16), label   # float16 节省磁盘空间


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def generate_dataset():
    cfg = DATASET_CONFIG
    rng = np.random.default_rng(cfg["seed"])

    images_dir = os.path.join(cfg["save_dir"], "images")
    labels_dir = os.path.join(cfg["save_dir"], "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    field_pts = _field_points(cfg)
    manifest  = []
    idx       = 0

    n_total_est = (len(field_pts)
                   * len(cfg["snr_levels"])
                   * len(cfg["defocus_levels"])
                   * len(cfg["n_calib_fibers_list"])
                   * len(cfg["calib_distributions"])
                   * cfg["n_repeat"])

    coverage_mm = round(IMG_W * FOCAL_PLANE_SCALE_UM_PX / 1000.0, 1)
    print(f"[DatasetGenerator] 图像尺寸：{IMG_W}×{IMG_H} px")
    print(f"[DatasetGenerator] 覆盖焦面：{coverage_mm} × {coverage_mm} mm")
    print(f"[DatasetGenerator] 每张图光纤数：{N_FIBERS}")
    print(f"[DatasetGenerator] 含径向畸变：K1={DISTORTION_K1}, K2={DISTORTION_K2}")
    print(f"[DatasetGenerator] 预计生成样本数：{n_total_est}")
    print(f"[DatasetGenerator] 保存路径：{os.path.abspath(cfg['save_dir'])}/")

    for (fx, fy, r_norm) in field_pts:
        for snr_key in cfg["snr_levels"]:
            for def_key in cfg["defocus_levels"]:
                for n_calib in cfg["n_calib_fibers_list"]:
                    for dist in cfg["calib_distributions"]:
                        for rep in range(cfg["n_repeat"]):

                            image, label = _generate_one(
                                fx, fy, r_norm,
                                snr_key, def_key,
                                n_calib, dist,
                                cfg, rng
                            )
                            if image is None:
                                continue

                            name               = f"sample_{idx:05d}"
                            label["sample_id"] = name
                            label["repeat"]    = rep

                            np.save(os.path.join(images_dir, f"{name}.npy"), image)
                            with open(os.path.join(labels_dir, f"{name}.json"), "w") as f:
                                json.dump(label, f, indent=2)

                            manifest.append({
                                "id":         name,
                                "r_norm":     r_norm,
                                "field_x_mm": fx,
                                "field_y_mm": fy,
                                "snr_level":  snr_key,
                                "defocus":    def_key,
                                "n_calib":    n_calib,
                                "calib_dist": dist,
                                "repeat":     rep,
                            })
                            idx += 1

                            if idx % 500 == 0:
                                print(f"  已生成 {idx} / {n_total_est} 个样本...")

    manifest_path = os.path.join(cfg["save_dir"], "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DatasetGenerator] 完成，共生成 {idx} 个样本")
    print(f"  图像：{images_dir}/  （float16，每张约 2MB，共约 {idx*2//1024} GB）")
    print(f"  标注：{labels_dir}/")
    print(f"  索引：{manifest_path}")


if __name__ == "__main__":
    generate_dataset()