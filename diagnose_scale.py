"""
优先级1诊断脚本：scale/单位问题
运行：python diagnose_scale.py
"""
import numpy as np
import json
import os
import sys

# ============================================================
# STEP 1: 打印config中所有scale相关参数
# ============================================================
print("=" * 60)
print("STEP 1: 配置参数检查")
print("=" * 60)

try:
    import config as cfg

    print("\n【config.py 中所有参数】")
    for attr in dir(cfg):
        if not attr.startswith('_'):
            val = getattr(cfg, attr)
            if not callable(val):
                print(f"  {attr:40s} = {val}")
except Exception as e:
    print(f"  ❌ 导入config失败: {e}")

# ============================================================
# STEP 2: 从标签文件读取已知物理坐标
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: 标签文件解析")
print("=" * 60)

# 自动查找标签文件
label_dir = "dataset/labels"
label_files = []
if os.path.exists(label_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    print(f"  找到标签文件: {label_files}")

if label_files:
    label_path = os.path.join(label_dir, label_files[0])
    print(f"\n  读取: {label_path}")
    with open(label_path, 'r') as f:
        label = json.load(f)

    print(f"  标签文件顶层键: {list(label.keys())}")

    # 打印前3个光纤的完整信息
    fibers = label.get('fibers', label.get('spots', label.get('references', [])))
    print(f"\n  光纤/基准点总数: {len(fibers)}")
    print("\n  前3个条目详情:")
    for i, fiber in enumerate(fibers[:3]):
        print(f"    [{i}] {json.dumps(fiber, indent=6)}")

# ============================================================
# STEP 3: 计算关键比值
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Scale一致性验证")
print("=" * 60)

try:
    # 从config获取参数
    pixel_size = getattr(cfg, 'PIXEL_SIZE_UM', None)
    focal_scale = getattr(cfg, 'FOCAL_PLANE_SCALE_UM_PX', None)
    plate_scale = getattr(cfg, 'PLATE_SCALE', None)
    image_width = getattr(cfg, 'IMAGE_WIDTH', None)
    image_height = getattr(cfg, 'IMAGE_HEIGHT', None)

    print(f"\n  像素物理尺寸:     {pixel_size} μm/px")
    print(f"  焦面比例因子:     {focal_scale} μm/px（焦面坐标系→像素）")
    print(f"  板比例:           {plate_scale}")
    print(f"  图像尺寸:         {image_width} × {image_height} px")

    if focal_scale:
        print(f"\n  【关键推导】")
        print(f"  0.1 px  × {focal_scale} = {0.1 * focal_scale:.3f} μm")
        print(f"  0.05 px × {focal_scale} = {0.05 * focal_scale:.3f} μm")
        print(f"  图像宽度覆盖焦面: {image_width * focal_scale / 1000:.1f} mm")
        print(f"  图像高度覆盖焦面: {image_height * focal_scale / 1000:.1f} mm")

        # 判断合理性
        if focal_scale < 5:
            print(f"\n  ✅ scale={focal_scale} μm/px → 合理（高分辨率）")
        elif focal_scale < 30:
            print(f"\n  ✅ scale={focal_scale} μm/px → 合理（标准FVC）")
        elif focal_scale < 100:
            print(f"\n  ⚠️  scale={focal_scale} μm/px → 偏大，需确认")
        else:
            print(f"\n  ❌ scale={focal_scale} μm/px → 异常！可能单位错误")

except Exception as e:
    print(f"  计算出错: {e}")

# ============================================================
# STEP 4: 从标签数据估算实际scale
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: 从标签数据反推实际scale")
print("=" * 60)

if label_files:
    try:
        with open(os.path.join(label_dir, label_files[0]), 'r') as f:
            label = json.load(f)

        fibers = label.get('fibers', label.get('spots',
                                               label.get('references', [])))

        if len(fibers) >= 2:
            # 提取所有点的像素坐标和物理坐标
            px_coords = []
            phy_coords = []

            for fiber in fibers:
                # 尝试不同的键名
                px = fiber.get('pixel_x', fiber.get('cx', fiber.get('x_px')))
                py = fiber.get('pixel_y', fiber.get('cy', fiber.get('y_px')))
                fx = fiber.get('focal_x', fiber.get('x_um', fiber.get('x_focal')))
                fy = fiber.get('focal_y', fiber.get('y_um', fiber.get('y_focal')))

                if all(v is not None for v in [px, py, fx, fy]):
                    px_coords.append([px, py])
                    phy_coords.append([fx, fy])

            px_coords = np.array(px_coords)
            phy_coords = np.array(phy_coords)

            print(f"\n  成功提取 {len(px_coords)} 个点对")

            if len(px_coords) >= 2:
                # 计算所有点对的距离比
                scales = []
                for i in range(min(20, len(px_coords))):
                    for j in range(i + 1, min(20, len(px_coords))):
                        dpx = np.linalg.norm(px_coords[i] - px_coords[j])
                        dphy = np.linalg.norm(phy_coords[i] - phy_coords[j])
                        if dpx > 10:  # 排除太近的点对
                            scales.append(dphy / dpx)

                scales = np.array(scales)
                print(f"\n  从点对距离反推scale:")
                print(f"  中位数: {np.median(scales):.4f} μm/px")
                print(f"  均值:   {np.mean(scales):.4f} μm/px")
                print(f"  标准差: {np.std(scales):.4f} μm/px")
                print(f"  范围:   [{scales.min():.4f}, {scales.max():.4f}] μm/px")

                actual_scale = np.median(scales)
                config_scale = getattr(cfg, 'FOCAL_PLANE_SCALE_UM_PX', None)

                if config_scale:
                    ratio = actual_scale / config_scale
                    print(f"\n  实际scale / config中scale = {ratio:.3f}")
                    if abs(ratio - 1.0) < 0.05:
                        print(f"  ✅ 两者一致（误差<5%），scale参数正确")
                    elif abs(ratio - 1.0) < 0.2:
                        print(f"  ⚠️  两者差异{(ratio - 1) * 100:.1f}%，轻微不一致")
                    else:
                        print(f"  ❌ 两者差异{(ratio - 1) * 100:.1f}%！scale参数错误！")
                        print(f"     应将 FOCAL_PLANE_SCALE_UM_PX 修改为 {actual_scale:.4f}")

                # 打印像素坐标和物理坐标的范围
                print(f"\n  像素坐标范围:")
                print(f"    X: [{px_coords[:, 0].min():.1f}, {px_coords[:, 0].max():.1f}] px")
                print(f"    Y: [{px_coords[:, 1].min():.1f}, {px_coords[:, 1].max():.1f}] px")
                print(f"  物理坐标范围:")
                print(f"    X: [{phy_coords[:, 0].min():.1f}, {phy_coords[:, 0].max():.1f}] μm(?)")
                print(f"    Y: [{phy_coords[:, 1].min():.1f}, {phy_coords[:, 1].max():.1f}] μm(?)")

                # 物理坐标单位推断
                phy_range = phy_coords.max() - phy_coords.min()
                print(f"\n  物理坐标量级: {phy_range.max():.1f}")
                if phy_range.max() > 100000:
                    print(f"  ⚠️  物理坐标可能单位为nm，需÷1000转μm")
                elif phy_range.max() > 1000:
                    print(f"  ✅ 物理坐标可能单位为μm（合理）")
                elif phy_range.max() > 10:
                    print(f"  ⚠️  物理坐标可能单位为mm，需×1000转μm")
                else:
                    print(f"  ⚠️  物理坐标可能单位为cm或m")

    except Exception as e:
        print(f"  分析出错: {e}")
        import traceback;

        traceback.print_exc()

# ============================================================
# STEP 5: 检测结果误差的单位确认
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: 18μm误差的来源确认")
print("=" * 60)
print("""
  问题：bt_main_fvccalibrator.py输出18μm误差

  可能情况A: scale正确，误差确实是18μm（检测/标定问题）
  可能情况B: scale=180μm/px（错误），实际像素误差0.1px被放大10倍
  可能情况C: 物理坐标单位是mm而非μm，误差0.018mm=18μm（实际合理）

  请检查 bt_main_fvccalibrator.py 中误差计算代码：
  误差 = ||predicted_focal - true_focal|| × ???

  如果true_focal单位是mm，误差×1000才是μm！
""")

print("=" * 60)
print("诊断完成！请将完整输出粘贴给我分析")
print("=" * 60)