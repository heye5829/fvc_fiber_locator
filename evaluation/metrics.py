"""
evaluation/metrics.py
提供统一的误差计算指标
"""
import numpy as np


def calculate_errors(true_pts, pred_pts):
    """
    计算两组点之间的误差指标
    参数:
        true_pts: 真值坐标点集，shape (N, 2)
        pred_pts: 预测坐标点集，shape (N, 2)
    返回:
        字典包含 RMSE, Mean, Max 误差及所有点的距离列表
    """
    true_pts = np.array(true_pts)
    pred_pts = np.array(pred_pts)

    # 计算每个点的欧氏距离
    distances = np.linalg.norm(true_pts - pred_pts, axis=1)

    return {
        "rmse": float(np.sqrt(np.mean(distances ** 2))),  # 均方根误差
        "mean": float(np.mean(distances)),  # 平均误差
        "max": float(np.max(distances)),  # 最大误差
        "distances": distances  # 保存所有误差以便画图
    }