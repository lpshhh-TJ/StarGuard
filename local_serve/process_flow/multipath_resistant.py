# Copyright 2026 lpshhh-TJ. Licensed under the MIT License.
#这是calculate_position.py中使用的数学计算模块

"""
抗多径定位模块
支持 RANSAC、加权最小二乘法、卡尔曼滤波
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Optional


class KalmanFilter3D:
    """
    三维位置卡尔曼滤波器

    状态: [x, y, z, vx, vy, vz] (位置 + 速度)
    """

    def __init__(self, process_noise=0.1, measure_noise=0.5, dt=0.1):
        """
        Args:
            process_noise: 过程噪声方差 Q (目标运动的不确定性)
            measure_noise: 测量噪声方差 R (定位测量的不确定性)
            dt: 时间步长 (秒)
        """
        self.dt = dt

        # 状态向量 [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)

        # 状态协方差矩阵 P
        self.P = np.eye(6) * 1.0

        # 状态转移矩阵 F (匀速模型)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 观测矩阵 H (只观测位置)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # 过程噪声协方差 Q
        self.Q = np.eye(6) * process_noise

        # 测量噪声协方差 R
        self.R = np.eye(3) * measure_noise

        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        更新滤波器

        Args:
            measurement: 测量值 [x, y, z]

        Returns:
            滤波后的位置 [x, y, z]
        """
        if not self.initialized:
            # 初始化状态
            self.x[:3] = measurement
            self.x[3:] = 0  # 初始速度为0
            self.initialized = True
            return measurement

        # 预测步骤
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新步骤
        y = measurement - self.H @ x_pred  # 残差
        S = self.H @ P_pred @ self.H.T + self.R  # 残差协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益

        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred

        return self.x[:3]

    def get_velocity(self) -> np.ndarray:
        """获取当前速度估计"""
        return self.x[3:6]


class RANSACPositioner:
    """
    RANSAC 定位器
    用于识别和剔除受多径影响的基站
    """

    def __init__(self, iterations=100, threshold=0.5, min_inliers=4):
        """
        Args:
            iterations: RANSAC 迭代次数
            threshold: 距离残差阈值 (米)，超过则视为离群点
            min_inliers: 最小内点数量
        """
        self.iterations = iterations
        self.threshold = threshold
        self.min_inliers = min_inliers

    def trilaterate(self, stations: Dict[int, Tuple], distances: Dict[int, float],
                   subset: List[int]) -> Optional[np.ndarray]:
        """
        使用指定基站子集计算位置

        Args:
            stations: 基站坐标 {id: (x, y, z)}
            distances: 测量距离 {id: distance}
            subset: 使用的基站 ID 列表

        Returns:
            位置 [x, y, z] 或 None
        """
        if len(subset) < 4:
            return None

        valid_subset = [s for s in subset if s in stations and s in distances and distances[s] is not None]
        if len(valid_subset) < 4:
            return None

        ref_id = valid_subset[0]
        ref_x, ref_y, ref_z = stations[ref_id]
        ref_d = distances[ref_id]

        A = []
        b = []

        for sid in valid_subset[1:]:
            xi, yi, zi = stations[sid]
            di = distances[sid]

            A.append([2 * (xi - ref_x), 2 * (yi - ref_y), 2 * (zi - ref_z)])
            b.append(xi**2 + yi**2 + zi**2 - ref_x**2 - ref_y**2 - ref_z**2 + ref_d**2 - di**2)

        if not A:
            return None

        try:
            A = np.array(A)
            b = np.array(b)
            result = np.linalg.lstsq(A, b, rcond=None)[0]
            return result
        except np.linalg.LinAlgError:
            return None

    def compute_residuals(self, position: np.ndarray, stations: Dict[int, Tuple],
                         distances: Dict[int, float]) -> Dict[int, float]:
        """
        计算每个基站的距离残差

        Args:
            position: 计算出的位置 [x, y, z]
            stations: 基站坐标
            distances: 测量距离

        Returns:
            每个基站的残差 {id: residual}
        """
        residuals = {}
        for sid, (xi, yi, zi) in stations.items():
            if sid not in distances or distances[sid] is None:
                continue
            expected_dist = np.linalg.norm(position - np.array([xi, yi, zi]))
            residuals[sid] = abs(expected_dist - distances[sid])
        return residuals

    def find_best_consensus(self, stations: Dict[int, Tuple],
                           distances: Dict[int, float]) -> Tuple[Optional[np.ndarray], List[int], Dict[int, float]]:
        """
        使用 RANSAC 找到最一致的基站组合

        Returns:
            (位置, 内点ID列表, 所有基站的残差)
        """
        valid_ids = [s for s in stations.keys() if s in distances and distances[s] is not None]

        if len(valid_ids) < self.min_inliers:
            return None, [], {}

        best_position = None
        best_inliers = []
        best_residuals = {}
        best_score = -1

        for _ in range(self.iterations):
            # 随机选择 min_inliers 个基站
            subset = np.random.choice(valid_ids, min(self.min_inliers, len(valid_ids)), replace=False).tolist()

            # 计算位置
            position = self.trilaterate(stations, distances, subset)
            if position is None:
                continue

            # 计算所有基站的残差
            residuals = self.compute_residuals(position, stations, distances)

            # 统计内点
            inliers = [sid for sid, res in residuals.items() if res < self.threshold]
            score = len(inliers)

            # 优先选择内点多的，其次选择残差和小的
            residual_sum = sum(residuals.values())
            if score > best_score or (score == best_score and residual_sum < sum(best_residuals.values())):
                best_score = score
                best_position = position
                best_inliers = inliers
                best_residuals = residuals

        return best_position, best_inliers, best_residuals


class WeightedLeastSquares:
    """
    加权最小二乘定位
    根据测量精度/残差对基站进行加权
    """

    def compute_weights(self, residuals: Dict[int, float], base_weight: float = 1.0) -> Dict[int, float]:
        """
        根据残差计算权重

        Args:
            residuals: 每个基站的残差
            base_weight: 基础权重

        Returns:
            每个基站的权重 {id: weight}
        """
        weights = {}
        for sid, res in residuals.items():
            # 使用指数衰减函数，残差越大权重越小
            # 权重 = base_weight * exp(-residual / scale)
            weights[sid] = base_weight * np.exp(-res / 0.5)
        return weights

    def solve(self, stations: Dict[int, Tuple], distances: Dict[int, float],
              weights: Optional[Dict[int, float]] = None) -> Optional[np.ndarray]:
        """
        加权最小二乘定位

        Args:
            stations: 基站坐标
            distances: 测量距离
            weights: 每个基站的权重 (可选)

        Returns:
            位置 [x, y, z] 或 None
        """
        valid_ids = [s for s in stations.keys() if s in distances and distances[s] is not None]

        if len(valid_ids) < 4:
            return None

        if weights is None:
            weights = {sid: 1.0 for sid in valid_ids}

        ref_id = valid_ids[0]
        ref_x, ref_y, ref_z = stations[ref_id]
        ref_d = distances[ref_id]
        ref_w = weights.get(ref_id, 1.0)

        A = []
        b = []
        w_list = []

        for sid in valid_ids[1:]:
            xi, yi, zi = stations[sid]
            di = distances[sid]
            w = weights.get(sid, 1.0)

            row = [2 * (xi - ref_x), 2 * (yi - ref_y), 2 * (zi - ref_z)]
            val = xi**2 + yi**2 + zi**2 - ref_x**2 - ref_y**2 - ref_z**2 + ref_d**2 - di**2

            A.append(row)
            b.append(val)
            w_list.append(w)

        if not A:
            return None

        try:
            A = np.array(A)
            b = np.array(b)
            W = np.diag(w_list)

            # 加权最小二乘: (A^T W A)^-1 A^T W b
            result = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b
            return result
        except np.linalg.LinAlgError:
            return None


class MultipathResistantPositioner:
    """
    综合抗多径定位器
    结合 RANSAC + 加权最小二乘 + 卡尔曼滤波
    """

    def __init__(self, ransac_iterations=100, ransac_threshold=0.5,
                 process_noise=0.1, measure_noise=0.5, dt=0.1,
                 enable_ransac=True, enable_kalman=True):
        """
        Args:
            ransac_iterations: RANSAC 迭代次数
            ransac_threshold: RANSAC 残差阈值 (米)
            process_noise: 卡尔曼过程噪声
            measure_noise: 卡尔曼测量噪声
            dt: 时间步长 (秒)
            enable_ransac: 是否启用 RANSAC
            enable_kalman: 是否启用卡尔曼滤波
        """
        self.ransac = RANSACPositioner(iterations=ransac_iterations, threshold=ransac_threshold)
        self.wls = WeightedLeastSquares()
        self.kf = KalmanFilter3D(process_noise=process_noise, measure_noise=measure_noise, dt=dt)

        self.enable_ransac = enable_ransac
        self.enable_kalman = enable_kalman

        # 统计信息
        self.stats = {
            'total_updates': 0,
            'ransac_inliers': [],
            'raw_positions': [],
            'filtered_positions': []
        }

    def compute_position(self, stations: Dict[int, Tuple], distances: Dict[int, float]) -> Tuple[Optional[np.ndarray], Dict]:
        """
        计算抗多径定位结果

        Args:
            stations: 基站坐标 {id: (x, y, z)}
            distances: 测量距离 {id: distance}

        Returns:
            (位置 [x, y, z], 详细信息字典)
        """
        self.stats['total_updates'] += 1
        info = {
            'method': 'standard',
            'inliers': [],
            'residuals': {},
            'raw_position': None,
            'filtered_position': None
        }

        # 准备有效基站数据
        valid_stations = {k: v for k, v in stations.items()
                         if k in distances and distances[k] is not None and distances[k] > 0}
        valid_distances = {k: v for k, v in distances.items()
                          if k in valid_stations}

        if len(valid_stations) < 4:
            return None, info

        position = None

        # 方法 1: RANSAC + 加权最小二乘
        if self.enable_ransac and len(valid_stations) >= 4:
            ransac_pos, inliers, residuals = self.ransac.find_best_consensus(valid_stations, valid_distances)

            if ransac_pos is not None and len(inliers) >= 4:
                # 使用所有内点基站进行加权最小二乘
                inlier_stations = {k: v for k, v in valid_stations.items() if k in inliers}
                inlier_distances = {k: v for k, v in valid_distances.items() if k in inliers}

                # 根据残差计算权重
                weights = self.wls.compute_weights(residuals)

                # 加权最小二乘定位
                position = self.wls.solve(inlier_stations, inlier_distances, weights)

                info['method'] = 'ransac_wls'
                info['inliers'] = inliers
                info['residuals'] = residuals
                self.stats['ransac_inliers'].append(len(inliers))
            else:
                # RANSAC 失败，使用标准方法
                position = self.wls.solve(valid_stations, valid_distances)
                info['method'] = 'standard_fallback'

        # 方法 2: 标准最小二乘（RANSAC 禁用或失败时）
        if position is None:
            position = self.wls.solve(valid_stations, valid_distances)
            info['method'] = 'standard'

        info['raw_position'] = position
        self.stats['raw_positions'].append(position)

        # 卡尔曼滤波
        if self.enable_kalman and position is not None:
            filtered_pos = self.kf.update(position)
            info['filtered_position'] = filtered_pos
            self.stats['filtered_positions'].append(filtered_pos)
            return filtered_pos, info

        return position, info

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['ransac_inliers']:
            stats['avg_inliers'] = np.mean(stats['ransac_inliers'])
            stats['min_inliers'] = np.min(stats['ransac_inliers'])
            stats['max_inliers'] = np.max(stats['ransac_inliers'])
        return stats

    def reset(self):
        """重置滤波器和统计"""
        self.kf = KalmanFilter3D(
            process_noise=self.kf.Q[0, 0],
            measure_noise=self.kf.R[0, 0],
            dt=self.kf.dt
        )
        self.stats = {
            'total_updates': 0,
            'ransac_inliers': [],
            'raw_positions': [],
            'filtered_positions': []
        }


# 便捷函数
def create_positioner(mode='auto', **kwargs) -> MultipathResistantPositioner:
    """
    创建定位器的便捷函数

    Args:
        mode: 'auto', 'ransac_only', 'kalman_only', 'standard'
        **kwargs: 传递给 MultipathResistantPositioner 的参数

    Returns:
        定位器实例
    """
    if mode == 'auto':
        return MultipathResistantPositioner(**kwargs)
    elif mode == 'ransac_only':
        return MultipathResistantPositioner(enable_ransac=True, enable_kalman=False, **kwargs)
    elif mode == 'kalman_only':
        return MultipathResistantPositioner(enable_ransac=False, enable_kalman=True, **kwargs)
    elif mode == 'standard':
        return MultipathResistantPositioner(enable_ransac=False, enable_kalman=False, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
