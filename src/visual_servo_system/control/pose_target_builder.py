"""从目标位姿计算机器人法兰的期望位姿。

闭环几何关系为：
``T_base_marker = T_base_flange_desired · T_flange_camera · T_camera_marker_desired``。
所以求解法兰时必须把右侧两个固定变换按相反顺序求逆。
"""

from __future__ import annotations

import numpy as np

from visual_servo_system.geometry.pose_transforms import (
    compose_transforms,
    invert_transform,
    so3_log,
    validate_transform,
)
from visual_servo_system.models.data_types import PredictedTrajectory


def build_desired_flange_pose(
    T_base_marker: np.ndarray,
    T_flange_camera: np.ndarray,
    T_camera_marker_desired: np.ndarray,
) -> np.ndarray:
    """返回保持期望相机—标志相对位姿所需的 ``T_base_flange``。"""

    return compose_transforms(
        validate_transform(T_base_marker),
        invert_transform(T_camera_marker_desired),
        invert_transform(T_flange_camera),
    )


def build_desired_flange_trajectory(
    marker_trajectory: PredictedTrajectory,
    T_flange_camera: np.ndarray,
    T_camera_marker_desired: np.ndarray,
) -> PredictedTrajectory:
    """逐点把目标轨迹转换成法兰参考轨迹。"""

    transforms = np.stack(
        [
            build_desired_flange_pose(transform, T_flange_camera, T_camera_marker_desired)
            for transform in marker_trajectory.transforms
        ]
    )
    twists = np.zeros((transforms.shape[0], 6), dtype=np.float64)
    # 固定手眼与相机—目标偏置会改变法兰线速度，不能把 marker twist 原样复制。
    # 用相邻法兰参考位姿计算世界坐标线速度和角速度，最后一点沿用上一段。
    for index in range(transforms.shape[0] - 1):
        dt = marker_trajectory.sample_times_s[index + 1] - marker_trajectory.sample_times_s[index]
        twists[index, :3] = (
            transforms[index + 1, :3, 3] - transforms[index, :3, 3]
        ) / dt
        # 左乘增量 ``R_next R_prev^T`` 直接给出 base/spatial 角速度。
        twists[index, 3:] = so3_log(
            transforms[index + 1, :3, :3] @ transforms[index, :3, :3].T
        ) / dt
    if transforms.shape[0] > 1:
        twists[-1] = twists[-2]
    return PredictedTrajectory(
        host_timestamp_s=marker_trajectory.host_timestamp_s,
        sample_times_s=marker_trajectory.sample_times_s,
        transforms=transforms,
        twists=twists,
        valid=marker_trajectory.valid,
        quality=marker_trajectory.quality,
        reason=marker_trajectory.reason,
    )
