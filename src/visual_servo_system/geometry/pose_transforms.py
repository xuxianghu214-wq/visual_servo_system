"""刚体位姿和旋转工具。

本项目统一采用 ``T_parent_child`` 命名：列向量 ``p_child`` 通过
``p_parent = T_parent_child @ p_child`` 映射到父坐标系。矩阵平移单位为米；UR 位姿向量
前三项为米，后三项为轴角旋转向量（弧度）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]


def _finite_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def is_rotation_matrix(rotation: ArrayLike, *, atol: float = 1e-7) -> bool:
    """返回矩阵是否属于 SO(3)，即正交且行列式为 +1。"""

    try:
        matrix = _finite_array(rotation, (3, 3), "rotation")
    except ValueError:
        return False
    return bool(
        np.allclose(matrix.T @ matrix, np.eye(3), atol=atol, rtol=0.0)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=atol, rtol=0.0)
    )


def validate_transform(transform: ArrayLike, *, atol: float = 1e-7) -> FloatArray:
    """校验并复制一个齐次 4×4 刚体变换，不接受缩放、反射或 NaN。"""

    matrix = _finite_array(transform, (4, 4), "transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    if not is_rotation_matrix(matrix[:3, :3], atol=atol):
        raise ValueError("transform rotation must be orthonormal with determinant +1")
    return np.array(matrix, copy=True)


def invert_transform(T_parent_child: ArrayLike) -> FloatArray:
    """返回 ``T_child_parent``，使用刚体逆而非通用矩阵求逆。"""

    transform = validate_transform(T_parent_child)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transforms(*transforms: ArrayLike) -> FloatArray:
    """按书写顺序复合相邻坐标变换。

    例如 ``compose_transforms(T_base_flange, T_flange_camera)`` 得到
    ``T_base_camera``。函数无法从数值矩阵推断坐标系名称，因此调用者仍需保证链条相邻。
    """

    if not transforms:
        raise ValueError("at least one transform is required")
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ validate_transform(transform)
    return validate_transform(result)


def so3_exp(rotation_vector_rad: ArrayLike) -> FloatArray:
    """SO(3) 指数映射：轴角旋转向量（rad）转 3×3 旋转矩阵。"""

    vector = _finite_array(rotation_vector_rad, (3,), "rotation_vector_rad")
    return Rotation.from_rotvec(vector).as_matrix()


def so3_log(rotation: ArrayLike) -> FloatArray:
    """SO(3) 对数映射：3×3 旋转矩阵转最短轴角旋转向量（rad）。"""

    matrix = _finite_array(rotation, (3, 3), "rotation")
    if not is_rotation_matrix(matrix):
        raise ValueError("rotation must be orthonormal with determinant +1")
    return Rotation.from_matrix(matrix).as_rotvec()


def ur_pose_to_transform(pose_ur: ArrayLike) -> FloatArray:
    """把 UR 的 ``[x,y,z,rx,ry,rz]``（m、rad）转为齐次变换。"""

    pose = _finite_array(pose_ur, (6,), "pose_ur")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = so3_exp(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_ur_pose(transform: ArrayLike) -> FloatArray:
    """把齐次变换转为 UR 的 ``[x,y,z,rx,ry,rz]``（m、rad）。"""

    matrix = validate_transform(transform)
    return np.concatenate((matrix[:3, 3], so3_log(matrix[:3, :3])))

