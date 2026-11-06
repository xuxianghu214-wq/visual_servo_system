"""线程之间传递的公共数据类型。

所有 ``host_timestamp_s`` 都使用主机单调时钟坐标，单位为秒。对有设备时钟的
相机和机器人，它表示设备采样/曝光时刻映射到 ``time.monotonic()`` 后的时刻；
对无设备时钟的后端才退化为主机接收时刻或调用中点，并明确标为不可靠。
该时间只能用于同一台主机上的先后顺序和时间差计算，不能当作 UTC 墙上时钟。
数组会在构造时复制并设为只读；
这样即使生产者随后复用相机缓冲区，消费者看到的数据也不会悄悄变化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from visual_servo_system.geometry.pose_transforms import validate_transform

FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.generic]


def _strict_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise TypeError(f"{name} must be a floating-point number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: object) -> float:
    result = _strict_float(value, "host_timestamp_s")
    if result < 0.0:
        raise ValueError("host_timestamp_s must be a finite, non-negative monotonic time")
    return result


def _quality(value: object) -> float:
    result = _strict_float(value, "quality")
    if not 0.0 <= result <= 1.0:
        raise ValueError("quality must be finite and within [0, 1]")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _strict_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer and must not be bool")
    return value


def _strict_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """复制到不可变 ``bytes`` 后备存储，使 WRITEABLE 标志不能被重新打开。"""

    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _require_ndarray(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    return value


def _readonly_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    source = _require_ndarray(value, name)
    try:
        array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric NumPy array") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _immutable_array(array)


def _readonly_dynamic_array(value: object, name: str) -> FloatArray:
    source = _require_ndarray(value, name)
    try:
        array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric NumPy array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _immutable_array(array)


def _transform(value: object, name: str) -> FloatArray:
    source = _require_ndarray(value, name)
    try:
        transform = validate_transform(source)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    return _immutable_array(transform)


def _non_negative_float(value: object, name: str) -> float:
    result = _strict_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _timestamp_uncertainty(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise TypeError("timestamp_uncertainty_s must be a floating-point number")
    result = float(value)
    if np.isnan(result) or result < 0.0:
        raise ValueError("timestamp_uncertainty_s must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class FramePacket:
    """一帧相机图像。

    ``image`` 使用相机原始像素坐标（列为 x、行为 y）；Mono8 图像形状为 ``(H, W)``。
    ``sequence_number`` 是采集线程递增序号，``quality`` 是归一化到 0～1 的采集质量。
    """

    host_timestamp_s: float
    sequence_number: int
    image: ImageArray
    valid: bool
    quality: float
    receive_timestamp_s: float | None = None
    timestamp_source: str = "host_receive"
    timestamp_reliable: bool = False
    timestamp_uncertainty_s: float = float("inf")

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        sequence_number = _strict_integer(self.sequence_number, "sequence_number")
        if sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        image_source = _require_ndarray(self.image, "image")
        if image_source.dtype.hasobject:
            raise TypeError("image must not use object dtype")
        image = _immutable_array(image_source)
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("image must be a non-empty HxW or HxWxC array")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(self, "quality", _quality(self.quality))
        receive = (
            self.host_timestamp_s
            if self.receive_timestamp_s is None
            else _timestamp(self.receive_timestamp_s)
        )
        if receive + 1e-9 < self.host_timestamp_s:
            raise ValueError("receive_timestamp_s must not precede capture host_timestamp_s")
        object.__setattr__(self, "receive_timestamp_s", receive)
        object.__setattr__(
            self, "timestamp_source", _strict_text(self.timestamp_source, "timestamp_source")
        )
        object.__setattr__(
            self,
            "timestamp_reliable",
            _strict_bool(self.timestamp_reliable, "timestamp_reliable"),
        )
        object.__setattr__(
            self,
            "timestamp_uncertainty_s",
            _timestamp_uncertainty(self.timestamp_uncertainty_s),
        )


@dataclass(frozen=True, slots=True)
class MarkerPoseObservation:
    """单次标志物位姿观测。

    ``T_camera_marker`` 遵循 ``T_parent_child`` 约定：把 marker 坐标系中的点映射到
    ``camera_optical`` 坐标系。平移单位为米，旋转无量纲；重投影误差单位为像素，
    ``pixel_area_px2`` 单位为像素平方。OpenCV ArUco 没有 decision margin，此时该字段为 None，
    绝不能用像素面积冒充检测器置信边界。
    """

    host_timestamp_s: float
    marker_id: int
    T_camera_marker: FloatArray
    reprojection_error_px: float
    decision_margin: float | None
    pixel_area_px2: float
    valid: bool
    quality: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        marker_id = _strict_integer(self.marker_id, "marker_id")
        if marker_id < 0:
            raise ValueError("marker_id must be non-negative")
        object.__setattr__(
            self, "T_camera_marker", _transform(self.T_camera_marker, "T_camera_marker")
        )
        object.__setattr__(
            self,
            "reprojection_error_px",
            _non_negative_float(self.reprojection_error_px, "reprojection_error_px"),
        )
        if self.decision_margin is not None:
            object.__setattr__(
                self,
                "decision_margin",
                _non_negative_float(self.decision_margin, "decision_margin"),
            )
        object.__setattr__(
            self,
            "pixel_area_px2",
            _non_negative_float(self.pixel_area_px2, "pixel_area_px2"),
        )
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(self, "quality", _quality(self.quality))


@dataclass(frozen=True, slots=True)
class RobotStateSample:
    """一个带主机时间戳的 UR 机器人状态。

    ``T_base_flange`` 把法兰坐标映射到机器人 base 坐标，平移单位为米。
    ``tcp_speed`` 前三项是 m/s、后三项是 rad/s；关节位置单位为 rad。这里明确保存法兰
    位姿而不是工具位姿，避免非零 TCP 设置被误当作手眼标定的一部分。
    """

    host_timestamp_s: float
    T_base_flange: FloatArray
    tcp_speed: FloatArray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    joint_positions_rad: FloatArray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    sequence_number: int = 0
    robot_mode: str = "unknown"
    valid: bool = True
    quality: float = 1.0
    receive_timestamp_s: float | None = None
    timestamp_source: str = "host_midpoint"
    timestamp_reliable: bool = False
    timestamp_uncertainty_s: float = float("inf")

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        object.__setattr__(
            self, "T_base_flange", _transform(self.T_base_flange, "T_base_flange")
        )
        object.__setattr__(self, "tcp_speed", _readonly_array(self.tcp_speed, (6,), "tcp_speed"))
        object.__setattr__(
            self,
            "joint_positions_rad",
            _readonly_array(self.joint_positions_rad, (6,), "joint_positions_rad"),
        )
        sequence_number = _strict_integer(self.sequence_number, "sequence_number")
        if sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        object.__setattr__(self, "robot_mode", _strict_text(self.robot_mode, "robot_mode"))
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(self, "quality", _quality(self.quality))
        receive = (
            self.host_timestamp_s
            if self.receive_timestamp_s is None
            else _timestamp(self.receive_timestamp_s)
        )
        if receive + 1e-9 < self.host_timestamp_s:
            raise ValueError("receive_timestamp_s must not precede sample host_timestamp_s")
        object.__setattr__(self, "receive_timestamp_s", receive)
        object.__setattr__(
            self, "timestamp_source", _strict_text(self.timestamp_source, "timestamp_source")
        )
        object.__setattr__(
            self,
            "timestamp_reliable",
            _strict_bool(self.timestamp_reliable, "timestamp_reliable"),
        )
        object.__setattr__(
            self,
            "timestamp_uncertainty_s",
            _timestamp_uncertainty(self.timestamp_uncertainty_s),
        )


@dataclass(frozen=True, slots=True)
class TargetStateEstimate:
    """目标在机器人 base 坐标系中的滤波状态。

    ``T_base_target`` 的平移单位为米；线速度在 base 中为 m/s，角速度在 target 局部坐标中
    为 rad/s，与 EKF 的右乘 SO(3) 误差定义一致。协方差固定为 12×12，对应位置、旋转误差、
    线速度、角速度各三维。``measurement_age_s`` 表示距
    最近视觉观测的时间，可供安全层判定掉线。
    """

    host_timestamp_s: float
    T_base_target: FloatArray
    linear_velocity_mps: FloatArray
    angular_velocity_rad_s: FloatArray
    covariance: FloatArray
    valid: bool
    quality: float
    measurement_age_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        object.__setattr__(self, "T_base_target", _transform(self.T_base_target, "T_base_target"))
        object.__setattr__(
            self,
            "linear_velocity_mps",
            _readonly_array(self.linear_velocity_mps, (3,), "linear_velocity_mps"),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _readonly_array(self.angular_velocity_rad_s, (3,), "angular_velocity_rad_s"),
        )
        covariance = _readonly_array(self.covariance, (12, 12), "covariance")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
            raise ValueError("covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", covariance)
        age = _non_negative_float(self.measurement_age_s, "measurement_age_s")
        object.__setattr__(self, "measurement_age_s", age)
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(self, "quality", _quality(self.quality))


@dataclass(frozen=True, slots=True)
class PredictedTrajectory:
    """MPC 从当前时刻起预测的目标轨迹。

    ``sample_times_s`` 是相对当前主机时间的秒数；``transforms[i]`` 为第 i 个
    ``T_base_target``；``twists`` 每行前三项为 base 中的 m/s、后三项为 base/spatial rad/s。
    """

    host_timestamp_s: float
    sample_times_s: FloatArray
    transforms: FloatArray
    twists: FloatArray
    valid: bool
    quality: float
    reason: str = "ok"

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        times = _readonly_dynamic_array(self.sample_times_s, "sample_times_s")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("sample_times_s must be a non-empty one-dimensional array")
        if np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("sample_times_s must be non-negative and strictly increasing")
        transforms = _readonly_dynamic_array(self.transforms, "transforms")
        expected_transform_shape = (times.size, 4, 4)
        if transforms.shape != expected_transform_shape:
            raise ValueError(
                f"transforms must have shape {expected_transform_shape}, got {transforms.shape}"
            )
        for transform in transforms:
            _transform(transform, "transforms")
        twists = _readonly_dynamic_array(self.twists, "twists")
        expected_twist_shape = (times.size, 6)
        if twists.shape != expected_twist_shape:
            raise ValueError(f"twists must have shape {expected_twist_shape}, got {twists.shape}")
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "twists", twists)
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(self, "quality", _quality(self.quality))
        object.__setattr__(self, "reason", _strict_text(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class MpcResult:
    """一次 MPC 求解输出。

    ``commanded_twist`` 表示控制坐标系中的速度旋量：前三项 m/s，后三项 rad/s。
    ``solve_time_s`` 是本次求解耗时，安全层必须检查它是否超过控制周期预算。
    """

    host_timestamp_s: float
    commanded_twist: FloatArray
    solve_time_s: float
    solver_status: str
    valid: bool
    quality: float
    predicted_trajectory: PredictedTrajectory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        object.__setattr__(
            self,
            "commanded_twist",
            _readonly_array(self.commanded_twist, (6,), "commanded_twist"),
        )
        solve_time = _non_negative_float(self.solve_time_s, "solve_time_s")
        object.__setattr__(self, "solve_time_s", solve_time)
        object.__setattr__(self, "solver_status", _strict_text(self.solver_status, "solver_status"))
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(self, "quality", _quality(self.quality))
        if self.predicted_trajectory is not None and not isinstance(
            self.predicted_trajectory, PredictedTrajectory
        ):
            raise TypeError("predicted_trajectory must be PredictedTrajectory or None")


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """安全层对控制指令作出的最终决定。

    ``safe_twist`` 是唯一允许发送给机器人的速度旋量（m/s、rad/s）。即使
    ``allow_motion`` 为假也保留六维零向量，便于执行线程采用统一接口安全停车。
    """

    host_timestamp_s: float
    allow_motion: bool
    safe_twist: FloatArray
    reason: str
    valid: bool
    quality: float
    decision_generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_timestamp_s", _timestamp(self.host_timestamp_s))
        object.__setattr__(
            self, "allow_motion", _strict_bool(self.allow_motion, "allow_motion")
        )
        object.__setattr__(self, "valid", _strict_bool(self.valid, "valid"))
        object.__setattr__(
            self, "safe_twist", _readonly_array(self.safe_twist, (6,), "safe_twist")
        )
        object.__setattr__(self, "reason", _strict_text(self.reason, "reason"))
        object.__setattr__(self, "quality", _quality(self.quality))
        generation = _strict_integer(self.decision_generation, "decision_generation")
        if generation < 0:
            raise ValueError("decision_generation must be non-negative")
        object.__setattr__(self, "decision_generation", generation)
        if (not self.allow_motion or not self.valid) and np.any(self.safe_twist != 0.0):
            raise ValueError("safe_twist must be exactly zero when motion is forbidden or invalid")
