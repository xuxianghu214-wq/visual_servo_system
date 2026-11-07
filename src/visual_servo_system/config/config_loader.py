"""安全、严格地加载 YAML 配置和标定文件。

这里刻意不提供“缺字段就猜默认值”的行为。视觉伺服会驱动物理机器人，拼写错误、NaN、
单位误用或把占位标定当真值都必须在启动控制前显式失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import yaml
from numpy.typing import NDArray

from visual_servo_system.geometry.pose_transforms import validate_transform

FloatArray = NDArray[np.float64]

_SUPPORTED_APRILTAG_FAMILIES = frozenset(
    {"tag16h5", "tag25h9", "tag36h10", "tag36h11"}
)
_SUPPORTED_ARUCO_DICTIONARIES = frozenset(
    {
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_4X4_1000",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_6X6_50",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_50",
        "DICT_7X7_100",
        "DICT_7X7_250",
        "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_ARUCO_MIP_36h12",
    }
)
_SUPPORTED_CAMERA_PIXEL_FORMATS = frozenset({"Mono8", "BayerRG8"})


class ConfigurationError(ValueError):
    """普通运行配置无效。"""


class CalibrationError(ConfigurationError):
    """相机内参或手眼标定无效/不完整。"""


@dataclass(frozen=True, slots=True)
class CameraIntrinsicsCalibration:
    """OpenCV 针孔相机内参；像素尺寸顺序固定为 ``(宽, 高)``。"""

    image_width_px: int
    image_height_px: int
    camera_matrix: FloatArray
    distortion_model: str
    distortion_coefficients: FloatArray
    units: str
    calibration_complete: bool = True

    @property
    def image_size_px(self) -> tuple[int, int]:
        return self.image_width_px, self.image_height_px


@dataclass(frozen=True, slots=True)
class HandEyeCalibration:
    """眼在手上外参；``T_flange_camera`` 的平移单位固定为米。"""

    parent_frame: str
    child_frame: str
    T_flange_camera: FloatArray
    translation_units: str
    calibration_complete: bool = True


@dataclass(frozen=True, slots=True)
class CameraSettings:
    selection_mode: str
    device_serial: str | None
    acquisition_mode: str
    pixel_format: str
    image_width_px: int
    image_height_px: int
    frame_rate_hz: float
    timestamp_max_uncertainty_s: float = 0.020


@dataclass(frozen=True, slots=True)
class MarkerSettings:
    detector: str
    supported_detectors: tuple[str, ...]
    family: str
    aruco_dictionary: str
    marker_id: int
    marker_size_m: float
    detection_scale: float


@dataclass(frozen=True, slots=True)
class RobotConnectionSettings:
    host: str
    controller_family: str
    polyscope_version: str
    state_frequency_hz: float
    clock_max_uncertainty_s: float
    clock_reset_backjump_s: float
    expected_tcp_pose_ur: FloatArray
    pose_units: str


@dataclass(frozen=True, slots=True)
class ServoTargetSettings:
    parent_frame: str
    child_frame: str
    T_camera_target_desired: FloatArray
    translation_units: str
    rotation_units: str


@dataclass(frozen=True, slots=True)
class MpcControllerSettings:
    control_frequency_hz: float
    dt_s: float
    horizon_steps: int
    solver_time_limit_s: float
    max_linear_acceleration_mps2: float
    max_angular_acceleration_deg_s2: float


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    initial_max_position_error_m: float
    initial_max_rotation_error_deg: float
    workspace_envelope_position_m: float
    workspace_envelope_rotation_deg: float
    commissioning_speed_mps: float
    experiment_speed_mps: float
    commissioning_angular_speed_deg_s: float
    experiment_angular_speed_deg_s: float
    dropout_timeout_s: float
    max_3sigma_position_m: float
    max_3sigma_rotation_deg: float
    stable_frames_required: int
    max_latency_s: float


@dataclass(frozen=True, slots=True)
class DiagnosticsSettings:
    """可丢弃性能诊断通道的有界资源配置。"""

    performance_mode: str
    queue_capacity: int
    batch_size: int
    flush_interval_s: float
    light_summary_interval_s: float
    close_timeout_s: float


@dataclass(frozen=True, slots=True)
class SystemConfig:
    camera: CameraSettings
    marker: MarkerSettings
    robot: RobotConnectionSettings
    target: ServoTargetSettings
    mpc: MpcControllerSettings
    safety: SafetyLimits
    diagnostics: DiagnosticsSettings


def _raise(error_type: type[ConfigurationError], message: str) -> NoReturn:
    raise error_type(message)


def _load_mapping(path: str | Path, error_type: type[ConfigurationError]) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise error_type(f"cannot safely load YAML {source}: {error}") from error
    if not isinstance(value, dict):
        raise error_type(f"{source} must contain one YAML mapping at its root")
    return value


def _exact_keys(
    data: dict[str, Any],
    required: set[str],
    source: str,
    error_type: type[ConfigurationError] = ConfigurationError,
) -> None:
    missing = required - data.keys()
    unknown = data.keys() - required
    if missing:
        _raise(error_type, f"{source} is missing required keys: {', '.join(sorted(missing))}")
    if unknown:
        _raise(error_type, f"{source} has unsupported keys: {', '.join(sorted(unknown))}")


def _finite_number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
    error_type: type[ConfigurationError] = ConfigurationError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _raise(error_type, f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        _raise(error_type, f"{name} must be finite (NaN and infinity are forbidden)")
    if positive and result <= 0.0:
        _raise(error_type, f"{name} must be positive")
    if non_negative and result < 0.0:
        _raise(error_type, f"{name} must be non-negative")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
    error_type: type[ConfigurationError] = ConfigurationError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(error_type, f"{name} must be an integer")
    if positive and value <= 0:
        _raise(error_type, f"{name} must be positive")
    if non_negative and value < 0:
        _raise(error_type, f"{name} must be non-negative")
    return value


def _text(
    value: Any, name: str, error_type: type[ConfigurationError] = ConfigurationError
) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise(error_type, f"{name} must be a non-empty string")
    return value


def _readonly_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
    error_type: type[ConfigurationError],
) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise error_type(f"{name} must be a numeric array") from error
    if array.shape != shape:
        raise error_type(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise error_type(f"{name} must contain only finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def load_camera_intrinsics(
    path: str | Path, *, expected_resolution: tuple[int, int] | None = None
) -> CameraIntrinsicsCalibration:
    """加载完整内参；占位模板和分辨率不匹配都会拒绝。"""

    data = _load_mapping(path, CalibrationError)
    required = {
        "calibration_complete",
        "image_width_px",
        "image_height_px",
        "camera_matrix",
        "distortion_model",
        "distortion_coefficients",
        "units",
    }
    _exact_keys(data, required, str(path), CalibrationError)
    if data["calibration_complete"] is not True:
        raise CalibrationError(f"camera calibration is incomplete: {path}")
    width = _integer(
        data["image_width_px"], "image_width_px", positive=True, error_type=CalibrationError
    )
    height = _integer(
        data["image_height_px"], "image_height_px", positive=True, error_type=CalibrationError
    )
    if expected_resolution is not None and (width, height) != tuple(expected_resolution):
        raise CalibrationError(
            "camera calibration resolution "
            f"{(width, height)} does not match configured resolution {tuple(expected_resolution)}"
        )
    matrix = _readonly_array(
        data["camera_matrix"], (3, 3), "camera_matrix", CalibrationError
    )
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or not np.allclose(
        matrix[2], [0.0, 0.0, 1.0], atol=1e-12
    ):
        raise CalibrationError("camera_matrix is not a valid OpenCV pinhole intrinsic matrix")
    model = _text(data["distortion_model"], "distortion_model", CalibrationError)
    coefficient_counts = {"opencv_plumb_bob": {4, 5, 8, 12, 14}, "opencv_fisheye": {4}}
    if model not in coefficient_counts:
        raise CalibrationError(f"unsupported distortion_model: {model}")
    try:
        coefficients = np.asarray(data["distortion_coefficients"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationError("distortion_coefficients must be numeric") from error
    if coefficients.ndim != 1 or coefficients.size not in coefficient_counts[model]:
        raise CalibrationError(
            f"distortion_coefficients has invalid length for distortion_model {model}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise CalibrationError("distortion_coefficients must contain only finite values")
    coefficients = np.array(coefficients, copy=True)
    coefficients.setflags(write=False)
    units = _text(data["units"], "units", CalibrationError)
    if units != "pixels":
        raise CalibrationError("camera intrinsic units must be 'pixels'")
    return CameraIntrinsicsCalibration(
        width, height, matrix, model, coefficients, units, calibration_complete=True
    )


def load_hand_eye_calibration(path: str | Path) -> HandEyeCalibration:
    """加载完整眼在手上标定，强制法兰→相机光学坐标和米制平移。"""

    data = _load_mapping(path, CalibrationError)
    required = {
        "calibration_complete",
        "parent_frame",
        "child_frame",
        "translation_units",
        "T_flange_camera",
    }
    _exact_keys(data, required, str(path), CalibrationError)
    if data["calibration_complete"] is not True:
        raise CalibrationError(f"hand-eye calibration is incomplete: {path}")
    parent = _text(data["parent_frame"], "parent_frame", CalibrationError)
    child = _text(data["child_frame"], "child_frame", CalibrationError)
    if parent != "flange" or child != "camera_optical":
        raise CalibrationError("hand-eye frames must be parent=flange and child=camera_optical")
    units = _text(data["translation_units"], "translation_units", CalibrationError)
    if units != "metres":
        raise CalibrationError("translation_units must be 'metres'")
    raw_transform = _readonly_array(
        data["T_flange_camera"], (4, 4), "T_flange_camera", CalibrationError
    )
    try:
        transform = validate_transform(raw_transform)
    except ValueError as error:
        raise CalibrationError(f"T_flange_camera {error}") from error
    transform.setflags(write=False)
    return HandEyeCalibration(parent, child, transform, units, calibration_complete=True)


def _load_camera_settings(path: Path) -> CameraSettings:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "selection_mode",
        "device_serial",
        "acquisition_mode",
        "pixel_format",
        "image_width_px",
        "image_height_px",
        "frame_rate_hz",
        "timestamp_max_uncertainty_s",
    }
    _exact_keys(data, required, str(path))
    selection = _text(data["selection_mode"], "selection_mode")
    if selection not in {"auto_unique", "serial"}:
        raise ConfigurationError("selection_mode must be 'auto_unique' or 'serial'")
    serial = data["device_serial"]
    if serial is not None:
        serial = _text(serial, "device_serial")
    if selection == "serial" and serial is None:
        raise ConfigurationError("device_serial is required for serial camera selection")
    acquisition = _text(data["acquisition_mode"], "acquisition_mode")
    if acquisition != "Continuous":
        raise ConfigurationError("acquisition_mode must be 'Continuous'")
    pixel_format = _text(data["pixel_format"], "pixel_format")
    if pixel_format not in _SUPPORTED_CAMERA_PIXEL_FORMATS:
        raise ConfigurationError(
            "pixel_format must be one of "
            f"{sorted(_SUPPORTED_CAMERA_PIXEL_FORMATS)}, got {pixel_format!r}"
        )
    return CameraSettings(
        selection,
        serial,
        acquisition,
        pixel_format,
        _integer(data["image_width_px"], "image_width_px", positive=True),
        _integer(data["image_height_px"], "image_height_px", positive=True),
        _finite_number(data["frame_rate_hz"], "frame_rate_hz", positive=True),
        _finite_number(
            data["timestamp_max_uncertainty_s"],
            "timestamp_max_uncertainty_s",
            positive=True,
        ),
    )


def _load_marker_settings(path: Path) -> MarkerSettings:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "detector",
        "supported_detectors",
        "family",
        "aruco_dictionary",
        "marker_id",
        "marker_size_m",
        "detection_scale",
    }
    _exact_keys(data, required, str(path))
    supported_raw = data["supported_detectors"]
    if not isinstance(supported_raw, list) or not supported_raw:
        raise ConfigurationError("supported_detectors must be a non-empty list")
    supported = tuple(_text(item, "supported_detectors item") for item in supported_raw)
    if len(set(supported)) != len(supported) or not set(supported).issubset({"apriltag", "aruco"}):
        raise ConfigurationError("supported_detectors may contain unique 'apriltag'/'aruco' values")
    detector = _text(data["detector"], "detector")
    if detector not in supported:
        raise ConfigurationError("detector must be listed in supported_detectors")
    family = _text(data["family"], "family")
    if family not in _SUPPORTED_APRILTAG_FAMILIES:
        raise ConfigurationError(
            f"family must be one of {sorted(_SUPPORTED_APRILTAG_FAMILIES)}, got {family!r}"
        )
    aruco = _text(data["aruco_dictionary"], "aruco_dictionary")
    if aruco not in _SUPPORTED_ARUCO_DICTIONARIES:
        raise ConfigurationError(
            "aruco_dictionary must be one of the OpenCV dictionaries supported by this project, "
            f"got {aruco!r}"
        )
    detection_scale = _finite_number(
        data["detection_scale"], "detection_scale", positive=True
    )
    if detection_scale < 0.25 or detection_scale > 1.0:
        raise ConfigurationError("detection_scale must be within [0.25, 1.0]")
    return MarkerSettings(
        detector,
        supported,
        family,
        aruco,
        _integer(data["marker_id"], "marker_id", non_negative=True),
        _finite_number(data["marker_size_m"], "marker_size_m", positive=True),
        detection_scale,
    )


def _load_robot_settings(path: Path) -> RobotConnectionSettings:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "host",
        "controller_family",
        "polyscope_version",
        "state_frequency_hz",
        "clock_max_uncertainty_s",
        "clock_reset_backjump_s",
        "expected_tcp_pose_ur",
        "pose_units",
    }
    _exact_keys(data, required, str(path))
    family = _text(data["controller_family"], "controller_family")
    if family != "CB3":
        raise ConfigurationError("controller_family must be 'CB3' for this commissioned system")
    units = _text(data["pose_units"], "pose_units")
    if units != "metres_radians":
        raise ConfigurationError("pose_units must be 'metres_radians'")
    tcp_pose = _readonly_array(
        data["expected_tcp_pose_ur"], (6,), "expected_tcp_pose_ur", ConfigurationError
    )
    if np.any(tcp_pose != 0.0):
        raise ConfigurationError("expected_tcp_pose_ur must be exactly zero in all six components")
    return RobotConnectionSettings(
        _text(data["host"], "host"),
        family,
        _text(data["polyscope_version"], "polyscope_version"),
        _finite_number(data["state_frequency_hz"], "state_frequency_hz", positive=True),
        _finite_number(
            data["clock_max_uncertainty_s"],
            "clock_max_uncertainty_s",
            positive=True,
        ),
        _finite_number(
            data["clock_reset_backjump_s"],
            "clock_reset_backjump_s",
            positive=True,
        ),
        tcp_pose,
        units,
    )


def _load_target_settings(path: Path) -> ServoTargetSettings:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "parent_frame",
        "child_frame",
        "translation_units",
        "rotation_units",
        "position_m",
        "rotation_vector_rad",
    }
    _exact_keys(data, required, str(path))
    translation_units = _text(data["translation_units"], "translation_units")
    rotation_units = _text(data["rotation_units"], "rotation_units")
    if translation_units != "metres" or rotation_units != "radians":
        raise ConfigurationError("target units must be metres and radians")
    parent_frame = _text(data["parent_frame"], "parent_frame")
    child_frame = _text(data["child_frame"], "child_frame")
    if parent_frame != "camera_optical":
        raise ConfigurationError("parent_frame must be 'camera_optical'")
    if child_frame != "target":
        raise ConfigurationError("child_frame must be 'target'")
    position = _readonly_array(data["position_m"], (3,), "position_m", ConfigurationError)
    rotation_vector = _readonly_array(
        data["rotation_vector_rad"], (3,), "rotation_vector_rad", ConfigurationError
    )
    from visual_servo_system.geometry.pose_transforms import ur_pose_to_transform

    transform = ur_pose_to_transform(np.concatenate((position, rotation_vector)))
    transform.setflags(write=False)
    return ServoTargetSettings(
        parent_frame,
        child_frame,
        transform,
        translation_units,
        rotation_units,
    )


def _load_mpc_settings(path: Path) -> MpcControllerSettings:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "control_frequency_hz",
        "dt_s",
        "horizon_steps",
        "solver_time_limit_s",
        "max_linear_acceleration_mps2",
        "max_angular_acceleration_deg_s2",
    }
    _exact_keys(data, required, str(path))
    frequency = _finite_number(data["control_frequency_hz"], "control_frequency_hz", positive=True)
    dt = _finite_number(data["dt_s"], "dt_s", positive=True)
    if not np.isclose(dt, 1.0 / frequency, atol=1e-12):
        raise ConfigurationError("dt_s must equal 1/control_frequency_hz")
    solver_time_limit = _finite_number(
        data["solver_time_limit_s"], "solver_time_limit_s", positive=True
    )
    if solver_time_limit > dt:
        raise ConfigurationError("solver_time_limit_s must be less than or equal to dt_s")
    return MpcControllerSettings(
        frequency,
        dt,
        _integer(data["horizon_steps"], "horizon_steps", positive=True),
        solver_time_limit,
        _finite_number(
            data["max_linear_acceleration_mps2"],
            "max_linear_acceleration_mps2",
            positive=True,
        ),
        _finite_number(
            data["max_angular_acceleration_deg_s2"],
            "max_angular_acceleration_deg_s2",
            positive=True,
        ),
    )


def _load_safety_limits(path: Path) -> SafetyLimits:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "initial_max_position_error_m",
        "initial_max_rotation_error_deg",
        "workspace_envelope_position_m",
        "workspace_envelope_rotation_deg",
        "commissioning_speed_mps",
        "experiment_speed_mps",
        "commissioning_angular_speed_deg_s",
        "experiment_angular_speed_deg_s",
        "dropout_timeout_s",
        "max_3sigma_position_m",
        "max_3sigma_rotation_deg",
        "stable_frames_required",
        "max_latency_s",
    }
    _exact_keys(data, required, str(path))
    values = {
        name: _finite_number(data[name], name, positive=True)
        for name in required - {"stable_frames_required"}
    }
    if values["initial_max_position_error_m"] > values["workspace_envelope_position_m"]:
        raise ConfigurationError("initial position error limit must fit inside workspace envelope")
    if values["initial_max_rotation_error_deg"] > values["workspace_envelope_rotation_deg"]:
        raise ConfigurationError("initial rotation error limit must fit inside workspace envelope")
    if values["commissioning_speed_mps"] > values["experiment_speed_mps"]:
        raise ConfigurationError("commissioning speed must not exceed experiment speed")
    if (
        values["commissioning_angular_speed_deg_s"]
        > values["experiment_angular_speed_deg_s"]
    ):
        raise ConfigurationError(
            "commissioning angular speed must not exceed experiment angular speed"
        )
    return SafetyLimits(
        values["initial_max_position_error_m"],
        values["initial_max_rotation_error_deg"],
        values["workspace_envelope_position_m"],
        values["workspace_envelope_rotation_deg"],
        values["commissioning_speed_mps"],
        values["experiment_speed_mps"],
        values["commissioning_angular_speed_deg_s"],
        values["experiment_angular_speed_deg_s"],
        values["dropout_timeout_s"],
        values["max_3sigma_position_m"],
        values["max_3sigma_rotation_deg"],
        _integer(data["stable_frames_required"], "stable_frames_required", positive=True),
        values["max_latency_s"],
    )


def _load_diagnostics_settings(path: Path) -> DiagnosticsSettings:
    data = _load_mapping(path, ConfigurationError)
    required = {
        "performance_mode",
        "queue_capacity",
        "batch_size",
        "flush_interval_s",
        "light_summary_interval_s",
        "close_timeout_s",
    }
    _exact_keys(data, required, str(path))
    mode = _text(data["performance_mode"], "performance_mode")
    if mode not in {"OFF", "LIGHT", "FULL_ASYNC"}:
        raise ConfigurationError("performance_mode must be OFF, LIGHT, or FULL_ASYNC")
    queue_capacity = _integer(data["queue_capacity"], "queue_capacity", positive=True)
    batch_size = _integer(data["batch_size"], "batch_size", positive=True)
    if batch_size > queue_capacity * 64:
        raise ConfigurationError("batch_size is unreasonably large for queue_capacity")
    return DiagnosticsSettings(
        performance_mode=mode,
        queue_capacity=queue_capacity,
        batch_size=batch_size,
        flush_interval_s=_finite_number(
            data["flush_interval_s"], "flush_interval_s", positive=True
        ),
        light_summary_interval_s=_finite_number(
            data["light_summary_interval_s"],
            "light_summary_interval_s",
            positive=True,
        ),
        close_timeout_s=_finite_number(
            data["close_timeout_s"], "close_timeout_s", positive=True
        ),
    )


def load_system_config(config_directory: str | Path) -> SystemConfig:
    """加载运行配置文件并返回不可变的类型化配置树。"""

    directory = Path(config_directory)
    return SystemConfig(
        camera=_load_camera_settings(directory / "camera_settings.yaml"),
        marker=_load_marker_settings(directory / "marker_settings.yaml"),
        robot=_load_robot_settings(directory / "robot_connection.yaml"),
        target=_load_target_settings(directory / "servo_target.yaml"),
        mpc=_load_mpc_settings(directory / "mpc_controller.yaml"),
        safety=_load_safety_limits(directory / "safety_limits.yaml"),
        diagnostics=_load_diagnostics_settings(directory / "diagnostics_settings.yaml"),
    )
