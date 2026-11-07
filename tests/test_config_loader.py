from pathlib import Path

import numpy as np
import pytest
import yaml

from visual_servo_system.config.config_loader import (
    CalibrationError,
    ConfigurationError,
    load_camera_intrinsics,
    load_hand_eye_calibration,
    load_system_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _copy_config_files(destination: Path) -> None:
    for source in (PROJECT_ROOT / "config_files").glob("*.yaml"):
        (destination / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )


def _valid_intrinsics() -> dict[str, object]:
    return {
        "calibration_complete": True,
        "image_width_px": 1280,
        "image_height_px": 1024,
        "camera_matrix": [[900.0, 0.0, 640.0], [0.0, 901.0, 512.0], [0.0, 0.0, 1.0]],
        "distortion_model": "opencv_plumb_bob",
        "distortion_coefficients": [0.1, -0.02, 0.0, 0.0, 0.0],
        "units": "pixels",
    }


def _valid_hand_eye() -> dict[str, object]:
    return {
        "calibration_complete": True,
        "parent_frame": "flange",
        "child_frame": "camera_optical",
        "translation_units": "metres",
        "T_flange_camera": [
            [1.0, 0.0, 0.0, 0.03],
            [0.0, 1.0, 0.0, -0.01],
            [0.0, 0.0, 1.0, 0.08],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_incomplete_calibration_templates_are_rejected_for_servo(tmp_path: Path) -> None:
    camera_path = tmp_path / "camera_intrinsics.yaml"
    hand_eye_path = tmp_path / "hand_eye_eye_in_hand.yaml"
    _write_yaml(
        camera_path,
        {
            "calibration_complete": False,
            "image_width_px": None,
            "image_height_px": None,
            "camera_matrix": None,
            "distortion_model": None,
            "distortion_coefficients": None,
            "units": None,
        },
    )
    _write_yaml(
        hand_eye_path,
        {
            "calibration_complete": False,
            "parent_frame": "flange",
            "child_frame": "camera_optical",
            "translation_units": "metres",
            "T_flange_camera": None,
        },
    )
    with pytest.raises(CalibrationError, match="incomplete"):
        load_camera_intrinsics(camera_path)
    with pytest.raises(CalibrationError, match="incomplete"):
        load_hand_eye_calibration(hand_eye_path)


def test_complete_calibration_is_loaded_into_typed_objects(tmp_path: Path) -> None:
    intrinsics_path = tmp_path / "intrinsics.yaml"
    hand_eye_path = tmp_path / "hand_eye.yaml"
    _write_yaml(intrinsics_path, _valid_intrinsics())
    _write_yaml(hand_eye_path, _valid_hand_eye())

    intrinsics = load_camera_intrinsics(intrinsics_path, expected_resolution=(1280, 1024))
    hand_eye = load_hand_eye_calibration(hand_eye_path)

    assert intrinsics.image_size_px == (1280, 1024)
    assert intrinsics.camera_matrix.shape == (3, 3)
    assert hand_eye.parent_frame == "flange"
    assert hand_eye.child_frame == "camera_optical"
    np.testing.assert_allclose(hand_eye.T_flange_camera[:3, 3], [0.03, -0.01, 0.08])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.update(camera_matrix=[[1.0, 0.0], [0.0, 1.0]]), "camera_matrix"),
        (lambda data: data.update(distortion_model="mystery_model"), "distortion_model"),
        (lambda data: data.update(units="millimetres"), "units"),
        (lambda data: data.update(image_width_px=None), "image_width_px"),
        (
            lambda data: data.update(
                camera_matrix=[[900.0, 0.0, float("nan")], [0.0, 900.0, 512.0], [0, 0, 1]]
            ),
            "finite",
        ),
    ],
)
def test_intrinsics_reject_invalid_values(tmp_path: Path, mutation, match: str) -> None:
    data = _valid_intrinsics()
    mutation(data)
    path = tmp_path / "intrinsics.yaml"
    _write_yaml(path, data)

    with pytest.raises(CalibrationError, match=match):
        load_camera_intrinsics(path)


def test_intrinsics_reject_resolution_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "intrinsics.yaml"
    _write_yaml(path, _valid_intrinsics())

    with pytest.raises(CalibrationError, match="resolution"):
        load_camera_intrinsics(path, expected_resolution=(640, 480))


def test_hand_eye_rejects_bad_rotation_and_wrong_units(tmp_path: Path) -> None:
    bad_rotation = _valid_hand_eye()
    bad_rotation["T_flange_camera"][0][0] = 1.1  # type: ignore[index]
    rotation_path = tmp_path / "rotation.yaml"
    _write_yaml(rotation_path, bad_rotation)

    with pytest.raises(CalibrationError, match="orthonormal"):
        load_hand_eye_calibration(rotation_path)

    bad_units = _valid_hand_eye()
    bad_units["translation_units"] = "millimetres"
    units_path = tmp_path / "units.yaml"
    _write_yaml(units_path, bad_units)

    with pytest.raises(CalibrationError, match="translation_units"):
        load_hand_eye_calibration(units_path)


def test_default_system_configuration_matches_approved_values() -> None:
    config = load_system_config(PROJECT_ROOT / "config_files")

    assert config.camera.selection_mode == "serial"
    assert config.camera.device_serial == "16276663"
    assert config.camera.acquisition_mode == "Continuous"
    # 彩色 BFS-U3-13Y3C 的 Mono8 会强制 ISP 2x binning，只能得到 640x512。
    assert config.camera.pixel_format == "BayerRG8"
    assert config.camera.timestamp_max_uncertainty_s == pytest.approx(0.020)
    assert config.marker.detector == "apriltag"
    assert config.marker.family == "tag36h11"
    assert config.marker.marker_id == 0
    assert config.marker.marker_size_m == pytest.approx(0.08)
    assert config.marker.detection_scale == pytest.approx(0.5)
    assert "aruco" in config.marker.supported_detectors
    assert config.robot.host == "192.168.0.10"
    assert config.robot.controller_family == "CB3"
    assert config.robot.polyscope_version == "3.7.2"
    assert config.robot.state_frequency_hz == pytest.approx(125.0)
    assert config.robot.clock_max_uncertainty_s == pytest.approx(0.005)
    assert config.robot.clock_reset_backjump_s == pytest.approx(1.0)
    np.testing.assert_allclose(config.robot.expected_tcp_pose_ur, np.zeros(6))
    np.testing.assert_allclose(config.target.T_camera_target_desired[:3, 3], [0, 0, 0.5])
    np.testing.assert_allclose(
        config.target.T_camera_target_desired[:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-12
    )
    assert config.mpc.control_frequency_hz == pytest.approx(20.0)
    assert config.mpc.dt_s == pytest.approx(0.05)
    assert config.mpc.horizon_steps == 20
    assert config.mpc.solver_time_limit_s == pytest.approx(0.02)
    assert config.mpc.max_linear_acceleration_mps2 == pytest.approx(0.25)
    assert config.mpc.max_angular_acceleration_deg_s2 == pytest.approx(90.0)
    assert config.safety.initial_max_position_error_m == pytest.approx(0.08)
    assert config.safety.initial_max_rotation_error_deg == pytest.approx(10.0)
    assert config.safety.workspace_envelope_position_m == pytest.approx(0.1)
    assert config.safety.workspace_envelope_rotation_deg == pytest.approx(15.0)
    assert config.safety.commissioning_speed_mps == pytest.approx(0.05)
    assert config.safety.experiment_speed_mps == pytest.approx(0.12)
    assert config.safety.commissioning_angular_speed_deg_s == pytest.approx(15.0)
    assert config.safety.experiment_angular_speed_deg_s == pytest.approx(30.0)
    assert config.safety.dropout_timeout_s == pytest.approx(1.0)
    assert config.safety.max_3sigma_position_m == pytest.approx(0.03)
    assert config.safety.max_3sigma_rotation_deg == pytest.approx(5.0)
    assert config.safety.stable_frames_required == 10
    assert config.safety.max_latency_s == pytest.approx(0.1)
    assert config.diagnostics.performance_mode == "OFF"
    assert config.diagnostics.queue_capacity == 4096
    assert config.diagnostics.batch_size == 256
    assert config.diagnostics.flush_interval_s == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["OFF", "LIGHT", "FULL_ASYNC"])
def test_diagnostics_mode_is_strictly_loaded(tmp_path: Path, mode: str) -> None:
    _copy_config_files(tmp_path)
    path = tmp_path / "diagnostics_settings.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["performance_mode"] = mode
    _write_yaml(path, data)

    assert load_system_config(tmp_path).diagnostics.performance_mode == mode


def test_diagnostics_rejects_unknown_mode_and_unbounded_capacity(tmp_path: Path) -> None:
    _copy_config_files(tmp_path)
    path = tmp_path / "diagnostics_settings.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["performance_mode"] = "DEBUG_FOREVER"
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="performance_mode"):
        load_system_config(tmp_path)

    data["performance_mode"] = "LIGHT"
    data["queue_capacity"] = 0
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="queue_capacity"):
        load_system_config(tmp_path)


def test_system_configuration_rejects_missing_and_nan_values(tmp_path: Path) -> None:
    _copy_config_files(tmp_path)

    camera_path = tmp_path / "camera_settings.yaml"
    camera = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    del camera["pixel_format"]
    _write_yaml(camera_path, camera)
    with pytest.raises(ConfigurationError, match="pixel_format"):
        load_system_config(tmp_path)

    camera["pixel_format"] = "Mono8"
    camera["frame_rate_hz"] = float("nan")
    _write_yaml(camera_path, camera)
    with pytest.raises(ConfigurationError, match="finite"):
        load_system_config(tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("parent_frame", "base"), ("child_frame", "marker")],
)
def test_target_rejects_unapproved_frame_names(
    tmp_path: Path, field: str, invalid_value: str
) -> None:
    _copy_config_files(tmp_path)
    target_path = tmp_path / "servo_target.yaml"
    target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    target[field] = invalid_value
    _write_yaml(target_path, target)

    with pytest.raises(ConfigurationError, match=field):
        load_system_config(tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("family", "tag999h999"), ("aruco_dictionary", "DICT_NOT_REAL")],
)
def test_marker_rejects_values_outside_supported_enumerations(
    tmp_path: Path, field: str, invalid_value: str
) -> None:
    _copy_config_files(tmp_path)
    marker_path = tmp_path / "marker_settings.yaml"
    marker = yaml.safe_load(marker_path.read_text(encoding="utf-8"))
    marker[field] = invalid_value
    _write_yaml(marker_path, marker)

    with pytest.raises(ConfigurationError, match=field):
        load_system_config(tmp_path)


@pytest.mark.parametrize("invalid_scale", [0.24, 1.01, float("nan")])
def test_marker_rejects_invalid_detection_scale(
    tmp_path: Path, invalid_scale: float
) -> None:
    _copy_config_files(tmp_path)
    marker_path = tmp_path / "marker_settings.yaml"
    marker = yaml.safe_load(marker_path.read_text(encoding="utf-8"))
    marker["detection_scale"] = invalid_scale
    _write_yaml(marker_path, marker)

    with pytest.raises(ConfigurationError, match="detection_scale|finite"):
        load_system_config(tmp_path)


def test_configuration_rejects_extra_fields(tmp_path: Path) -> None:
    _copy_config_files(tmp_path)
    camera_path = tmp_path / "camera_settings.yaml"
    camera = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    camera["silent_typo"] = 123
    _write_yaml(camera_path, camera)

    with pytest.raises(ConfigurationError, match="silent_typo"):
        load_system_config(tmp_path)


def test_robot_configuration_rejects_any_nonzero_expected_tcp(tmp_path: Path) -> None:
    _copy_config_files(tmp_path)
    robot_path = tmp_path / "robot_connection.yaml"
    robot = yaml.safe_load(robot_path.read_text(encoding="utf-8"))
    robot["expected_tcp_pose_ur"][2] = 1e-12
    _write_yaml(robot_path, robot)

    with pytest.raises(ConfigurationError, match="expected_tcp_pose_ur.*exactly zero"):
        load_system_config(tmp_path)


def test_mpc_solver_time_limit_must_not_exceed_control_step(tmp_path: Path) -> None:
    _copy_config_files(tmp_path)
    mpc_path = tmp_path / "mpc_controller.yaml"
    mpc = yaml.safe_load(mpc_path.read_text(encoding="utf-8"))
    mpc["solver_time_limit_s"] = 0.051
    _write_yaml(mpc_path, mpc)

    with pytest.raises(ConfigurationError, match="solver_time_limit_s.*dt_s"):
        load_system_config(tmp_path)
