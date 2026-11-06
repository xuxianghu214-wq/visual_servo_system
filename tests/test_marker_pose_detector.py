from __future__ import annotations

import cv2
import numpy as np
import pytest

from visual_servo_system.vision.marker_pose_detector import (
    MarkerPoseDetector,
    MarkerPoseDetectorConfig,
    dictionary_name_for_detector,
)


def _intrinsics() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
    )


def _synthetic_marker(dictionary_name: str, marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    canvas[160:320, 240:400] = marker
    return canvas


def test_dictionary_mapping_supports_apriltag_and_aruco() -> None:
    assert (
        dictionary_name_for_detector("apriltag", "tag36h11", "DICT_4X4_50")
        == "DICT_APRILTAG_36h11"
    )
    assert dictionary_name_for_detector("aruco", "tag36h11", "DICT_4X4_50") == "DICT_4X4_50"
    with pytest.raises(ValueError, match="不支持"):
        dictionary_name_for_detector("apriltag", "bad", "DICT_4X4_50")


def test_detector_config_uses_explicit_whitelists_even_if_opencv_has_attribute(monkeypatch) -> None:
    monkeypatch.setattr(cv2.aruco, "DICT_NOT_PROJECT_APPROVED", 999, raising=False)
    with pytest.raises(ValueError, match="不支持"):
        MarkerPoseDetectorConfig(
            "aruco", "tag36h11", "DICT_NOT_PROJECT_APPROVED", 0, 0.08
        )
    with pytest.raises(ValueError, match="不支持"):
        MarkerPoseDetectorConfig("apriltag", "bad-family", "DICT_4X4_50", 0, 0.08)
    with pytest.raises(ValueError, match="不支持"):
        MarkerPoseDetectorConfig(
            "apriltag", "tag36h11", "DICT_NOT_PROJECT_APPROVED", 0, 0.08
        )


def test_fixed_id_detection_and_pose_on_synthetic_apriltag() -> None:
    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig("apriltag", "tag36h11", "DICT_4X4_50", 0, 0.08),
        camera_matrix,
        distortion,
    )
    observation = detector.detect(_synthetic_marker("DICT_APRILTAG_36h11", 0), 3.0)
    assert observation.valid
    assert observation.marker_id == 0
    assert observation.T_camera_marker[2, 3] > 0.0
    assert observation.reprojection_error_px < 2.0


def test_half_resolution_detection_maps_corners_back_to_full_image() -> None:
    """缩小图只用于快速搜索；PnP 仍必须使用原图像素坐标。"""

    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig(
            "apriltag",
            "tag36h11",
            "DICT_4X4_50",
            0,
            0.08,
            detection_scale=0.5,
        ),
        camera_matrix,
        distortion,
    )

    class FakeScaledDetector:
        input_shape = None

        def detectMarkers(self, image):
            self.input_shape = image.shape
            corners = np.array(
                [[[120.0, 80.0], [200.0, 80.0], [200.0, 160.0], [120.0, 160.0]]],
                dtype=np.float32,
            )
            return [corners], np.array([[0]], dtype=np.int32), []

    fake = FakeScaledDetector()
    detector._detector = fake
    transform = np.eye(4)
    transform[2, 3] = 0.5
    detector._solve_pose = lambda _corners: (transform, 0.1)

    observation = detector.detect(np.full((480, 640), 127, np.uint8), 3.0)

    assert observation.valid
    assert fake.input_shape == (240, 320)
    np.testing.assert_allclose(
        detector.last_corners,
        [[240.0, 160.0], [400.0, 160.0], [400.0, 320.0], [240.0, 320.0]],
        atol=1e-5,
    )


def test_detector_does_not_switch_to_wrong_visible_id() -> None:
    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig("aruco", "tag36h11", "DICT_4X4_50", 0, 0.08),
        camera_matrix,
        distortion,
    )
    observation = detector.detect(_synthetic_marker("DICT_4X4_50", 1), 4.0)
    assert not observation.valid
    assert observation.marker_id == 0
    assert observation.quality == 0.0


def test_short_edge_quality_gate_rejects_tiny_marker() -> None:
    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig("aruco", "tag36h11", "DICT_4X4_50", 0, 0.08, min_side_px=40.0),
        camera_matrix,
        distortion,
    )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 30)
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[220:250, 305:335] = marker
    assert not detector.detect(image, 5.0).valid


def test_continuity_gate_rejects_large_pose_jump(monkeypatch) -> None:
    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig(
            "aruco", "tag36h11", "DICT_4X4_50", 0, 0.08, max_position_jump_m=0.01
        ),
        camera_matrix,
        distortion,
    )
    image = _synthetic_marker("DICT_4X4_50", 0)
    assert detector.detect(image, 1.0).valid
    original = cv2.solvePnPGeneric

    def jumped(*args, **kwargs):
        retval, rvecs, tvecs, errors = original(*args, **kwargs)
        shifted = tuple(np.asarray(value) + np.array([[1.0], [0.0], [0.0]]) for value in tvecs)
        return retval, rvecs, shifted, errors

    monkeypatch.setattr(cv2, "solvePnPGeneric", jumped)
    assert not detector.detect(image, 1.05).valid


def test_missing_frames_clear_continuity_anchor_and_allow_reacquisition(monkeypatch) -> None:
    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig(
            "aruco",
            "tag36h11",
            "DICT_4X4_50",
            0,
            0.08,
            max_position_jump_m=0.01,
            reacquire_after_missing_frames=2,
            reacquire_after_missing_s=10.0,
        ),
        camera_matrix,
        distortion,
    )
    image = _synthetic_marker("DICT_4X4_50", 0)
    assert detector.detect(image, 1.0).valid
    blank = np.full_like(image, 255)
    assert not detector.detect(blank, 1.05).valid
    assert not detector.detect(blank, 1.10).valid
    original = cv2.solvePnPGeneric

    def moved(*args, **kwargs):
        retval, rvecs, tvecs, errors = original(*args, **kwargs)
        shifted = tuple(np.asarray(value) + np.array([[1.0], [0.0], [0.0]]) for value in tvecs)
        return retval, rvecs, shifted, errors

    monkeypatch.setattr(cv2, "solvePnPGeneric", moved)
    # 本用例只隔离验证“连续性锚点”；人为平移 tvec 会必然改变重投影，
    # 因此固定合法重投影误差，避免误把重投影质量门当成重获失败。
    monkeypatch.setattr(detector, "_reprojection_error", lambda *_args: 0.1)
    assert detector.detect(image, 1.15).valid


def test_single_jump_stays_rejected_and_explicit_reset_allows_reacquire(monkeypatch) -> None:
    camera_matrix, distortion = _intrinsics()
    detector = MarkerPoseDetector(
        MarkerPoseDetectorConfig(
            "aruco",
            "tag36h11",
            "DICT_4X4_50",
            0,
            0.08,
            max_position_jump_m=0.01,
            reacquire_after_missing_frames=3,
        ),
        camera_matrix,
        distortion,
    )
    image = _synthetic_marker("DICT_4X4_50", 0)
    assert detector.detect(image, 1.0).valid
    original = cv2.solvePnPGeneric

    def moved(*args, **kwargs):
        retval, rvecs, tvecs, errors = original(*args, **kwargs)
        shifted = tuple(np.asarray(value) + np.array([[1.0], [0.0], [0.0]]) for value in tvecs)
        return retval, rvecs, shifted, errors

    monkeypatch.setattr(cv2, "solvePnPGeneric", moved)
    monkeypatch.setattr(detector, "_reprojection_error", lambda *_args: 0.1)
    assert not detector.detect(image, 1.05).valid
    detector.reset_tracking()
    assert detector.detect(image, 1.10).valid

