"""固定 ID AprilTag/ArUco 检测、方形 PnP 和观测质量门。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from visual_servo_system.geometry.pose_transforms import validate_transform
from visual_servo_system.models.data_types import MarkerPoseObservation

_APRILTAG_DICTIONARIES = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}
_ARUCO_DICTIONARIES = frozenset(
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


def dictionary_name_for_detector(detector: str, family: str, aruco_dictionary: str) -> str:
    """把配置名称映射到 OpenCV ``aruco`` 常量名。"""

    if detector == "apriltag":
        try:
            return _APRILTAG_DICTIONARIES[family]
        except KeyError as error:
            raise ValueError(f"不支持的 AprilTag family：{family}") from error
    if (
        detector == "aruco"
        and aruco_dictionary in _ARUCO_DICTIONARIES
        and hasattr(cv2.aruco, aruco_dictionary)
    ):
        return aruco_dictionary
    raise ValueError(f"不支持的 ArUco 字典或检测器：{aruco_dictionary}")


@dataclass(frozen=True, slots=True)
class MarkerPoseDetectorConfig:
    detector: str
    family: str
    aruco_dictionary: str
    marker_id: int
    marker_size_m: float
    detection_scale: float = 0.5
    max_reprojection_error_px: float = 2.0
    min_side_px: float = 40.0
    max_position_jump_m: float = 0.15
    max_rotation_jump_rad: float = np.deg2rad(45.0)
    reacquire_after_missing_frames: int = 5
    reacquire_after_missing_s: float = 0.25

    def __post_init__(self) -> None:
        if self.detector not in {"apriltag", "aruco"}:
            raise ValueError(f"不支持的检测器：{self.detector}")
        if self.family not in _APRILTAG_DICTIONARIES:
            raise ValueError(f"不支持的 AprilTag family：{self.family}")
        if self.aruco_dictionary not in _ARUCO_DICTIONARIES:
            raise ValueError(f"不支持的 ArUco 字典：{self.aruco_dictionary}")
        dictionary_name_for_detector(self.detector, self.family, self.aruco_dictionary)
        if type(self.marker_id) is not int or self.marker_id < 0:
            raise ValueError("marker_id 必须为非负整数")
        if (
            type(self.reacquire_after_missing_frames) is not int
            or self.reacquire_after_missing_frames <= 0
        ):
            raise ValueError("reacquire_after_missing_frames 必须为正整数")
        if not np.isfinite(self.marker_size_m) or self.marker_size_m <= 0.0:
            raise ValueError("marker_id 必须非负，marker_size_m 必须为正")
        if not np.isfinite(self.detection_scale) or not 0.25 <= self.detection_scale <= 1.0:
            raise ValueError("detection_scale 必须位于 [0.25, 1.0]")
        limits = (
            self.max_reprojection_error_px,
            self.min_side_px,
            self.max_position_jump_m,
            self.max_rotation_jump_rad,
            self.reacquire_after_missing_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in limits):
            raise ValueError("检测质量阈值必须为有限正数")


def _object_points(marker_size_m: float) -> np.ndarray:
    """IPPE_SQUARE 顺序：左上、右上、右下、左下；marker +Y 向上、+Z 朝观察者。"""

    half = marker_size_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def _invalid_observation(timestamp_s: float, marker_id: int) -> MarkerPoseObservation:
    return MarkerPoseObservation(
        float(timestamp_s), marker_id, np.eye(4), 0.0, None, 0.0, False, 0.0
    )


class MarkerPoseDetector:
    """只追踪配置中的固定 ID，绝不在多个标志之间自动切换。"""

    def __init__(
        self,
        config: MarkerPoseDetectorConfig,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
    ) -> None:
        self.config = config
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
        if self.camera_matrix.shape != (3, 3) or not np.all(np.isfinite(self.camera_matrix)):
            raise ValueError("camera_matrix 必须为有限 3x3 矩阵")
        if not np.all(np.isfinite(self.distortion)):
            raise ValueError("distortion_coefficients 必须有限")
        dictionary_name = dictionary_name_for_detector(
            config.detector, config.family, config.aruco_dictionary
        )
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._object_points = _object_points(config.marker_size_m)
        self._last_transform: np.ndarray | None = None
        self._last_valid_timestamp_s: float | None = None
        self._consecutive_missing = 0
        self.last_corners: np.ndarray | None = None

    def reset_tracking(self) -> None:
        """清除连续性锚点；重新使能或场景切换时由上层显式调用。"""

        self._last_transform = None
        self._last_valid_timestamp_s = None
        self._consecutive_missing = 0
        self.last_corners = None

    def _reset_continuity_anchor(self) -> None:
        self._last_transform = None
        self._last_valid_timestamp_s = None
        self._consecutive_missing = 0

    def _prepare_reacquisition(self, timestamp_s: float) -> None:
        if (
            self._last_valid_timestamp_s is not None
            and timestamp_s - self._last_valid_timestamp_s
            >= self.config.reacquire_after_missing_s
        ):
            self._reset_continuity_anchor()

    def _missing_observation(
        self, timestamp_s: float, corners: np.ndarray | None = None
    ) -> MarkerPoseObservation:
        self.last_corners = corners
        self._consecutive_missing += 1
        if self._consecutive_missing >= self.config.reacquire_after_missing_frames:
            self._reset_continuity_anchor()
        return _invalid_observation(timestamp_s, self.config.marker_id)

    def _candidate_transform(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        return validate_transform(transform)

    def _reprojection_error(self, corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> float:
        projected, _ = cv2.projectPoints(
            self._object_points, rvec, tvec, self.camera_matrix, self.distortion
        )
        return float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))

    def _continuity(self, transform: np.ndarray) -> tuple[float, float]:
        if self._last_transform is None:
            return 0.0, 0.0
        position_jump = float(
            np.linalg.norm(transform[:3, 3] - self._last_transform[:3, 3])
        )
        relative = self._last_transform[:3, :3].T @ transform[:3, :3]
        rotation_jump = float(Rotation.from_matrix(relative).magnitude())
        return position_jump, rotation_jump

    def _solve_pose(self, corners: np.ndarray) -> tuple[np.ndarray, float] | None:
        result = cv2.solvePnPGeneric(
            self._object_points,
            corners.reshape(4, 1, 2),
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if len(result) < 3 or not result[0]:
            return None
        rvecs, tvecs = result[1], result[2]
        candidates: list[tuple[float, np.ndarray, float]] = []
        for rvec, tvec in zip(rvecs, tvecs, strict=True):
            transform = self._candidate_transform(rvec, tvec)
            camera_points = (transform[:3, :3] @ self._object_points.T).T + transform[:3, 3]
            if np.any(camera_points[:, 2] <= 0.0):
                continue
            error = self._reprojection_error(corners, rvec, tvec)
            position_jump, rotation_jump = self._continuity(transform)
            if self._last_transform is not None and (
                position_jump > self.config.max_position_jump_m
                or rotation_jump > self.config.max_rotation_jump_rad
            ):
                continue
            score = error + 4.0 * position_jump + 0.2 * rotation_jump
            candidates.append((score, transform, error))
        if not candidates:
            return None
        _, transform, error = min(candidates, key=lambda item: item[0])
        return transform, error

    def detect(self, image: np.ndarray, host_timestamp_s: float) -> MarkerPoseObservation:
        timestamp_s = float(host_timestamp_s)
        self._prepare_reacquisition(timestamp_s)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("image 必须是灰度或 BGR 图像")
        scale = self.config.detection_scale
        if scale < 1.0:
            detection_image = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = gray
        corners_list, ids, _rejected = self._detector.detectMarkers(detection_image)
        if ids is None:
            return self._missing_observation(timestamp_s)
        matching = np.flatnonzero(ids.reshape(-1) == self.config.marker_id)
        if matching.size != 1:
            return self._missing_observation(timestamp_s)
        corners = np.asarray(corners_list[int(matching[0])], dtype=np.float64).reshape(4, 2)
        if scale < 1.0:
            # 搜索可以用小图加速，但相机内参和 PnP 都属于原分辨率。
            # 因此先恢复原图坐标，再在原灰度图上做一次亚像素精修。
            corners /= scale
            refined = corners.astype(np.float32).reshape(-1, 1, 2)
            height, width = gray.shape
            refinement_margin = 6.0
            inside = bool(
                np.all(refined[:, 0, 0] >= refinement_margin)
                and np.all(refined[:, 0, 0] < width - refinement_margin)
                and np.all(refined[:, 0, 1] >= refinement_margin)
                and np.all(refined[:, 0, 1] < height - refinement_margin)
            )
            if inside:
                cv2.cornerSubPix(
                    gray,
                    refined,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
                )
                corners = refined.reshape(4, 2).astype(np.float64)
        side_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        area = abs(float(cv2.contourArea(corners.astype(np.float32))))
        if float(np.min(side_lengths)) < self.config.min_side_px or area <= 0.0:
            return self._missing_observation(timestamp_s, corners)
        solved = self._solve_pose(corners)
        if solved is None:
            return self._missing_observation(timestamp_s, corners)
        transform, error = solved
        if not np.isfinite(error) or error > self.config.max_reprojection_error_px:
            return self._missing_observation(timestamp_s, corners)
        self._last_transform = transform
        self._last_valid_timestamp_s = timestamp_s
        self._consecutive_missing = 0
        self.last_corners = corners
        quality = float(
            np.clip(1.0 - error / self.config.max_reprojection_error_px, 0.0, 1.0)
        )
        # OpenCV ArUco 不提供 decision margin；None 与像素平方单位的面积分别保存。
        return MarkerPoseObservation(
            timestamp_s,
            self.config.marker_id,
            transform,
            float(error),
            None,
            float(area),
            True,
            quality,
        )

    def draw_detection(self, image: np.ndarray, observation: MarkerPoseObservation) -> np.ndarray:
        """返回带角点和坐标轴的副本，不修改采集线程持有的原图。"""

        output = np.array(image, copy=True)
        if self.last_corners is not None:
            corners = self.last_corners.reshape(1, 4, 2).astype(np.float32)
            cv2.polylines(output, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        if observation.valid:
            rvec, _ = cv2.Rodrigues(observation.T_camera_marker[:3, :3])
            cv2.drawFrameAxes(
                output,
                self.camera_matrix,
                self.distortion,
                rvec,
                observation.T_camera_marker[:3, 3],
                self.config.marker_size_m * 0.5,
            )
        return output

