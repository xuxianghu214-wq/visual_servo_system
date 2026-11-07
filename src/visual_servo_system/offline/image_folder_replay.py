"""保留原始帧身份和采集时间的离线图像回放。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visual_servo_system.models.data_types import FramePacket


@dataclass(frozen=True, slots=True)
class ImageReplayConfig:
    fps: float = 30.0
    start_host_s: float = 10.0
    fixed_delay_s: float = 0.0
    jitter_s: float = 0.0
    random_drop_probability: float = 0.0
    drop_indices: tuple[int, ...] = ()
    duplicate_indices: tuple[int, ...] = ()
    reorder_pairs: tuple[tuple[int, int], ...] = ()
    random_seed: int = 20261107

    def __post_init__(self) -> None:
        values = (self.fps, self.start_host_s, self.fixed_delay_s, self.jitter_s)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("image replay timing values must be finite")
        if self.fps <= 0.0 or self.start_host_s < 0.0 or self.fixed_delay_s < 0.0:
            raise ValueError("fps must be positive and replay times must be non-negative")
        if self.jitter_s < 0.0:
            raise ValueError("jitter_s must be non-negative")
        if not 0.0 <= self.random_drop_probability <= 1.0:
            raise ValueError("random_drop_probability must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ImageReplayEvent:
    frame: FramePacket
    delivery_order: int
    simulated: bool = True


def load_image_folder(directory: str | Path) -> list[np.ndarray]:
    """按文件名排序读取回放图像，不做缩放或伪造时间戳。"""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"image replay directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"image replay directory contains no supported images: {root}")
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot decode replay image: {path}")
        images.append(image)
    return images


def replay_images(
    images: Iterable[np.ndarray], config: ImageReplayConfig | None = None
) -> Iterator[ImageReplayEvent]:
    """生成可重复的模拟接收序列；重复帧保留相同帧号和采集时刻。"""

    config = ImageReplayConfig() if config is None else config
    rng = np.random.default_rng(config.random_seed)
    drop_indices = set(config.drop_indices)
    duplicate_indices = set(config.duplicate_indices)
    specs: list[tuple[int, np.ndarray, bool]] = []
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("replay images must be non-empty NumPy arrays")
        if index in drop_indices or rng.random() < config.random_drop_probability:
            continue
        specs.append((index, image, False))
        if index in duplicate_indices:
            specs.append((index, image, True))

    for left_identity, right_identity in config.reorder_pairs:
        left = next(
            (position for position, item in enumerate(specs) if item[0] == left_identity),
            None,
        )
        right = next(
            (position for position, item in enumerate(specs) if item[0] == right_identity),
            None,
        )
        if left is not None and right is not None:
            specs[left], specs[right] = specs[right], specs[left]

    period_s = 1.0 / config.fps
    for delivery_order, (identity, image, _duplicate) in enumerate(specs):
        capture_s = config.start_host_s + identity * period_s
        jitter = float(rng.uniform(-config.jitter_s, config.jitter_s))
        nominal_receive = config.start_host_s + delivery_order * period_s
        receive_s = max(capture_s, nominal_receive + config.fixed_delay_s + jitter)
        frame = FramePacket(
            float(capture_s),
            int(identity),
            image,
            True,
            1.0,
            receive_timestamp_s=float(receive_s),
            timestamp_source="offline_simulated_camera",
            timestamp_reliable=True,
            timestamp_uncertainty_s=0.0,
        )
        yield ImageReplayEvent(frame, delivery_order, simulated=True)


def replay_image_folder(
    directory: str | Path, config: ImageReplayConfig | None = None
) -> Iterator[ImageReplayEvent]:
    return replay_images(load_image_folder(directory), config)
