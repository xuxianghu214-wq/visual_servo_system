from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visual_servo_system.camera.camera_gentl_capture import (
    CameraCaptureWorker,
    CameraConnectionError,
    CameraFrameError,
    CameraShutdownTimeout,
    GentlCameraCapture,
    convert_component_to_image,
    select_camera_device,
)
from visual_servo_system.camera.spinnaker_cti_discovery import (
    CtiDiscoveryError,
    discover_cti_files,
    require_spinnaker_cti,
)
from visual_servo_system.config.config_loader import CameraSettings
from visual_servo_system.models.data_types import FramePacket


def _settings(
    serial: str | None = None, *, pixel_format: str = "Mono8"
) -> CameraSettings:
    return CameraSettings(
        "serial" if serial else "auto_unique",
        serial,
        "Continuous",
        pixel_format,
        4,
        3,
        30.0,
    )


def test_cti_discovery_preserves_priority_and_deduplicates(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    extra_dir = tmp_path / "extra"
    env_dir.mkdir()
    extra_dir.mkdir()
    first = env_dir / "Spinnaker_GenTL_v140.cti"
    second = extra_dir / "Other.cti"
    first.write_bytes(b"cti")
    second.write_bytes(b"cti")

    result = discover_cti_files(
        extra_paths=[extra_dir, first],
        environ={"GENICAM_GENTL64_PATH": str(env_dir) + os.pathsep + str(first)},
        standard_paths=[],
    )

    assert result == [first.resolve(), second.resolve()]
    assert require_spinnaker_cti(result) == first.resolve()


def test_cti_discovery_gives_readable_error_when_missing(tmp_path: Path) -> None:
    with pytest.raises(CtiDiscoveryError, match="Spinnaker.*CTI"):
        require_spinnaker_cti(
            discover_cti_files(extra_paths=[tmp_path], environ={}, standard_paths=[])
        )


def test_capture_uses_camera_specific_timestamp_uncertainty_limit(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    settings = CameraSettings(
        "auto_unique",
        None,
        "Continuous",
        "BayerRG8",
        1280,
        1024,
        30.0,
        timestamp_max_uncertainty_s=0.020,
    )

    capture = GentlCameraCapture(settings, cti_files=[cti])

    assert capture._clock_mapper.max_uncertainty_s == pytest.approx(0.020)


def test_camera_selection_never_guesses_between_multiple_devices() -> None:
    devices = [SimpleNamespace(serial_number="A"), SimpleNamespace(serial_number="B")]
    with pytest.raises(CameraConnectionError, match="多台"):
        select_camera_device(devices, serial=None)
    assert select_camera_device(devices, serial="B") is devices[1]
    with pytest.raises(CameraConnectionError, match="不存在"):
        select_camera_device(devices, serial="C")
    with pytest.raises(CameraConnectionError, match="未发现"):
        select_camera_device([], serial=None)


def test_camera_selection_reports_spinnaker_incompatible_driver_placeholder() -> None:
    device = SimpleNamespace(
        serial_number="",
        property_dict={
            "serial_number": "",
            "display_name": "DISPLAYNAME_INCOMPATIBLE_DRIVER",
            "id_": r"USB\VID_1E10&PID_4000\00F85CB7_0",
        },
    )

    with pytest.raises(CameraConnectionError, match="驱动不兼容"):
        select_camera_device([device], serial="16276663")


@pytest.mark.parametrize(
    ("pixel_format", "raw", "expected_shape", "first_pixel"),
    [
        ("Mono8", np.arange(12, dtype=np.uint8), (3, 4), 0),
        ("RGB8", np.array([255, 0, 0] * 12, dtype=np.uint8), (3, 4, 3), (0, 0, 255)),
        ("BGR8", np.array([1, 2, 3] * 12, dtype=np.uint8), (3, 4, 3), (1, 2, 3)),
    ],
)
def test_pixel_conversion(pixel_format, raw, expected_shape, first_pixel) -> None:
    component = SimpleNamespace(width=4, height=3, data=raw, data_format=pixel_format)
    image = convert_component_to_image(component)
    assert image.shape == expected_shape
    assert image.flags.c_contiguous
    actual = image[0, 0]
    assert tuple(actual) == first_pixel if image.ndim == 3 else int(actual) == first_pixel


@pytest.mark.parametrize(
    ("pixel_format", "red_positions", "blue_positions"),
    [
        (
            "BayerRG8",
            (slice(0, None, 2), slice(0, None, 2)),
            (slice(1, None, 2), slice(1, None, 2)),
        ),
        (
            "BayerBG8",
            (slice(1, None, 2), slice(1, None, 2)),
            (slice(0, None, 2), slice(0, None, 2)),
        ),
        (
            "BayerGR8",
            (slice(0, None, 2), slice(1, None, 2)),
            (slice(1, None, 2), slice(0, None, 2)),
        ),
        (
            "BayerGB8",
            (slice(1, None, 2), slice(0, None, 2)),
            (slice(0, None, 2), slice(1, None, 2)),
        ),
    ],
)
def test_bayer_conversion_keeps_red_and_blue_channels(pixel_format, red_positions, blue_positions):
    mosaic = np.full((8, 8), 100, dtype=np.uint8)
    mosaic[red_positions] = 220
    mosaic[blue_positions] = 30
    component = SimpleNamespace(
        width=8, height=8, data=mosaic.reshape(-1), data_format=pixel_format
    )
    bgr = convert_component_to_image(component)
    assert tuple(bgr[3, 3]) == (30, 100, 220)


class _Node:
    def __init__(self, value=None, *, reject=False):
        self._value = value
        self.reject = reject

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.reject:
            raise RuntimeError("unsupported")
        self._value = value


class _IgnoringWriteNode(_Node):
    """模拟 GenApi “赋值不报错但相机拒绝切换”的节点。"""

    @_Node.value.setter
    def value(self, _value):
        pass


class _FactoryDefaultCommand:
    """模拟 UserSetLoad(Default)：清除相机当前运行态中的 ISP 2x binning。"""

    def __init__(self, node_map, events: list[str]):
        self.node_map = node_map
        self.events = events

    def execute(self):
        self.events.append("load_default")
        self.node_map.BinningHorizontal._value = 1
        self.node_map.BinningVertical._value = 1
        self.node_map.PixelFormat._value = "BayerRG8"
        self.node_map.Width._value = 4
        self.node_map.Height._value = 3


def _node_map_requiring_factory_default(events: list[str]):
    node_map = _node_map_with_reduced_geometry()
    node_map.BinningHorizontal = _IgnoringWriteNode(2)
    node_map.BinningVertical = _IgnoringWriteNode(2)
    node_map.UserSetSelector = _Node("UserSet1")
    node_map.UserSetLoad = _FactoryDefaultCommand(node_map, events)
    return node_map


class _GeometryConstrainedNode(_Node):
    """模拟真实相机：ROI/binning 未复位时，扩大 Width/Height 会被设备忽略。"""

    def __init__(self, node_map, axis: str, value: int):
        super().__init__(value)
        self.node_map = node_map
        self.axis = axis

    @_Node.value.setter
    def value(self, value):
        suffix = "Horizontal" if self.axis == "x" else "Vertical"
        offset = self.node_map.OffsetX if self.axis == "x" else self.node_map.OffsetY
        dependencies_are_full_frame = (
            getattr(self.node_map, f"Binning{suffix}").value == 1
            and getattr(self.node_map, f"Decimation{suffix}").value == 1
            and offset.value == 0
        )
        if dependencies_are_full_frame:
            self._value = value


def _node_map_with_reduced_geometry():
    node_map = SimpleNamespace(
        AcquisitionMode=_Node(),
        TriggerMode=_Node(),
        PixelFormat=_Node("Mono8"),
        BinningSelector=_Node("Sensor"),
        BinningHorizontal=_Node(2),
        BinningVertical=_Node(2),
        DecimationSelector=_Node("Sensor"),
        DecimationHorizontal=_Node(1),
        DecimationVertical=_Node(1),
        OffsetX=_Node(2),
        OffsetY=_Node(1),
        AcquisitionFrameRateEnable=_Node(),
        AcquisitionFrameRate=_Node(),
    )
    node_map.Width = _GeometryConstrainedNode(node_map, "x", 2)
    node_map.Height = _GeometryConstrainedNode(node_map, "y", 2)
    return node_map


class _FetchedBuffer:
    def __init__(self):
        component = SimpleNamespace(
            width=4,
            height=3,
            data=np.arange(12, dtype=np.uint8),
            data_format="Mono8",
        )
        self.payload = SimpleNamespace(components=[component])
        self.timestamp_ns = None
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.exited = True


class _ImageAcquirer:
    def __init__(
        self,
        *,
        reject_newest=False,
        pixel_format="BayerRG8",
        reject_pixel=True,
    ):
        self.remote_device = SimpleNamespace(
            node_map=SimpleNamespace(
                AcquisitionMode=_Node(),
                TriggerMode=_Node(),
                PixelFormat=_Node(pixel_format, reject=reject_pixel),
                Width=_Node(),
                Height=_Node(),
                AcquisitionFrameRateEnable=_Node(),
                AcquisitionFrameRate=_Node(),
            )
        )
        stream_map = SimpleNamespace(StreamBufferHandlingMode=_Node(reject=reject_newest))
        self.data_streams = [SimpleNamespace(node_map=stream_map)]
        self.buffer = _FetchedBuffer()
        self.started = False
        self.destroyed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def destroy(self):
        self.destroyed = True

    def fetch(self, timeout):
        assert timeout == pytest.approx(0.2)
        return self.buffer


class _Harvester:
    def __init__(
        self,
        *,
        reject_newest=False,
        pixel_format="BayerRG8",
        reject_pixel=True,
    ):
        self.device_info_list = [SimpleNamespace(serial_number="A")]
        self.files = []
        self.ia = _ImageAcquirer(
            reject_newest=reject_newest,
            pixel_format=pixel_format,
            reject_pixel=reject_pixel,
        )
        self.reset_called = False

    def add_file(self, path, check_existence=True, check_validity=True):
        self.files.append(path)

    def update(self):
        pass

    def create(self, selector):
        assert selector in (0, {"serial_number": "A"})
        return self.ia

    def reset(self):
        self.reset_called = True


def test_capture_configures_supported_nodes_and_releases_every_resource(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester()
    capture = GentlCameraCapture(
        _settings(),
        cti_files=[cti],
        harvester_factory=lambda: harvester,
        monotonic_ns=lambda: 1_250_000_000,
    )
    capture.connect()
    packet = capture.fetch_frame(timeout_s=0.2)
    assert packet.host_timestamp_s == pytest.approx(1.25)
    assert packet.receive_timestamp_s == pytest.approx(1.25)
    assert packet.timestamp_source == "host_receive"
    assert not packet.timestamp_reliable
    assert packet.sequence_number == 0
    assert packet.image.shape == (3, 4)
    assert packet.quality < 1.0
    assert capture.pixel_format_fallback
    assert "Mono8" in capture.status_message
    assert harvester.ia.remote_device.node_map.Width.value == 4
    assert harvester.ia.remote_device.node_map.Height.value == 3
    assert harvester.ia.remote_device.node_map.AcquisitionFrameRate.value == 30.0
    assert harvester.ia.remote_device.node_map.AcquisitionMode.value == "Continuous"
    assert harvester.ia.remote_device.node_map.TriggerMode.value == "Off"
    assert harvester.ia.data_streams[0].node_map.StreamBufferHandlingMode.value == "NewestOnly"
    assert harvester.ia.buffer.exited
    capture.close()
    capture.close()
    assert harvester.ia.destroyed
    assert harvester.reset_called


def test_capture_profiles_wait_mapping_conversion_copy_and_packet_build(tmp_path: Path) -> None:
    class Recorder:
        def __init__(self):
            self.records = []

        def record(self, **values):
            self.records.append(values)

    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester()
    recorder = Recorder()
    capture = GentlCameraCapture(
        _settings(),
        cti_files=[cti],
        harvester_factory=lambda: harvester,
        monotonic_ns=lambda: 1_250_000_000,
        performance_profiler=recorder,
    )
    capture.connect()

    capture.fetch_frame()

    stages = {record["stage"] for record in recorder.records}
    assert {
        "fetch_wait",
        "timestamp_mapping",
        "component_array_view",
        "image_copy",
        "packet_construction",
        "fetch_total",
    } <= stages
    assert next(
        record for record in recorder.records if record["stage"] == "fetch_wait"
    )["kind"] == "wait"
    capture.close()


def test_capture_resets_geometry_dependencies_before_requesting_full_resolution(
    tmp_path: Path,
) -> None:
    """相机上次保存了 ROI/binning 时，仍应恢复到配置的完整分辨率。"""

    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester(pixel_format="Mono8", reject_pixel=False)
    node_map = _node_map_with_reduced_geometry()
    harvester.ia.remote_device.node_map = node_map
    capture = GentlCameraCapture(
        _settings(), cti_files=[cti], harvester_factory=lambda: harvester
    )

    capture.connect()

    assert node_map.BinningSelector.value == "All"
    assert node_map.BinningHorizontal.value == 1
    assert node_map.BinningVertical.value == 1
    assert node_map.DecimationSelector.value == "All"
    assert node_map.DecimationHorizontal.value == 1
    assert node_map.DecimationVertical.value == 1
    assert node_map.OffsetX.value == 0
    assert node_map.OffsetY.value == 0
    assert node_map.Width.value == 4
    assert node_map.Height.value == 3
    capture.close()


def test_capture_loads_default_then_uses_configured_bayer_before_full_resolution(
    tmp_path: Path,
) -> None:
    """实机 Mono8 会锁死 ISP 2x binning；连接必须从 Default+BayerRG8 开始。"""

    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    events: list[str] = []
    harvester = _Harvester(pixel_format="Mono8", reject_pixel=False)
    node_map = _node_map_requiring_factory_default(events)
    harvester.ia.remote_device.node_map = node_map
    capture = GentlCameraCapture(
        _settings(pixel_format="BayerRG8"),
        cti_files=[cti],
        harvester_factory=lambda: harvester,
    )

    capture.connect()

    assert events == ["load_default"]
    assert node_map.UserSetSelector.value == "Default"
    assert node_map.PixelFormat.value == "BayerRG8"
    assert node_map.BinningHorizontal.value == 1
    assert node_map.BinningVertical.value == 1
    assert node_map.Width.value == 4
    assert node_map.Height.value == 3
    assert not capture.pixel_format_fallback
    assert "BayerRG8" in capture.status_message
    capture.close()


def test_capture_maps_device_exposure_timestamp_to_host_clock(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester()
    host_receives = iter((11_001_000_000, 12_000_500_000, 13_001_500_000, 14_000_800_000))
    capture = GentlCameraCapture(
        _settings(),
        cti_files=[cti],
        harvester_factory=lambda: harvester,
        monotonic_ns=lambda: next(host_receives),
    )
    capture.connect()
    packets = []
    for device_second in (1, 2, 3, 4):
        harvester.ia.buffer.timestamp_ns = device_second * 1_000_000_000
        packets.append(capture.fetch_frame())
    last = packets[-1]
    assert last.timestamp_source == "camera_device"
    assert last.timestamp_reliable
    assert last.host_timestamp_s == pytest.approx(14.0, abs=0.003)
    assert last.receive_timestamp_s == pytest.approx(14.0008)
    capture.close()


def test_capture_refuses_to_start_without_newest_only(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester(reject_newest=True)
    capture = GentlCameraCapture(
        _settings(), cti_files=[cti], harvester_factory=lambda: harvester
    )
    with pytest.raises(CameraConnectionError, match="NewestOnly"):
        capture.connect()
    assert harvester.ia.destroyed
    assert harvester.reset_called


def test_capture_refuses_unsupported_native_pixel_fallback(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester(pixel_format="YUV422", reject_pixel=True)
    capture = GentlCameraCapture(
        _settings(), cti_files=[cti], harvester_factory=lambda: harvester
    )
    with pytest.raises(CameraConnectionError, match="像素格式"):
        capture.connect()
    assert harvester.ia.destroyed


def test_pixel_format_write_must_be_read_back_before_claiming_mono8(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester(reject_pixel=False)
    harvester.ia.remote_device.node_map.PixelFormat = _IgnoringWriteNode("BayerRG8")
    capture = GentlCameraCapture(
        _settings(), cti_files=[cti], harvester_factory=lambda: harvester
    )
    capture.connect()
    assert capture.pixel_format_fallback
    assert "BayerRG8" in capture.status_message
    capture.close()


def test_frame_resolution_mismatch_is_never_published_for_pnp(tmp_path: Path) -> None:
    cti = tmp_path / "Spinnaker.cti"
    cti.write_bytes(b"x")
    harvester = _Harvester()
    component = harvester.ia.buffer.payload.components[0]
    component.width = 5
    component.data = np.arange(15, dtype=np.uint8)
    capture = GentlCameraCapture(
        _settings(), cti_files=[cti], harvester_factory=lambda: harvester
    )
    capture.connect()
    with pytest.raises(CameraFrameError, match="分辨率"):
        capture.fetch_frame()
    capture.close()


def test_capture_worker_keeps_only_newest_frame() -> None:
    class BurstCapture:
        def __init__(self):
            self.index = 0
            self.closed = False

        def connect(self):
            pass

        def fetch_frame(self, _timeout):
            if self.index == 3:
                raise RuntimeError("burst complete")
            packet = FramePacket(
                float(self.index), self.index, np.full((2, 2), self.index, np.uint8), True, 1.0
            )
            self.index += 1
            return packet

        def close(self):
            self.closed = True

    capture = BurstCapture()
    worker = CameraCaptureWorker(capture)  # type: ignore[arg-type]
    worker.start()
    deadline = time.monotonic() + 1.0
    while worker.last_error is None and time.monotonic() < deadline:
        time.sleep(0.001)
    latest = worker.latest_frame.get_latest()
    assert latest is not None
    assert latest[1].sequence_number == 2
    assert np.all(latest[1].image == 2)
    worker.stop()
    assert capture.closed


def test_capture_worker_does_not_close_resource_while_fetch_is_blocked() -> None:
    class BlockingCapture:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.close_thread_ids: list[int] = []

        def connect(self):
            pass

        def fetch_frame(self, _timeout):
            self.entered.set()
            self.release.wait()
            raise RuntimeError("released")

        def close(self):
            self.close_thread_ids.append(threading.get_ident())

    capture = BlockingCapture()
    worker = CameraCaptureWorker(capture)  # type: ignore[arg-type]
    worker.start()
    assert capture.entered.wait(1.0)
    caller_thread = threading.get_ident()
    with pytest.raises(CameraShutdownTimeout, match="仍在运行"):
        worker.stop(timeout_s=0.01)
    assert capture.close_thread_ids == []
    capture.release.set()
    worker.stop(timeout_s=1.0)
    assert capture.close_thread_ids
    assert all(thread_id != caller_thread for thread_id in capture.close_thread_ids)

