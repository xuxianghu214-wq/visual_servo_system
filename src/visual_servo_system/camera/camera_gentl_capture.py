"""通过 Harvesters/GenTL 采集 Point Grey/Teledyne 相机图像。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visual_servo_system.camera.spinnaker_cti_discovery import (
    discover_cti_files,
    require_spinnaker_cti,
)
from visual_servo_system.config.config_loader import CameraSettings
from visual_servo_system.models.data_types import FramePacket
from visual_servo_system.runtime.device_clock_mapper import DeviceClockMapper
from visual_servo_system.runtime.latest_value_buffer import LatestValueBuffer


class CameraConnectionError(RuntimeError):
    """相机发现、选择或初始化失败。"""


class CameraFrameError(RuntimeError):
    """一帧数据缺失或像素格式不受支持。"""


class CameraShutdownTimeout(RuntimeError):
    """取帧线程仍在底层 C 驱动中运行，资源不能安全释放。"""


_SUPPORTED_PIXEL_FORMATS = frozenset(
    {
        "Mono8",
        "Mono8Packed",
        "RGB8",
        "RGB8Packed",
        "BGR8",
        "BGR8Packed",
        "BayerRG8",
        "BayerGR8",
        "BayerGB8",
        "BayerBG8",
    }
)


def _device_serial(device: Any) -> str:
    for name in ("serial_number", "serial", "id_"):
        value = getattr(device, name, None)
        if value is not None:
            return str(value)
    property_dict = getattr(device, "property_dict", None)
    if isinstance(property_dict, dict):
        for name in ("serial_number", "serial"):
            if name in property_dict:
                return str(property_dict[name])
    return ""


def _uses_incompatible_driver_placeholder(device: Any) -> bool:
    """识别 Spinnaker 在 USB 驱动不兼容时返回的占位设备。"""

    property_dict = getattr(device, "property_dict", None)
    if not isinstance(property_dict, dict):
        return False
    display_name = str(property_dict.get("display_name", ""))
    return "INCOMPATIBLE_DRIVER" in display_name.upper()


def _raise_incompatible_driver_error(devices: Sequence[Any]) -> None:
    incompatible = [device for device in devices if _uses_incompatible_driver_placeholder(device)]
    if not incompatible:
        return
    identifiers = [
        str(getattr(device, "property_dict", {}).get("id_", "未知 USB 设备"))
        for device in incompatible
    ]
    raise CameraConnectionError(
        "Spinnaker 已发现相机，但 Windows USB 驱动不兼容"
        f"（{identifiers}）。请关闭相机程序，运行工程根目录的“安装相机驱动.ps1”，"
        "完成后重新插拔相机；不要通过修改图像分辨率绕过此错误。"
    )


def select_camera_device(devices: Sequence[Any], serial: str | None) -> Any:
    """选择设备；多相机且未指定序列号时故意失败，绝不随机连接。"""

    if not devices:
        raise CameraConnectionError("未发现任何 GenTL 相机，请检查 USB、驱动和 CTI 路径")
    if serial is not None:
        matches = [device for device in devices if _device_serial(device) == serial]
        if len(matches) != 1:
            _raise_incompatible_driver_error(devices)
            found = [value for device in devices if (value := _device_serial(device))]
            raise CameraConnectionError(
                f"配置的相机序列号 {serial!r} 不存在；当前发现：{found or '无法读取序列号'}"
            )
        return matches[0]
    if len(devices) != 1:
        serials = [_device_serial(device) or "未知" for device in devices]
        raise CameraConnectionError(
            f"发现多台相机 {serials}，必须在 camera_settings.yaml 配置 device_serial"
        )
    _raise_incompatible_driver_error(devices)
    return devices[0]


def _format_name(component: Any) -> str:
    raw = getattr(component, "data_format", "")
    return str(getattr(raw, "name", raw)).split(".")[-1]


def convert_component_to_image(component: Any) -> np.ndarray:
    """把 GenICam component 复制为连续 Mono8 或 OpenCV BGR 图像。"""

    width = int(component.width)
    height = int(component.height)
    if width <= 0 or height <= 0:
        raise CameraFrameError("相机返回了无效图像尺寸")
    raw = np.asarray(component.data, dtype=np.uint8)
    name = _format_name(component)
    if name in {"Mono8", "Mono8Packed"}:
        image = raw[: width * height].reshape(height, width)
    elif name in {"RGB8", "RGB8Packed"}:
        rgb = raw[: width * height * 3].reshape(height, width, 3)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif name in {"BGR8", "BGR8Packed"}:
        image = raw[: width * height * 3].reshape(height, width, 3)
    elif name in {"BayerRG8", "BayerGR8", "BayerGB8", "BayerBG8"}:
        bayer = raw[: width * height].reshape(height, width)
        # OpenCV 的 Bayer 常量以“输出首/末通道位置”命名，和 GenICam 像素位置名相反。
        # 因此 GenICam RG 必须使用 OpenCV BG 常量，GR/GB 同理；非对称彩色测试锁定此约定。
        conversion = {
            "BayerRG8": cv2.COLOR_BayerBG2BGR,
            "BayerGR8": cv2.COLOR_BayerGB2BGR,
            "BayerGB8": cv2.COLOR_BayerGR2BGR,
            "BayerBG8": cv2.COLOR_BayerRG2BGR,
        }[name]
        image = cv2.cvtColor(bayer, conversion)
    else:
        raise CameraFrameError(f"不支持的相机像素格式：{name or '未知'}")
    return np.ascontiguousarray(image).copy()


def _set_node_if_supported(node_map: Any, name: str, value: str) -> bool:
    """可选节点不存在/不可写时返回 False；关键节点由调用端决定是否失败。"""

    try:
        node = getattr(node_map, name, None)
    except Exception:
        # GenApi 的 NodeMap 对不存在的节点可能抛 AccessException，而不是返回 None。
        return False
    if node is None:
        return False
    try:
        node.value = value
    except Exception:  # GenApi 的具体异常类型随驱动版本变化。
        return False
    return True


def _configure_and_verify_node(
    node_map: Any,
    name: str,
    expected: int | float | bool | str,
    *,
    tolerance: float = 0.0,
) -> None:
    """节点存在时尝试写入并强制回读；节点缺失则留给逐帧校验兜底。"""

    try:
        node = getattr(node_map, name, None)
    except Exception:
        # 这些是跨相机型号的可选 GenICam 节点；缺失时由最终分辨率校验兜底。
        return
    if node is None:
        return
    try:
        node.value = expected
    except Exception:
        pass
    try:
        actual = node.value
        if isinstance(expected, bool):
            matches = bool(actual) is expected
        elif isinstance(expected, str):
            actual_name = str(getattr(actual, "name", actual)).split(".")[-1]
            matches = actual_name == expected
        else:
            matches = bool(
                np.isclose(float(actual), float(expected), atol=tolerance, rtol=0.0)
            )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise CameraConnectionError(
            f"相机节点 {name} 回读值 {getattr(node, 'value', None)!r} 与配置 {expected!r} 不符"
        )


def _configure_full_frame_geometry(
    node_map: Any, *, width_px: int, height_px: int
) -> None:
    """清除相机保存的 ROI/binning/decimation，再设置配置要求的图像尺寸。

    Width/Height 的可写上限依赖 Offset、binning 和 decimation。若直接扩大 Width，
    部分相机会静默保留旧值（例如二倍 binning 下仍为 640），所以顺序不能颠倒。
    """

    _configure_and_verify_node(node_map, "BinningSelector", "All")
    _configure_and_verify_node(node_map, "BinningHorizontal", 1)
    _configure_and_verify_node(node_map, "BinningVertical", 1)
    _configure_and_verify_node(node_map, "DecimationSelector", "All")
    _configure_and_verify_node(node_map, "DecimationHorizontal", 1)
    _configure_and_verify_node(node_map, "DecimationVertical", 1)
    _configure_and_verify_node(node_map, "OffsetX", 0)
    _configure_and_verify_node(node_map, "OffsetY", 0)
    _configure_and_verify_node(node_map, "Width", width_px)
    _configure_and_verify_node(node_map, "Height", height_px)


def _load_factory_default_user_set(node_map: Any) -> bool:
    """加载厂家 Default，清除上次连接残留的 ISP binning/ROI 运行状态。"""

    try:
        selector = getattr(node_map, "UserSetSelector", None)
        load_command = getattr(node_map, "UserSetLoad", None)
    except Exception:
        return False
    if selector is None or load_command is None:
        return False
    try:
        selector.value = "Default"
        selected = str(getattr(selector.value, "name", selector.value)).split(".")[-1]
        if selected != "Default":
            raise CameraConnectionError(
                f"相机 UserSetSelector 回读为 {selected!r}，无法选择厂家 Default"
            )
        load_command.execute()
    except CameraConnectionError:
        raise
    except Exception as error:
        raise CameraConnectionError(f"加载相机厂家 UserSet Default 失败：{error}") from error
    return True


class GentlCameraCapture:
    """单相机同步采集器；线程策略由 ``CameraCaptureWorker`` 单独负责。"""

    def __init__(
        self,
        settings: CameraSettings,
        *,
        cti_files: Iterable[str | Path] = (),
        harvester_factory: Callable[[], Any] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        clock_mapper: DeviceClockMapper | None = None,
    ) -> None:
        self.settings = settings
        found = discover_cti_files(cti_files)
        self._cti_files = [require_spinnaker_cti(found)]
        self._harvester_factory = harvester_factory or self._default_harvester_factory
        self._clock_ns = monotonic_ns
        self._clock_mapper = clock_mapper or DeviceClockMapper(
            max_uncertainty_s=settings.timestamp_max_uncertainty_s
        )
        self._harvester: Any | None = None
        self._acquirer: Any | None = None
        self._sequence = 0
        self._closed = True
        self.pixel_format_fallback = False
        self.status_message = "尚未连接"

    @staticmethod
    def _default_harvester_factory() -> Any:
        from harvesters.core import Harvester

        return Harvester()

    @property
    def connected(self) -> bool:
        return self._acquirer is not None and not self._closed

    def connect(self) -> None:
        if self.connected:
            return
        harvester = self._harvester_factory()
        acquirer: Any | None = None
        try:
            for path in self._cti_files:
                harvester.add_file(str(path), check_existence=True, check_validity=True)
            harvester.update()
            devices = list(harvester.device_info_list)
            selected = select_camera_device(devices, self.settings.device_serial)
            if self.settings.device_serial is not None:
                selector: Any = {"serial_number": self.settings.device_serial}
            else:
                selector = devices.index(selected)
            acquirer = harvester.create(selector)
            node_map = acquirer.remote_device.node_map
            _load_factory_default_user_set(node_map)
            if not _set_node_if_supported(node_map, "AcquisitionMode", "Continuous"):
                raise CameraConnectionError("相机不支持连续采集 AcquisitionMode=Continuous")
            if not _set_node_if_supported(node_map, "TriggerMode", "Off"):
                raise CameraConnectionError("无法关闭相机触发模式 TriggerMode=Off")

            # 必须先确定像素格式，再配置几何节点。实机 BFS-U3-13Y3C 若选择 Mono8，
            # 会自动固定 ISP 2x binning，使 WidthMax/HeightMax 降为 640x512。
            requested_format = self.settings.pixel_format
            format_write_ok = _set_node_if_supported(
                node_map, "PixelFormat", requested_format
            )
            try:
                pixel_format_node = node_map.PixelFormat
                current = pixel_format_node.value
            except Exception as error:
                raise CameraConnectionError(
                    "无法回读 PixelFormat，不能确认后续像素转换安全"
                ) from error
            current_name = str(getattr(current, "name", current)).split(".")[-1]
            if current_name not in _SUPPORTED_PIXEL_FORMATS:
                raise CameraConnectionError(
                    f"像素格式 {requested_format} 设置未生效，设备当前格式 "
                    f"{current_name or '未知'} 不受支持"
                )
            self.pixel_format_fallback = current_name != requested_format
            if self.pixel_format_fallback:
                write_state = "设置未生效" if format_write_ok else "设置失败"
                self.status_message = (
                    f"{requested_format} {write_state}，显式降级为 {current_name}"
                )
            else:
                self.status_message = requested_format

            _configure_full_frame_geometry(
                node_map,
                width_px=self.settings.image_width_px,
                height_px=self.settings.image_height_px,
            )
            _configure_and_verify_node(node_map, "AcquisitionFrameRateEnable", True)
            _configure_and_verify_node(
                node_map,
                "AcquisitionFrameRate",
                self.settings.frame_rate_hz,
                tolerance=1e-3,
            )
            streams = tuple(getattr(acquirer, "data_streams", ()))
            if not streams or not all(
                _set_node_if_supported(
                    stream.node_map, "StreamBufferHandlingMode", "NewestOnly"
                )
                for stream in streams
            ):
                raise CameraConnectionError(
                    "无法配置 StreamBufferHandlingMode=NewestOnly；为防止延迟堆积，拒绝启动"
                )
            acquirer.start()
        except Exception as error:
            if acquirer is not None:
                try:
                    acquirer.stop()
                except Exception:
                    pass
                try:
                    acquirer.destroy()
                except Exception:
                    pass
            try:
                harvester.reset()
            except Exception:
                pass
            if isinstance(error, CameraConnectionError):
                raise
            raise CameraConnectionError(f"相机初始化失败：{error}") from error
        self._harvester = harvester
        self._acquirer = acquirer
        self._sequence = 0
        self._closed = False

    def fetch_frame(self, timeout_s: float = 0.2) -> FramePacket:
        if not self.connected:
            raise CameraConnectionError("相机尚未连接")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        try:
            with self._acquirer.fetch(timeout=float(timeout_s)) as buffer:
                # receive_timestamp 是主机拿到缓冲区的时刻；若相机提供曝光设备时间，
                # 再通过时钟映射得到真正用于配准的 capture host timestamp。
                receive_timestamp_s = float(self._clock_ns()) / 1_000_000_000.0
                device_timestamp_ns = getattr(buffer, "timestamp_ns", None)
                if device_timestamp_ns is None:
                    capture_timestamp_s = receive_timestamp_s
                    timestamp_source = "host_receive"
                    timestamp_reliable = False
                    timestamp_uncertainty_s = float("inf")
                else:
                    try:
                        device_timestamp_s = float(device_timestamp_ns) / 1_000_000_000.0
                    except (TypeError, ValueError) as error:
                        raise CameraFrameError("相机设备时间戳不是有效数值") from error
                    mapping = self._clock_mapper.observe(
                        device_timestamp_s, receive_timestamp_s
                    )
                    capture_timestamp_s = mapping.host_timestamp_s
                    timestamp_source = "camera_device"
                    timestamp_reliable = mapping.reliable
                    timestamp_uncertainty_s = mapping.uncertainty_s
                components = buffer.payload.components
                if not components:
                    raise CameraFrameError("相机缓冲区没有图像 component")
                component = components[0]
                actual_size = (int(component.width), int(component.height))
                expected_size = (
                    self.settings.image_width_px,
                    self.settings.image_height_px,
                )
                if actual_size != expected_size:
                    raise CameraFrameError(
                        f"实际图像分辨率 {actual_size} 与配置/标定分辨率 {expected_size} 不符"
                    )
                image = convert_component_to_image(component)
        except CameraFrameError:
            raise
        except Exception as error:
            raise CameraFrameError(f"取帧失败：{error}") from error
        quality = 0.9 if self.pixel_format_fallback else 1.0
        packet = FramePacket(
            capture_timestamp_s,
            self._sequence,
            image,
            True,
            float(quality),
            receive_timestamp_s=receive_timestamp_s,
            timestamp_source=timestamp_source,
            timestamp_reliable=timestamp_reliable,
            timestamp_uncertainty_s=timestamp_uncertainty_s,
        )
        self._sequence += 1
        return packet

    def close(self) -> None:
        if self._closed:
            return
        acquirer, harvester = self._acquirer, self._harvester
        self._closed = True
        self._acquirer = None
        self._harvester = None
        if acquirer is not None:
            try:
                acquirer.stop()
            except Exception:
                pass
            try:
                acquirer.destroy()
            except Exception:
                pass
        if harvester is not None:
            try:
                harvester.reset()
            except Exception:
                pass

    def __enter__(self) -> GentlCameraCapture:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class CameraCaptureWorker:
    """连续取帧线程；缓冲容量恒为一，慢消费者不会造成延迟堆积。"""

    def __init__(
        self, capture: GentlCameraCapture, *, fetch_timeout_s: float = 0.2
    ) -> None:
        self.capture = capture
        self.fetch_timeout_s = fetch_timeout_s
        self.latest_frame: LatestValueBuffer[FramePacket] = LatestValueBuffer()
        self.last_error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.last_error = None
        self.capture.connect()
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self.latest_frame.publish(self.capture.fetch_frame(self.fetch_timeout_s))
        except Exception as error:
            self.last_error = error
            self._stop.set()
        finally:
            self.capture.close()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            self.capture.close()
            return
        if thread is not threading.current_thread():
            thread.join(timeout_s)
        if thread.is_alive():
            raise CameraShutdownTimeout(
                "相机取帧线程仍在运行，底层资源保持打开；上层必须进入故障停车并稍后重试"
            )
        # 资源由采集线程的 finally 释放，避免调用者与 C 驱动取帧并发 destroy。

