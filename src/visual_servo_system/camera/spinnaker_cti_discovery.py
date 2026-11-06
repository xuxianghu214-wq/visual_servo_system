"""查找 Teledyne Spinnaker 的 GenTL ``.cti`` 生产者文件。"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path


class CtiDiscoveryError(RuntimeError):
    """没有找到可用 Spinnaker CTI 时抛出。"""


DEFAULT_SPINNAKER_CTI_PATHS = (
    Path(r"C:\Program Files\Teledyne\Spinnaker\cti64\vs2015\Spinnaker_GenTL_v140.cti"),
    Path(r"C:\Program Files\FLIR Systems\Spinnaker\cti64\vs2015\Spinnaker_GenTL_v140.cti"),
)


def _cti_files_at(path: Path) -> list[Path]:
    """把单个 CTI 或 CTI 目录展开为确定顺序的文件列表。"""

    if path.is_file() and path.suffix.lower() == ".cti":
        return [path.resolve()]
    if path.is_dir():
        return sorted(
            (item.resolve() for item in path.glob("*.cti") if item.is_file()),
            key=lambda item: item.name.casefold(),
        )
    return []


def discover_cti_files(
    extra_paths: Iterable[str | Path] = (),
    *,
    environ: Mapping[str, str] | None = None,
    standard_paths: Iterable[str | Path] | None = None,
) -> list[Path]:
    """按环境变量、工程附加路径、系统标准路径的优先级查找 CTI。"""

    environment = os.environ if environ is None else environ
    candidates: list[Path] = []
    raw_environment = environment.get("GENICAM_GENTL64_PATH", "")
    candidates.extend(Path(part) for part in raw_environment.split(os.pathsep) if part.strip())
    candidates.extend(Path(path) for path in extra_paths)
    candidates.extend(
        Path(path)
        for path in (DEFAULT_SPINNAKER_CTI_PATHS if standard_paths is None else standard_paths)
    )

    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        for cti in _cti_files_at(candidate):
            key = os.path.normcase(str(cti))
            if key not in seen:
                seen.add(key)
                result.append(cti)
    return result


def require_spinnaker_cti(cti_files: Iterable[str | Path]) -> Path:
    """返回首个 Spinnaker CTI；没有时给出面向现场人员的错误。"""

    paths = [Path(path).resolve() for path in cti_files]
    for path in paths:
        if "spinnaker" in path.name.casefold() and path.is_file():
            return path
    searched = "、".join(str(path) for path in paths) or "环境变量和标准安装目录"
    raise CtiDiscoveryError(
        "未找到 Teledyne Spinnaker GenTL CTI 文件。请先安装 Spinnaker，"
        f"并检查 GENICAM_GENTL64_PATH。已检查：{searched}"
    )

