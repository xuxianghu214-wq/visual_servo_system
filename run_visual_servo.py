"""视觉伺服中文界面入口。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from visual_servo_system.runtime.production_factory import (  # noqa: E402
    build_production_orchestrator,
)
from visual_servo_system.ui.visual_servo_main_window import VisualServoMainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    try:
        orchestrator = build_production_orchestrator(ROOT)
    except Exception as error:
        QMessageBox.critical(
            None,
            "启动失败",
            f"无法初始化工程：{error}\n\n请先运行 scripts/verify_environment.py。",
        )
        return 2
    window = VisualServoMainWindow(orchestrator)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
