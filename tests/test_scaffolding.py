from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "visual_servo_system"


def test_project_uses_src_layout_and_declares_supported_python() -> None:
    pyproject = PROJECT_ROOT / "pyproject.toml"

    assert pyproject.is_file()
    contents = pyproject.read_text(encoding="utf-8")
    assert 'name = "visual-servo-system"' in contents
    assert 'requires-python = ">=3.11,<3.12"' in contents
    assert 'package-dir = {"" = "src"}' in contents
    assert 'requires = ["setuptools==84.0.0"]' in contents


def test_expected_packages_exist() -> None:
    expected = {
        "camera",
        "vision",
        "geometry",
        "estimation",
        "control",
        "robot",
        "safety",
        "ui",
        "logging",
        "runtime",
        "config",
        "models",
        "diagnostics",
        "offline",
    }

    assert (PACKAGE_ROOT / "__init__.py").is_file()
    assert expected == {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }


def test_lock_file_contains_exact_requested_direct_dependencies() -> None:
    expected = {
        "numpy==2.2.6",
        "opencv-contrib-python-headless==4.12.0.88",
        "PySide6==6.11.2",
        "harvesters==1.4.3",
        "genicam==1.6.0",
        "PyYAML==6.0.3",
        "scipy==1.16.3",
        "osqp==1.1.3",
        "ur_rtde==1.6.5",
        "pytest==9.1.1",
        "pytest-qt==4.5.0",
        "ruff==0.16.10",
        "cloudpickle==3.1.2",
        "colorama==0.4.6",
        "iniconfig==2.3.1",
        "Jinja2==3.1.6",
        "joblib==1.6.0",
        "MarkupSafe==3.0.4",
        "packaging==26.3",
        "pluggy==1.7.0",
        "Pygments==2.21.0",
        "PySide6_Addons==6.11.2",
        "PySide6_Essentials==6.11.2",
        "setuptools==84.0.0",
        "shiboken6==6.11.2",
        "typing_extensions==4.16.0",
    }
    lock_contents = (PROJECT_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    lock_lines = {
        line.strip()
        for line in lock_contents.splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert lock_lines == expected
