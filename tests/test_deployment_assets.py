import base64
import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_entry_scripts_and_documents_exist():
    for relative in (
        "run_visual_servo.py",
        "启动视觉伺服.ps1",
        "一键离线安装.ps1",
        "安装相机驱动.ps1",
        "scripts/verify_environment.py",
        "scripts/Verify-FileManifest.ps1",
        "docs/实机联调步骤.md",
        "docs/故障排查.md",
        "docs/迁移验收.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_offline_payload_is_not_ignored_but_generated_environments_are():
    ignore_rules = (ROOT / ".gitignore").read_text("utf-8-sig").splitlines()
    assert "offline_packages/" not in ignore_rules
    assert ".venv/" in ignore_rules
    assert ".python311/" in ignore_rules
    assert ".pip_offline_temp_*/" in ignore_rules


def test_powershell_scripts_use_project_root_and_offline_install():
    install = (ROOT / "一键离线安装.ps1").read_text("utf-8-sig")
    launch = (ROOT / "启动视觉伺服.ps1").read_text("utf-8-sig")
    driver = (ROOT / "安装相机驱动.ps1").read_text("utf-8-sig")
    assert "$PSScriptRoot" in install
    assert "--no-index" in install and "offline_packages" in install
    assert r"third_party_installers\Python_3.11.9_x64" in install
    assert "$PSScriptRoot" in launch
    assert "$PSScriptRoot" in driver and "管理员" in driver
    assert "Invoke-NativeChecked" in install
    assert "$LASTEXITCODE" in install
    assert "Assert-ProcessSucceeded" in install
    assert "Test-Sha256Manifest" in install
    assert "New-ShortPathMapping" in install
    assert "subst.exe" in install
    assert "$env:TEMP" in install and "$env:TMP" in install
    assert "finally" in install
    assert '"-e", $MappedRoot' not in install
    assert '"--no-build-isolation",' in install
    assert '"--force-reinstall", $MappedRoot' in install
    portable_check = (
        "Assert-InstalledProjectPortable -FilePath $PythonExe "
        "-ProjectRoot $ProjectRoot"
    )
    assert portable_check in install
    assert install.rindex(portable_check) > install.rindex("finally {")
    assert "runpy.run_path" in install
    assert "[string]$PythonOverride" not in install
    python_signature = install.index("Get-AuthenticodeSignature -LiteralPath $PythonInstaller")
    python_execution = install.index("Start-Process -FilePath $PythonInstaller")
    assert python_signature < python_execution
    assert "Test-Sha256Manifest" in driver
    assert "Get-AuthenticodeSignature" in driver
    assert "SpinnakerSDK_FULL_4.4.0.246_x64.exe" in driver
    standard_position = launch.index(r".venv\Scripts\python.exe")
    development_position = launch.index(r".venv\python.exe")
    assert standard_position < development_position


def test_all_powershell_entry_scripts_parse_without_errors():
    for name in ("启动视觉伺服.ps1", "一键离线安装.ps1", "安装相机驱动.ps1"):
        path = ROOT / name
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        command = (
            "$tokens=$null;$errors=$null;"
            "$text=([Text.Encoding]::UTF8.GetString("
            f"[Convert]::FromBase64String('{encoded}'))).TrimStart([char]0xFEFF);"
            "[System.Management.Automation.Language.Parser]::ParseInput($text,"
            "[ref]$tokens,[ref]$errors)|Out-Null;"
            "if($errors.Count){$errors|ForEach-Object{$_.Message};exit 1}"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert completed.returncode == 0, f"{name}: {completed.stdout}{completed.stderr}"


def test_offline_installer_propagates_native_failure_and_never_prints_success(tmp_path):
    script = tmp_path / "一键离线安装.ps1"
    script.write_bytes((ROOT / "一键离线安装.ps1").read_bytes())
    helper_dir = tmp_path / "scripts"
    helper_dir.mkdir()
    (helper_dir / "Verify-FileManifest.ps1").write_bytes(
        (ROOT / "scripts" / "Verify-FileManifest.ps1").read_bytes()
    )
    (tmp_path / "offline_packages").mkdir()
    (tmp_path / "offline_packages" / "SHA256SUMS.txt").write_text("", encoding="ascii")
    python_package = (
        tmp_path / "third_party_installers" / "Python_3.11.9_x64"
    )
    python_package.mkdir(parents=True)
    python_installer = python_package / "python-3.11.9-amd64.exe"
    python_installer.write_bytes(b"test installer placeholder")
    python_digest = hashlib.sha256(python_installer.read_bytes()).hexdigest()
    (python_package / "SHA256SUMS.txt").write_text(
        f"{python_digest}  {python_installer.name}\n", encoding="ascii"
    )
    (tmp_path / "requirements-lock.txt").write_text("", encoding="utf-8")
    fake = tmp_path / "fake-python.cmd"
    fake.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
    fake_subst = tmp_path / "fake-subst.cmd"
    fake_subst.write_text("@echo off\r\nexit /b 7\r\n", encoding="ascii")

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TestOnlyPythonOverride",
            str(fake),
            "-SubstExecutable",
            str(fake_subst),
        ],
        capture_output=True,
        text=False,
        check=False,
        timeout=15,
    )

    assert completed.returncode != 0
    output = (completed.stdout or b"") + (completed.stderr or b"")
    assert b"7" in output
    assert "安装完成".encode() not in output
    assert "安装完成".encode("gbk") not in output
    assert not list(tmp_path.glob(".pip_offline_temp_*"))


def test_offline_installer_requires_python_manifest_even_with_override(tmp_path):
    script = tmp_path / "一键离线安装.ps1"
    script.write_bytes((ROOT / "一键离线安装.ps1").read_bytes())
    helper_dir = tmp_path / "scripts"
    helper_dir.mkdir()
    (helper_dir / "Verify-FileManifest.ps1").write_bytes(
        (ROOT / "scripts" / "Verify-FileManifest.ps1").read_bytes()
    )
    (tmp_path / "offline_packages").mkdir()
    (tmp_path / "offline_packages" / "SHA256SUMS.txt").write_text(
        "", encoding="ascii"
    )
    (tmp_path / "requirements-lock.txt").write_text("", encoding="utf-8")
    fake = tmp_path / "fake-python.cmd"
    fake.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TestOnlyPythonOverride",
            str(fake),
        ],
        capture_output=True,
        text=False,
        check=False,
        timeout=15,
    )

    assert completed.returncode != 0
    output = (completed.stdout or b"") + (completed.stderr or b"")
    assert b"Python_3.11.9_x64" in output
    assert not list(tmp_path.glob(".pip_offline_temp_*"))


def test_offline_installer_rejects_a_non_exact_existing_python(tmp_path):
    script = tmp_path / "一键离线安装.ps1"
    script.write_bytes((ROOT / "一键离线安装.ps1").read_bytes())
    helper_dir = tmp_path / "scripts"
    helper_dir.mkdir()
    (helper_dir / "Verify-FileManifest.ps1").write_bytes(
        (ROOT / "scripts" / "Verify-FileManifest.ps1").read_bytes()
    )
    (tmp_path / "offline_packages").mkdir()
    (tmp_path / "offline_packages" / "SHA256SUMS.txt").write_text(
        "", encoding="ascii"
    )
    python_package = (
        tmp_path / "third_party_installers" / "Python_3.11.9_x64"
    )
    python_package.mkdir(parents=True)
    python_installer = python_package / "python-3.11.9-amd64.exe"
    python_installer.write_bytes(b"test installer placeholder")
    python_digest = hashlib.sha256(python_installer.read_bytes()).hexdigest()
    (python_package / "SHA256SUMS.txt").write_text(
        f"{python_digest}  {python_installer.name}\n", encoding="ascii"
    )
    standard_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    standard_python.parent.mkdir(parents=True)
    standard_python.write_text("@echo off\r\nexit /b 9\r\n", encoding="ascii")

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        capture_output=True,
        text=False,
        check=False,
        timeout=15,
    )

    assert completed.returncode != 0
    output = (completed.stdout or b"") + (completed.stderr or b"")
    assert b"3.11.9 x64" in output


def test_python_installer_must_have_a_valid_authenticode_signature(tmp_path):
    script = tmp_path / "一键离线安装.ps1"
    script.write_bytes((ROOT / "一键离线安装.ps1").read_bytes())
    helper_dir = tmp_path / "scripts"
    helper_dir.mkdir()
    (helper_dir / "Verify-FileManifest.ps1").write_bytes(
        (ROOT / "scripts" / "Verify-FileManifest.ps1").read_bytes()
    )
    (tmp_path / "offline_packages").mkdir()
    (tmp_path / "offline_packages" / "SHA256SUMS.txt").write_text(
        "", encoding="ascii"
    )
    python_package = tmp_path / "third_party_installers" / "Python_3.11.9_x64"
    python_package.mkdir(parents=True)
    python_installer = python_package / "python-3.11.9-amd64.exe"
    python_installer.write_bytes(b"unsigned test installer; never execute")
    python_digest = hashlib.sha256(python_installer.read_bytes()).hexdigest()
    (python_package / "SHA256SUMS.txt").write_text(
        f"{python_digest}  {python_installer.name}\n", encoding="ascii"
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        capture_output=True,
        text=False,
        check=False,
        timeout=15,
    )

    assert completed.returncode != 0
    output = (completed.stdout or b"") + (completed.stderr or b"")
    assert b"Authenticode" in output
    assert not (tmp_path / ".python311").exists()


def test_test_only_python_override_is_rejected_outside_pytest(tmp_path):
    script = tmp_path / "一键离线安装.ps1"
    script.write_bytes((ROOT / "一键离线安装.ps1").read_bytes())
    fake = tmp_path / "fake-python.cmd"
    fake.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
    environment = os.environ.copy()
    environment.pop("PYTEST_CURRENT_TEST", None)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TestOnlyPythonOverride",
            str(fake),
        ],
        capture_output=True,
        text=False,
        check=False,
        timeout=15,
        env=environment,
    )

    assert completed.returncode != 0
    output = (completed.stdout or b"") + (completed.stderr or b"")
    assert b"TEST-ONLY Python override is disabled outside pytest" in output


def test_user_docs_use_standard_venv_launcher_and_consistent_dry_run_order():
    readme = (ROOT / "README_先看这里.md").read_text("utf-8")
    migration = (ROOT / "docs" / "迁移验收.md").read_text("utf-8")
    combined = readme + migration
    assert r".venv\Scripts\python.exe" in combined
    assert r".venv\python.exe" not in combined
    assert "连接机器人只读" in migration
    assert "干运行不发送" in migration
    assert "507 MB" in combined
    assert "不要复制 `.venv`" in combined
    assert "subst.exe" in combined
    assert "SHA256SUMS.txt" in combined
    assert "Python 安装程序的 Authenticode" in combined
    assert "非 editable" in combined


def _run_manifest_check(directory: Path) -> subprocess.CompletedProcess[bytes]:
    helper = ROOT / "scripts" / "Verify-FileManifest.ps1"
    manifest = directory / "SHA256SUMS.txt"

    def ps_quote(value: Path) -> str:
        return str(value).replace("'", "''")

    command = (
        "$ErrorActionPreference='Stop'; try { "
        f". '{ps_quote(helper)}'; "
        f"Test-Sha256Manifest -Directory '{ps_quote(directory)}' "
        f"-ManifestPath '{ps_quote(manifest)}' | Out-Null; exit 0 "
        "} catch { Write-Error $_; exit 1 }"
    )
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        check=False,
        timeout=15,
    )


def test_sha256_manifest_accepts_exact_files_and_rejects_tampering(tmp_path):
    package = tmp_path / "离线 包"
    package.mkdir()
    payload = package / "example.whl"
    payload.write_bytes(b"known bytes")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (package / "SHA256SUMS.txt").write_text(
        f"{digest}  {payload.name}\n", encoding="ascii"
    )
    assert _run_manifest_check(package).returncode == 0

    payload.write_bytes(b"tampered")
    assert _run_manifest_check(package).returncode != 0


def test_sha256_manifest_rejects_missing_extra_and_traversal_files(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    missing_hash = hashlib.sha256(b"missing").hexdigest()
    manifest = package / "SHA256SUMS.txt"
    manifest.write_text(f"{missing_hash}  missing.whl\n", encoding="ascii")
    assert _run_manifest_check(package).returncode != 0

    payload = package / "listed.whl"
    payload.write_bytes(b"listed")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest.write_text(f"{digest}  listed.whl\n", encoding="ascii")
    (package / "unlisted.whl").write_bytes(b"extra")
    assert _run_manifest_check(package).returncode != 0

    (package / "unlisted.whl").unlink()
    (package / "unlisted_directory").mkdir()
    assert _run_manifest_check(package).returncode != 0

    (package / "unlisted_directory").rmdir()
    manifest.write_text(f"{digest}  ..\\listed.whl\n", encoding="ascii")
    assert _run_manifest_check(package).returncode != 0
