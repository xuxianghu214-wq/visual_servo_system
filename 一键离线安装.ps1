param(
    [Parameter(DontShow = $true)][string]$TestOnlyPythonOverride = "",
    [Parameter(DontShow = $true)][string]$SubstExecutable = "subst.exe"
)

$ErrorActionPreference = "Stop"
if (
    $TestOnlyPythonOverride -and
    $env:PYTEST_CURRENT_TEST -notmatch '(^|[\\/])test_deployment_assets\.py::test_'
) {
    throw "TEST-ONLY Python override is disabled outside pytest."
}
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$ManifestHelper = Join-Path $ProjectRoot "scripts\Verify-FileManifest.ps1"
if (-not (Test-Path -LiteralPath $ManifestHelper -PathType Leaf)) {
    throw "缺少 SHA256 清单校验脚本：$ManifestHelper"
}
. $ManifestHelper

$OfflinePackages = Join-Path $ProjectRoot "offline_packages"
$OfflineManifest = Join-Path $OfflinePackages "SHA256SUMS.txt"
$Environment = Join-Path $ProjectRoot ".venv"
$StandardPython = Join-Path $Environment "Scripts\python.exe"
$DevelopmentPython = Join-Path $Environment "python.exe"
$PythonPackageRoot = Join-Path $ProjectRoot "third_party_installers\Python_3.11.9_x64"
$PythonManifest = Join-Path $PythonPackageRoot "SHA256SUMS.txt"
$PythonInstaller = Join-Path $PythonPackageRoot "python-3.11.9-amd64.exe"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$StepName
    )

    & $FilePath @ArgumentList
    $NativeExitCode = $LASTEXITCODE
    if ($NativeExitCode -ne 0) {
        throw "$StepName 失败，退出码 $NativeExitCode。"
    }
}

function Assert-ProcessSucceeded {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$StepName
    )

    if ($Process.ExitCode -ne 0) {
        throw "$StepName 失败，退出码 $($Process.ExitCode)。"
    }
}

function Assert-ExactPythonRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath
    )

    # 锁定补丁版本和位数，避免误用旧电脑遗留的 .venv 或 32 位 Python。
    $CheckCode = "import struct, sys; raise SystemExit(0 if sys.version_info[:3] == (3, 11, 9) and struct.calcsize('P') == 8 else 91)"
    try {
        & $FilePath -I -c $CheckCode
        $VersionExitCode = $LASTEXITCODE
    }
    catch {
        throw "无法执行 Python 3.11.9 x64 版本校验：$($_.Exception.Message)"
    }
    if ($VersionExitCode -ne 0) {
        throw "检测到的解释器不是 Python 3.11.9 x64。请删除旧 .venv 后重新运行本脚本。"
    }
}

function Assert-InstalledProjectPortable {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )

    # -I 排除当前目录和 PYTHONPATH，只从新环境的 site-packages 导入。
    # 此检查在 subst 解除后执行，并以非 __main__ 名称加载真实启动入口：
    # 既能捕获残留 Z:\src 的 editable .pth，又不会创建界面或连接任何硬件。
    $Launcher = Join-Path $ProjectRoot "run_visual_servo.py"
    $CheckCode = "import runpy, sys, visual_servo_system; namespace = runpy.run_path(sys.argv[1], run_name='visual_servo_portability_check'); raise SystemExit(0 if callable(namespace.get('main')) else 92)"
    try {
        & $FilePath -I -c $CheckCode $Launcher
        $ImportExitCode = $LASTEXITCODE
    }
    catch {
        throw "解除临时盘符后无法导入视觉伺服工程：$($_.Exception.Message)"
    }
    if ($ImportExitCode -ne 0) {
        throw "解除临时盘符后无法导入视觉伺服工程，退出码 $ImportExitCode。"
    }
}

function New-ShortPathMapping {
    param(
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    # 从 Z: 向下寻找未使用盘符，避免长中文路径触发 PySide6 的 Win32 260 字符限制。
    for ($Code = [int][char]'Z'; $Code -ge [int][char]'D'; $Code -= 1) {
        $DriveName = [char]$Code
        $Drive = "$DriveName`:"
        if ($null -ne (Get-PSDrive -Name $DriveName -ErrorAction SilentlyContinue)) {
            continue
        }

        & $SubstExecutable $Drive $TargetPath
        $SubstExitCode = $LASTEXITCODE
        if ($SubstExitCode -ne 0) {
            throw "创建临时短路径 $Drive 失败，subst.exe 退出码 $SubstExitCode。"
        }

        $MappedRoot = "$Drive\"
        if (-not (Test-Path -LiteralPath $MappedRoot -PathType Container)) {
            & $SubstExecutable $Drive /D
            $CleanupExitCode = $LASTEXITCODE
            if ($CleanupExitCode -ne 0) {
                throw "临时短路径 $Drive 创建后不可访问，且清理失败，退出码 $CleanupExitCode。"
            }
            throw "临时短路径 $Drive 创建后不可访问。"
        }
        return [PSCustomObject]@{ Drive = $Drive; Root = $MappedRoot }
    }
    throw "未找到可供离线安装使用的空闲盘符（已检查 Z: 到 D:）。"
}

function Remove-ShortPathMapping {
    param(
        [Parameter(Mandatory = $true)]$Mapping
    )

    & $SubstExecutable $Mapping.Drive /D
    $SubstExitCode = $LASTEXITCODE
    if ($SubstExitCode -ne 0) {
        throw "解除临时短路径 $($Mapping.Drive) 失败，subst.exe 退出码 $SubstExitCode。"
    }
}

function Convert-ToMappedProjectPath {
    param(
        [Parameter(Mandatory = $true)][string]$RealPath,
        [Parameter(Mandatory = $true)][string]$MappedRoot
    )

    $FullPath = [IO.Path]::GetFullPath($RealPath)
    $RootWithSeparator = $ProjectRoot + [IO.Path]::DirectorySeparatorChar
    if ($FullPath.Equals($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)) {
        return $MappedRoot
    }
    if ($FullPath.StartsWith($RootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        return Join-Path $MappedRoot $FullPath.Substring($RootWithSeparator.Length)
    }
    return $FullPath
}

if (-not (Test-Path -LiteralPath $OfflinePackages -PathType Container)) {
    throw "缺少 offline_packages 离线 wheel 目录。"
}
Test-Sha256Manifest -Directory $OfflinePackages -ManifestPath $OfflineManifest | Out-Null

# Python 安装资源是可迁移工程的一部分；即使使用测试/开发解释器也不能跳过完整性校验。
if (-not (Test-Path -LiteralPath $PythonPackageRoot -PathType Container)) {
    throw "缺少 Python 3.11.9 x64 安装包目录：$PythonPackageRoot"
}
Test-Sha256Manifest -Directory $PythonPackageRoot -ManifestPath $PythonManifest | Out-Null

if ($TestOnlyPythonOverride) {
    if (-not (Test-Path -LiteralPath $TestOnlyPythonOverride -PathType Leaf)) {
        throw "指定的 TestOnlyPythonOverride 不存在：$TestOnlyPythonOverride"
    }
    $PythonExe = [IO.Path]::GetFullPath($TestOnlyPythonOverride)
}
elseif (Test-Path -LiteralPath $StandardPython -PathType Leaf) {
    $PythonExe = $StandardPython
}
elseif (Test-Path -LiteralPath $DevelopmentPython -PathType Leaf) {
    # 兼容本仓库现有的 Conda 开发环境。迁移时不要复制它，应在目标电脑重新创建标准 venv。
    $PythonExe = $DevelopmentPython
}
else {
    if (-not (Test-Path -LiteralPath $PythonInstaller -PathType Leaf)) {
        throw "未找到确切版本的 Python 3.11.9 x64 安装包：$PythonInstaller"
    }
    $PythonSignature = Get-AuthenticodeSignature -LiteralPath $PythonInstaller
    if ($PythonSignature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        throw "Python 3.11.9 x64 安装程序的 Authenticode 签名无效，状态：$($PythonSignature.Status)。已拒绝运行。"
    }

    $PortablePython = Join-Path $ProjectRoot ".python311"
    $Arguments = "/quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 Include_pip=1 PrependPath=0 TargetDir=`"$PortablePython`""
    $InstallerProcess = Start-Process -FilePath $PythonInstaller -ArgumentList $Arguments -Wait -WindowStyle Hidden -PassThru
    Assert-ProcessSucceeded -Process $InstallerProcess -StepName "Python 3.11.9 安装程序"

    $PortablePythonExe = Join-Path $PortablePython "python.exe"
    if (-not (Test-Path -LiteralPath $PortablePythonExe -PathType Leaf)) {
        throw "Python 安装程序返回成功，但没有生成：$PortablePythonExe"
    }
    # venv 必须用真实路径创建，保证 pyvenv.cfg 不会引用稍后解除的临时盘符。
    Invoke-NativeChecked -FilePath $PortablePythonExe -ArgumentList @("-m", "venv", $Environment) -StepName "创建标准 venv"
    if (-not (Test-Path -LiteralPath $StandardPython -PathType Leaf)) {
        throw "标准 venv 创建完成后仍未找到：$StandardPython"
    }
    $PythonExe = $StandardPython
}

if (-not $TestOnlyPythonOverride) {
    Assert-ExactPythonRuntime -FilePath $PythonExe
}

$Mapping = $null
$TemporaryDirectoryReal = $null
$OldTemp = [Environment]::GetEnvironmentVariable("TEMP", "Process")
$OldTmp = [Environment]::GetEnvironmentVariable("TMP", "Process")
try {
    $Mapping = New-ShortPathMapping -TargetPath $ProjectRoot
    $MappedRoot = $Mapping.Root
    $TemporaryDirectoryName = ".pip_offline_temp_$([guid]::NewGuid().ToString('N'))"
    $TemporaryDirectoryShort = Join-Path $MappedRoot $TemporaryDirectoryName
    $TemporaryDirectoryReal = Join-Path $ProjectRoot $TemporaryDirectoryName
    New-Item -ItemType Directory -Path $TemporaryDirectoryShort -ErrorAction Stop | Out-Null
    $env:TEMP = $TemporaryDirectoryShort
    $env:TMP = $TemporaryDirectoryShort

    $ShortPython = Convert-ToMappedProjectPath -RealPath $PythonExe -MappedRoot $MappedRoot
    $ShortOfflinePackages = Join-Path $MappedRoot "offline_packages"
    $ShortRequirements = Join-Path $MappedRoot "requirements-lock.txt"
    $ShortVerifyScript = Join-Path $MappedRoot "scripts\verify_environment.py"

    Invoke-NativeChecked -FilePath $ShortPython -ArgumentList @(
        "-m", "pip", "install", "--no-index", "--find-links", $ShortOfflinePackages,
        "-r", $ShortRequirements
    ) -StepName "安装锁定依赖"
    Invoke-NativeChecked -FilePath $ShortPython -ArgumentList @(
        "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation",
        "--force-reinstall", $MappedRoot
    ) -StepName "安装视觉伺服工程"
    Invoke-NativeChecked -FilePath $ShortPython -ArgumentList @($ShortVerifyScript) -StepName "验证运行环境"
}
finally {
    if ($null -eq $OldTemp) {
        Remove-Item Env:TEMP -ErrorAction SilentlyContinue
    }
    else {
        $env:TEMP = $OldTemp
    }
    if ($null -eq $OldTmp) {
        Remove-Item Env:TMP -ErrorAction SilentlyContinue
    }
    else {
        $env:TMP = $OldTmp
    }

    # 即使临时目录清理失败，也必须继续尝试解除 subst 映射。
    try {
        if ($null -ne $TemporaryDirectoryReal -and (Test-Path -LiteralPath $TemporaryDirectoryReal)) {
            Remove-Item -LiteralPath $TemporaryDirectoryReal -Recurse -Force -ErrorAction Stop
        }
    }
    finally {
        if ($null -ne $Mapping) {
            Remove-ShortPathMapping -Mapping $Mapping
        }
    }
}

Assert-InstalledProjectPortable -FilePath $PythonExe -ProjectRoot $ProjectRoot
Write-Host "离线 Python 环境安装完成。相机驱动需单独运行“安装相机驱动.ps1”并确认管理员权限。"
