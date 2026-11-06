$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$ManifestHelper = Join-Path $ProjectRoot "scripts\Verify-FileManifest.ps1"
if (-not (Test-Path -LiteralPath $ManifestHelper -PathType Leaf)) {
    throw "缺少 SHA256 清单校验脚本：$ManifestHelper"
}
. $ManifestHelper

$PackageRoot = Join-Path $ProjectRoot "third_party_installers\Teledyne_Spinnaker_4.4.0.246_x64"
$Manifest = Join-Path $PackageRoot "SHA256SUMS.txt"
$Installer = Join-Path $PackageRoot "SpinnakerSDK_FULL_4.4.0.246_x64.exe"

# 先校验目录完整性，再允许用户确认和提权。MSI 只是完整 EXE 的随附资源，本脚本不会逐个调用。
Test-Sha256Manifest -Directory $PackageRoot -ManifestPath $Manifest | Out-Null
if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "未找到 Teledyne Spinnaker 4.4.0.246 完整安装入口：$Installer"
}
$Signature = Get-AuthenticodeSignature -LiteralPath $Installer
if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Spinnaker 完整安装程序的 Authenticode 签名无效，状态：$($Signature.Status)。已拒绝运行。"
}

$Answer = Read-Host "安装相机驱动需要管理员权限并会修改 Windows。输入 YES 继续"
if ($Answer -cne "YES") {
    Write-Host "已取消，未修改系统。"
    exit 0
}

$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-Object Security.Principal.WindowsPrincipal($Identity)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $ElevationArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    $ElevationProcess = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $ElevationArguments -Wait -PassThru
    if ($ElevationProcess.ExitCode -ne 0) {
        throw "管理员安装进程失败，退出码 $($ElevationProcess.ExitCode)。"
    }
    exit 0
}

$InstallerProcess = Start-Process -FilePath $Installer -Wait -PassThru
if ($InstallerProcess.ExitCode -ne 0) {
    throw "Spinnaker 完整安装程序失败，退出码 $($InstallerProcess.ExitCode)。"
}
Write-Host "Spinnaker 完整安装程序已成功退出。请重新运行环境检查。"
