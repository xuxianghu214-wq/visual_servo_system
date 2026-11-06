$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$StandardPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DevelopmentPython = Join-Path $ProjectRoot ".venv\python.exe"

if (Test-Path -LiteralPath $StandardPython -PathType Leaf) {
    $VenvPython = $StandardPython
}
elseif (Test-Path -LiteralPath $DevelopmentPython -PathType Leaf) {
    Write-Warning "正在使用仓库内的 Conda 开发环境；迁移电脑请删除 .venv 后运行一键离线安装.ps1。"
    $VenvPython = $DevelopmentPython
}
else {
    throw "未找到标准环境 .venv\Scripts\python.exe，也未找到兼容的开发环境 .venv\python.exe。请先运行一键离线安装.ps1。"
}

& $VenvPython (Join-Path $ProjectRoot "run_visual_servo.py")
$ApplicationExitCode = $LASTEXITCODE
if ($ApplicationExitCode -ne 0) {
    throw "视觉伺服程序异常退出，退出码 $ApplicationExitCode。"
}
exit 0
