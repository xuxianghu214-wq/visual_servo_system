$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
$StandardPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PortablePython = Join-Path $ProjectRoot ".venv\python.exe"

if (Test-Path -LiteralPath $StandardPython -PathType Leaf) {
    $VenvPython = $StandardPython
}
elseif (Test-Path -LiteralPath $PortablePython -PathType Leaf) {
    $VenvPython = $PortablePython
}
else {
    throw "未找到 Python 环境。请先运行一键离线安装.ps1。"
}

& $VenvPython (Join-Path $ProjectRoot "run_visual_servo.py")
$ApplicationExitCode = $LASTEXITCODE
if ($ApplicationExitCode -ne 0) {
    throw "视觉伺服程序异常退出，退出码 $ApplicationExitCode。"
}
exit 0
