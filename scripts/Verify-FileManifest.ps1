<#
.SYNOPSIS
严格校验某个安装包目录中的 SHA256SUMS.txt。

.DESCRIPTION
清单格式固定为“64 位十六进制 SHA256、两个空格、文件名”。为避免清单被用来读取目录外
文件，本函数只接受当前目录中的普通文件名，不接受绝对路径、子目录或“..”。除了清单自身，
目录中的每个文件必须恰好在清单中出现一次；多文件、少文件、重复条目和哈希不符都会失败。
#>
function Test-Sha256Manifest {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )

    $DirectoryItem = Get-Item -LiteralPath $Directory -ErrorAction Stop
    if (-not $DirectoryItem.PSIsContainer) {
        throw "清单校验目录不是文件夹：$Directory"
    }

    $ManifestItem = Get-Item -LiteralPath $ManifestPath -ErrorAction Stop
    if ($ManifestItem.PSIsContainer) {
        throw "SHA256 清单不是文件：$ManifestPath"
    }

    $RootPath = [IO.Path]::GetFullPath($DirectoryItem.FullName).TrimEnd('\', '/')
    $ManifestFullPath = [IO.Path]::GetFullPath($ManifestItem.FullName)
    $ExpectedManifestPath = Join-Path $RootPath $ManifestItem.Name
    if (-not $ManifestFullPath.Equals($ExpectedManifestPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "SHA256 清单必须直接位于待校验目录中：$ManifestFullPath"
    }

    $ExpectedHashes = @{}
    $LineNumber = 0
    foreach ($Line in (Get-Content -LiteralPath $ManifestFullPath -Encoding UTF8)) {
        $LineNumber += 1
        if ([string]::IsNullOrWhiteSpace($Line)) {
            continue
        }

        $Match = [regex]::Match($Line, '^(?<hash>[0-9A-Fa-f]{64})  (?<name>.+)$')
        if (-not $Match.Success) {
            throw "SHA256 清单第 $LineNumber 行格式错误。"
        }

        $FileName = $Match.Groups['name'].Value
        if (
            [IO.Path]::IsPathRooted($FileName) -or
            $FileName -eq '.' -or
            $FileName -eq '..' -or
            $FileName -ne [IO.Path]::GetFileName($FileName) -or
            $FileName.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0
        ) {
            throw "SHA256 清单第 $LineNumber 行包含不安全路径：$FileName"
        }
        if ($ExpectedHashes.ContainsKey($FileName)) {
            throw "SHA256 清单包含重复文件：$FileName"
        }

        $ExpectedHashes[$FileName] = $Match.Groups['hash'].Value.ToUpperInvariant()
    }

    foreach ($FileName in $ExpectedHashes.Keys) {
        $Candidate = Join-Path $RootPath $FileName
        $CandidateItem = Get-Item -LiteralPath $Candidate -ErrorAction SilentlyContinue
        if ($null -eq $CandidateItem -or $CandidateItem.PSIsContainer) {
            throw "SHA256 清单中的文件缺失：$FileName"
        }
        # 直接使用 .NET，避免目标电脑的 PowerShell 模块搜索路径被第三方软件覆盖。
        $Stream = [IO.File]::OpenRead($CandidateItem.FullName)
        try {
            $Hasher = [Security.Cryptography.SHA256]::Create()
            try {
                $HashBytes = $Hasher.ComputeHash($Stream)
                $ActualHash = ([BitConverter]::ToString($HashBytes)).Replace('-', '')
            }
            finally {
                $Hasher.Dispose()
            }
        }
        finally {
            $Stream.Dispose()
        }
        if (-not $ActualHash.Equals($ExpectedHashes[$FileName], [StringComparison]::OrdinalIgnoreCase)) {
            throw "SHA256 校验失败，文件可能损坏或被替换：$FileName"
        }
    }

    $ActualDirectories = @(Get-ChildItem -LiteralPath $RootPath -Force -Directory)
    if ($ActualDirectories.Count -ne 0) {
        throw "安装包目录包含未列入 SHA256 清单的额外子目录：$($ActualDirectories[0].Name)"
    }

    $ActualFiles = @(
        Get-ChildItem -LiteralPath $RootPath -Force -File | Where-Object {
            -not $_.FullName.Equals($ManifestFullPath, [StringComparison]::OrdinalIgnoreCase)
        }
    )
    foreach ($ActualFile in $ActualFiles) {
        if (-not $ExpectedHashes.ContainsKey($ActualFile.Name)) {
            throw "安装包目录包含未列入 SHA256 清单的额外文件：$($ActualFile.Name)"
        }
    }
    if ($ActualFiles.Count -ne $ExpectedHashes.Count) {
        throw "SHA256 清单条目数与目录文件数不一致。"
    }

    return $true
}
