# 构建脚本：用随 Visual Studio 安装的 CMake 配置并编译（不需要 DevEco、不需要改环境变量）
#   powershell -File tools\build.ps1            配置 + 编译 Release
#   powershell -File tools\build.ps1 -Clean     先删构建目录再从头编
param(
  [switch]$Clean,
  [string]$BuildDir = 'D:\NNE-win-build',
  [string]$Config = 'Release'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path $cmake)) {
  $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source
}
if (-not $cmake) { throw '找不到 cmake.exe（装 Visual Studio 2022 时会自带）' }

if ($Clean -and (Test-Path $BuildDir)) {
  Remove-Item $BuildDir -Recurse -Force
}

& $cmake -S $root -B $BuildDir -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw "配置失败：$LASTEXITCODE" }

& $cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "编译失败：$LASTEXITCODE" }

Write-Host ''
Write-Host "构建完成：$BuildDir\$Config" -ForegroundColor Green
Get-ChildItem (Join-Path $BuildDir $Config) -Filter '*.exe' | ForEach-Object { Write-Host ('  ' + $_.Name) }
