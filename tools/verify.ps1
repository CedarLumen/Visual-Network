# 一键验证：构建 → 全量断言 → 命令行端到端，结果同时落成日志文件。
#
#   powershell -File tools\verify.ps1                 完整跑一遍（含构建）
#   powershell -File tools\verify.ps1 -SkipBuild      只跑验证（用已有产物）
#   powershell -File tools\verify.ps1 -BuildDir D:\NNE-win-build -Config Release
#
# 退出码：0 = 全部通过；1 = 有断言失败或命令失败。
param(
  [string]$BuildDir = 'D:\NNE-win-build',
  [string]$Config = 'Release',
  [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding $false
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outDir = Join-Path $root 'tools\out'
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $outDir "verify-$stamp.log"

function Say([string]$text, [string]$color = 'Gray') {
  Write-Host $text -ForegroundColor $color
  Add-Content -Path $log -Value $text -Encoding UTF8
}

# 找 cmake：优先用 Visual Studio 自带的那份
$cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path $cmake)) {
  $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source
}
if (-not $cmake) { throw '找不到 cmake.exe（装 Visual Studio 2022 时会自带）' }

Say "== 神经网络模拟引擎 · 一键验证 ==" 'Cyan'
Say "工程目录：$root"
Say "构建目录：$BuildDir（$Config）"
Say "日志文件：$log"

if (-not $SkipBuild) {
  Say ''
  Say '-- 1. 配置与编译 --'
  if (-not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) {
    & $cmake -S $root -B $BuildDir -G 'Visual Studio 17 2022' -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
    if ($LASTEXITCODE -ne 0) { throw "配置失败：$LASTEXITCODE" }
  }
  & $cmake --build $BuildDir --config $Config --parallel
  if ($LASTEXITCODE -ne 0) { throw "编译失败：$LASTEXITCODE" }
  Say '构建完成'
}

$exeDir = Join-Path $BuildDir $Config
$tests = Join-Path $exeDir 'nne_tests.exe'
$cli = Join-Path $exeDir 'nne_cli.exe'
foreach ($f in @($tests, $cli)) {
  if (-not (Test-Path $f)) { throw "缺少产物：$f（先去掉 -SkipBuild 跑一次构建）" }
}

$bad = 0

# ---- 断言套件 ----
Say ''
Say '-- 2. 全量断言套件 --'
$testOut = & $tests 2>&1
$testOut | ForEach-Object { Add-Content -Path $log -Value $_ -Encoding UTF8 }
foreach ($line in $testOut) {
  # 每套件的结果行长这样：「-- F 画布交互数学：通过 41，失败 1」。
  # 只认这一种行，别去匹配尾部的「==== 总计：通过 N，失败 M ====」，否则总计会被
  # 当成最后一个套件的失败数重复计入。
  if ($line -match '^\s*--\s(.+?)：通过\s(\d+)，失败\s(\d+)\s*$') {
    $sfail = [int]$Matches[3]
    if ($sfail -gt 0) { Say ("  [失败] {0}：{1} 项" -f $Matches[1], $sfail) 'Red'; $bad += $sfail }
  }
  if ($line -match '^====\s总计：通过\s(\d+)，失败\s(\d+)\s====$') {
    $pass = [int]$Matches[1]; $fail = [int]$Matches[2]
    Say ("  断言总计：通过 {0}，失败 {1}" -f $pass, $fail) ($(if ($fail -eq 0) { 'Green' } else { 'Red' }))
  }
}

# ---- 命令行端到端 ----
Say ''
Say '-- 3. 命令行端到端 --'
$steps = @(
  @{ name = '界面自检（离屏 + 断言 + 截图）'; args = @('--selftest'); tail = 3 },
  @{ name = '自带 20 个手写数字'; args = @('--samples'); tail = 6 },
  @{ name = '拟合任务训练 400 步'; args = @('--train', 'fit', '400'); tail = 4 },
  @{ name = '异或任务训练 800 步'; args = @('--train', 'xor', '800'); tail = 6 },
  @{ name = 'MNIST 10000 张：CPU 与 CUDA 两条路径'; args = @('--mnist', 'both'); tail = 8 }
)
foreach ($s in $steps) {
  Say ''
  Say ("  > nne_cli {0}" -f ($s.args -join ' '))
  $o = & $cli @($s.args) 2>&1
  $o | ForEach-Object { Add-Content -Path $log -Value $_ -Encoding UTF8 }
  $o | Select-Object -Last $s.tail | ForEach-Object { Say "    $_" }
  if ($o -match '\[失败\]') { Say '    这一项里有失败断言' 'Red'; $bad++ }
  if ($LASTEXITCODE -ne 0) { Say ("    退出码 {0}" -f $LASTEXITCODE) 'Yellow' }
}

# ---- 画布网格直线性（像素级：网格里不允许出现斜线/折线）----
Say ''
Say '-- 4. 画布网格直线性 --'
$py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
# 注意别用 $home 当变量名：PowerShell 里 $HOME 是只读自动变量，赋值会直接报错。
$homeShot = Join-Path $outDir 'ui\01-home.png'
if (-not $py) {
  Say '    跳过：没找到 python.exe（这一项要 python + numpy + pillow）' 'Yellow'
} elseif (-not (Test-Path $homeShot)) {
  Say "    跳过：缺少截图 $homeShot（上面的界面自检没跑成）" 'Yellow'
} else {
  $grid = & $py (Join-Path $root 'tools\grid_check.py') $homeShot 2>&1
  $grid | ForEach-Object { Add-Content -Path $log -Value $_ -Encoding UTF8 }
  $grid | Select-Object -Last 5 | ForEach-Object { Say "    $_" }
  if ($LASTEXITCODE -ne 0) { Say '    网格里出现了斜线或折线' 'Red'; $bad++ }
}

Say ''
if ($bad -eq 0) {
  Say '== 全部通过 ==' 'Green'
} else {
  Say ("== 有 {0} 处失败，详见 {1} ==" -f $bad, $log) 'Red'
}
Say "完整日志：$log"
exit $(if ($bad -eq 0) { 0 } else { 1 })
