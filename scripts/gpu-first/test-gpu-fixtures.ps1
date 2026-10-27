param(
    [Parameter(Mandatory=$true)][string]$FixtureRoot,
    [string]$BuildRoot = '',
    [string]$OutputRoot = '',
    [string]$PrecisionFixture = '',
    [string]$CudaNemotronFixture = '',
    [string]$ReportRoot = '',
    [string]$CasePattern = '.*',
    [long[]]$CacheBytes = @(),
    [switch]$ResidentOnly,
    [switch]$SkipResidentDiagnostics
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (!$BuildRoot) { $BuildRoot = Join-Path $repoRoot '.build\gpu-first-moe' }
if (!$OutputRoot) { $OutputRoot = Join-Path $BuildRoot 'out' }
if (!$PrecisionFixture) { $PrecisionFixture = Join-Path $BuildRoot 'fixtures\gpt-oss-wide-MXFP4-prec-q8.gguf' }
if (!$ReportRoot) { $ReportRoot = Join-Path $BuildRoot ('reports\gpu-fixtures-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
foreach ($parameter in @('FixtureRoot','BuildRoot','OutputRoot','PrecisionFixture','ReportRoot')) {
    Set-Variable -Name $parameter -Value $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath((Get-Variable -Name $parameter -ValueOnly))
}
if ($CudaNemotronFixture) { $CudaNemotronFixture = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CudaNemotronFixture) }
$executable = Join-Path $OutputRoot 'bin\test-gpu-exclusive.exe'
if (!(Test-Path -LiteralPath $executable)) { throw "Missing validation executable: $executable" }
New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null
$env:TEMP = Join-Path $BuildRoot 'temp'
$env:TMP = $env:TEMP
$env:CUDA_CACHE_PATH = Join-Path $BuildRoot 'cuda-cache'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:CUDA_CACHE_PATH | Out-Null
$cases = @(
    @('qwen3moe-q4.gguf',1048576,256,32), @('qwen3moe-q4.gguf',32768,256,32),
    @('llama-tied-wide-Q4_K_M.gguf',32768,256,32), @('llama-tied-wide-Q4_K_M.gguf',1048576,512,64),
    @('gpt-oss-wide-MXFP4.gguf',32768,256,32), @('gpt-oss-wide-MXFP4.gguf',1048576,256,32),
    @('qwen2moe.gguf',131072,256,32), @('qwen2moe.gguf',1048576,512,64),
    @('olmoe.gguf',131072,256,32), @('olmoe.gguf',1048576,256,32),
    @('gemma4-q4.gguf',65536,256,32), @('gemma4-q4.gguf',1048576,512,64),
    @('nemotron_h_moe-q4.gguf',65536,256,32), @('nemotron_h_moe-q4.gguf',1048576,512,64),
    @('llama-tied-wide.gguf',131072,256,32), @('llama-tied-wide.gguf',1048576,512,64),
    @('qwen3moe-split-00001-of-00006.gguf',131072,256,32),
    @($PrecisionFixture,1048576,256,32), @($PrecisionFixture,32768,256,32)
)
if ($CudaNemotronFixture) {
    $cases += ,@($CudaNemotronFixture,65536,256,32)
    $cases += ,@($CudaNemotronFixture,1048576,512,64)
}
$selectedCases = [Collections.Generic.List[object]]::new()
foreach ($case in $cases) {
    if (([IO.Path]::GetFileName($case[0]) -match $CasePattern) -and (!$CacheBytes.Count -or $CacheBytes -contains $case[1])) {
        $selectedCases.Add($case)
    }
}
$cases = $selectedCases.ToArray()
if (!$cases.Count) { throw "No fixture names match CasePattern: $CasePattern" }
$results = [Collections.Generic.List[object]]::new()
$invariant = [Globalization.CultureInfo]::InvariantCulture
$number = [Globalization.NumberStyles]::Float
function Invoke-FixtureCase($case, [bool]$ResidentDiagnostic) {
    $fixture = if ([IO.Path]::IsPathRooted($case[0])) { $case[0] } else { Join-Path $FixtureRoot $case[0] }
    if (!(Test-Path -LiteralPath $fixture)) { throw "Missing fixture: $fixture" }
    $name = "$([IO.Path]::GetFileName($case[0]))-$($case[1])-$($case[2])-$($case[3])"
    if ($ResidentDiagnostic) { $name += '-resident-chunks' }
    $log = Join-Path $ReportRoot ($name + '.log')
    $stdout = Join-Path $ReportRoot ($name + '.stdout.log')
    Write-Output "Testing $name"
    # All model paths are passed as one Windows argument, including paths with spaces.
    $arguments = @(('"' + $fixture + '"'), $case[1], $case[2], $case[3])
    if ($ResidentDiagnostic) { $arguments += 'resident-chunks' }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $executable -ArgumentList $arguments -WindowStyle Hidden -PassThru -Wait -RedirectStandardError $log -RedirectStandardOutput $stdout
    $watch.Stop()
    $status = $process.ExitCode
    $lines = Get-Content -LiteralPath $log
    $comparison = $lines | Where-Object { $_ -match '^comparison=' } | Select-Object -Last 1
    $finiteMismatch = $status -eq 1 -and $comparison -match ' finite=1 ' -and ($lines -match '^FAIL: GPU streaming logits exceed numerical tolerance$')
    $classification = if ($status -eq 0) { 'PASS' } elseif ($finiteMismatch) { 'FAIL_NUMERICAL' } else { 'FAIL_RUNTIME_OR_NONFINITE' }
    $result = [ordered]@{
        fixture = $fixture; role = $(if ($ResidentDiagnostic) { 'resident-batch-variation' } else { 'streaming' })
        cache_bytes = $case[1]; context = $case[2]; prefill = $case[3]; prefix_without_logits = 2
        tolerance_scaled = 0.0002; exit_code = $status; verdict = $classification
        wall_ms = $watch.ElapsedMilliseconds; log = $log
        result = ($lines | Where-Object { $_ -match '^(PASS|FAIL)' } | Select-Object -Last 1)
        comparison = $comparison
        strict_stats = ($lines | Where-Object { $_ -match '^mode=strict' } | Select-Object -Last 1)
    }
    foreach ($key in @('max_abs','max_scaled','rmse','nmse')) {
        if ($comparison -match ("(?:^| )" + $key + '=([^ ]+)')) {
            $value = 0.0
            if ([double]::TryParse($Matches[1],$number,$invariant,[ref]$value) -and ![double]::IsNaN($value) -and ![double]::IsInfinity($value)) {
                $result[$key] = $value
            } else {
                $result[$key] = $Matches[1]
            }
        }
    }
    if ($comparison -match 'outside_tolerance=([0-9]+)') { $result['outside_tolerance'] = [long]$Matches[1] }
    if ($comparison -match 'top1_same=([0-9]+)/9') { $result['top1_same'] = [int]$Matches[1] }
    $results.Add([pscustomobject]$result)
    $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ReportRoot 'results.json') -Encoding utf8
    Write-Output ($results[-1] | ConvertTo-Json -Compress)
    # Preserve every finite numerical failure. Runtime failures and nonfinite
    # logits stop the suite immediately so an invalid run cannot be hidden.
    if ($classification -eq 'FAIL_RUNTIME_OR_NONFINITE') { throw "Stopped after runtime/nonfinite failure: $log" }
}
$manifest = @()
foreach ($case in $cases) {
    $fixture = if ([IO.Path]::IsPathRooted($case[0])) { $case[0] } else { Join-Path $FixtureRoot $case[0] }
    if (!(Test-Path -LiteralPath $fixture)) { throw "Missing fixture: $fixture" }
    if ($manifest.path -notcontains $fixture) {
        $manifest += [pscustomobject]@{ path=$fixture; bytes=(Get-Item -LiteralPath $fixture).Length; sha256=(Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash }
    }
}
Get-ChildItem -LiteralPath $FixtureRoot -Filter 'qwen3moe-split-*.gguf' | ForEach-Object {
    if ($manifest.path -notcontains $_.FullName) { $manifest += [pscustomobject]@{ path=$_.FullName; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ReportRoot 'fixtures.json') -Encoding utf8
$binaries = @($executable) + @(Get-ChildItem -LiteralPath (Join-Path $OutputRoot 'bin') -Filter '*.dll' | Select-Object -ExpandProperty FullName)
$binaries | ForEach-Object {
    [pscustomobject]@{ path=$_; bytes=(Get-Item -LiteralPath $_).Length; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ReportRoot 'binaries.json') -Encoding utf8
if ($ResidentOnly) {
    $done = @{}
    foreach ($case in $cases) {
        $key = "$($case[0])|$($case[2])|$($case[3])"
        if (!$done.ContainsKey($key)) { Invoke-FixtureCase $case $true; $done[$key] = $true }
    }
} else {
    foreach ($case in $cases) { Invoke-FixtureCase $case $false }
}
if (!$ResidentOnly -and !$SkipResidentDiagnostics) {
    $done = @{}
    foreach ($result in @($results.ToArray())) {
        if ($result.verdict -ne 'FAIL_NUMERICAL') { continue }
        $key = "$($result.fixture)|$($result.context)|$($result.prefill)"
        if (!$done.ContainsKey($key)) {
            Invoke-FixtureCase @($result.fixture,$result.cache_bytes,$result.context,$result.prefill) $true
            $done[$key] = $true
        }
    }
}
if ($results | Where-Object { $_.role -eq 'streaming' -and $_.verdict -ne 'PASS' }) { exit 1 }
if ($ResidentOnly -and ($results | Where-Object { $_.verdict -ne 'PASS' })) { exit 1 }
exit 0
