param(
    [string]$NativeSource = '',
    [string]$BuildRoot = '',
    [switch]$Build,
    [switch]$SkipConfigure,
    [string]$InstallRoot = '',
    [string]$CudaVersion = '',
    [string]$CudaArchitectures = '120',
    [int]$Parallel = 4,
    [string[]]$Targets = @('llama-server','llama-quantize','test-moe-ram','test-moe-ram-metrics','test-gpu-exclusive-scheduler','test-gpu-exclusive','test-gpu-exclusive-op','test-gpu-exclusive-ssm','test-gpu-layer-diagnostics','test-cuda-memory-budget','test-server-component','test-moe-admission','test-moe-budget','test-moe-partition','test-cuda-upload','test-cuda-host-op','test-prepare-async','test-moe-ram-pool','test-expert-trace','test-cuda-dense','test-cuda-kv-growth','test-kv-growth','test-dense-embedding')
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'build-integrity.ps1')
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if(!$BuildRoot) { $BuildRoot = Join-Path $repoRoot '.build\gpu-first-moe' }
$buildDirectory = [IO.Path]::GetFullPath($BuildRoot)
if(!$NativeSource) { $NativeSource = Join-Path $buildDirectory 'native' }
$nativeDirectory = [IO.Path]::GetFullPath($NativeSource)
if(!(Test-Path -LiteralPath (Join-Path $nativeDirectory 'CMakeLists.txt') -PathType Leaf)) {
    throw "Prepared llama.cpp source missing: $nativeDirectory. Apply the pinned compatibility patches first."
}
if($Parallel -lt 1) { throw 'Parallel must be positive' }
$outputDirectory = Join-Path $buildDirectory 'out'
$temporaryDirectory = Join-Path $buildDirectory 'temp'
$integrityDirectory = Join-Path $buildDirectory 'source-integrity'
$taintMarker = Join-Path $integrityDirectory 'tainted-build.json'
$vsRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
$ninja = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$cudaBase = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
if(!$CudaVersion) {
    # Some toolkit version directories contain runtime-only installations.
    foreach($version in @('13.0','12.9','12.8')) {
        if(Test-Path -LiteralPath (Join-Path $cudaBase "v$version\bin\nvcc.exe") -PathType Leaf) {
            $CudaVersion = $version
            break
        }
    }
    if(!$CudaVersion) { throw 'No existing CUDA 12.8/12.9/13.0 compiler found; this script never installs tools.' }
}
$cudaDirectory = Join-Path $cudaBase "v$CudaVersion"
$cudaCmakeDirectory = $cudaDirectory.Replace('\','/')
$runnerDirectory = if($CudaVersion.StartsWith('13.')) { 'cuda_v13' } else { 'cuda_v12' }
foreach($required in @($vcvars,$ninja,$cmake,(Join-Path $cudaDirectory 'bin\nvcc.exe'))) {
    if(!(Test-Path -LiteralPath $required -PathType Leaf)) { throw "Existing tool not found: $required" }
}
New-Item -ItemType Directory -Force -Path $temporaryDirectory | Out-Null
$env:TEMP = $temporaryDirectory; $env:TMP = $temporaryDirectory; $env:TMPDIR = $temporaryDirectory
$env:CUDA_CACHE_PATH = Join-Path $buildDirectory 'cuda-cache'
$env:VSLANG = '1033'
New-Item -ItemType Directory -Force -Path $env:CUDA_CACHE_PATH | Out-Null
# All environment changes are process-local. No tool installation, user settings,
# production executable, model directory or running server is modified.
$compilerEnvironment = & $env:ComSpec /d /s /c ('call "' + $vcvars + '" >nul && set')
if($LASTEXITCODE -ne 0) { throw 'Visual Studio compiler environment failed' }
foreach($entry in $compilerEnvironment) {
    $separator = $entry.IndexOf('=')
    if($separator -gt 0) { [Environment]::SetEnvironmentVariable($entry.Substring(0,$separator),$entry.Substring($separator+1),'Process') }
}
$env:CUDA_PATH = $cudaDirectory
$env:PATH = (Join-Path $cudaDirectory 'bin') + ';' + $env:PATH
$env:OLLAMA_LLAMA_CPP_SOURCE = $null
if(!$SkipConfigure) {
    # Keep a tiny bootstrap in BuildRoot. This preserves CMAKE_HOME_DIRECTORY
    # when using the already configured validation build; no second CUDA tree.
    $trackedCmake = (Join-Path $PSScriptRoot 'CMakeLists.txt').Replace('\','/')
    $bootstrap = "cmake_minimum_required(VERSION 3.24)`nproject(wildllama_gpu_first_validation C CXX)`ninclude(`"$trackedCmake`")`n"
    [IO.File]::WriteAllText((Join-Path $buildDirectory 'CMakeLists.txt'),$bootstrap,[Text.UTF8Encoding]::new($false))
    & $cmake -S $buildDirectory -B $outputDirectory -G Ninja `
        "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_BUILD_TYPE=Release `
        "-DWILDLLAMA_NATIVE=$($nativeDirectory.Replace('\','/'))" `
        "-DCMAKE_CUDA_COMPILER=$cudaCmakeDirectory/bin/nvcc.exe" `
        "-DCUDAToolkit_ROOT=$cudaCmakeDirectory" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
        -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=ON `
        -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON `
        "-DOLLAMA_RUNNER_DIR=$runnerDirectory" "-DCMAKE_CUDA_FLAGS=-t 2"
    if($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
}
$sourceRoots = @($nativeDirectory, (Join-Path $repoRoot 'llama\compat'),
    (Join-Path $repoRoot 'llama\server'))
$sourceFiles = @((Join-Path $repoRoot 'LLAMA_CPP_VERSION'),
    (Join-Path $PSScriptRoot 'CMakeLists.txt'), (Join-Path $PSScriptRoot 'build-windows.ps1'),
    (Join-Path $PSScriptRoot 'build-integrity.ps1'))
if($Build) {
    New-Item -ItemType Directory -Force -Path $integrityDirectory | Out-Null
    if(Test-Path -LiteralPath $taintMarker -PathType Leaf) {
        $taint = Get-Content -Raw -LiteralPath $taintMarker | ConvertFrom-Json
        if(!$taint.cleaned_before_retry) {
            Write-Output 'Previous build is tainted. Cleaning all host/CUDA targets once before retry.'
            & $cmake --build $outputDirectory --target clean
            if($LASTEXITCODE -ne 0) { throw 'Cannot clean the tainted native build' }
            $taint.cleaned_before_retry = $true
            $taint | Add-Member -NotePropertyName cleaned_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
            $taint | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taintMarker -Encoding utf8
        }
    }
    $dependencyRecords = @(Get-GpuFirstNinjaDependencies -Ninja $ninja -OutputDirectory $outputDirectory)
    $dependencyRepair = Repair-GpuFirstHeaderDependencies -BuildRoot $buildDirectory -OutputDirectory $outputDirectory -Records $dependencyRecords -NativeSource $nativeDirectory
    if($dependencyRepair.RemovedCount) {
        Write-Output "Invalidated $($dependencyRepair.RemovedCount) stale host objects with zero recorded header dependencies. CUDA objects and valid nonzero records are preserved."
    }
    if($dependencyRepair.CudaZeroRecords) {
        throw "Build blocked: Ninja reports $($dependencyRepair.CudaZeroRecords) CUDA objects with zero dependencies. CUDA objects were preserved; repair dependency output and explicitly rebuild them."
    }
    if(!$InstallRoot -and $Targets.Count -eq 1 -and $Targets[0] -eq 'test-gpu-exclusive-op') {
        # This isolated target links ggml only. Unrelated model/PLE edits are
        # not its build inputs and must not taint the full CUDA runtime cache.
        $sourceRoots = @((Join-Path $nativeDirectory 'ggml'), (Join-Path $nativeDirectory 'cmake'),
            (Join-Path $repoRoot 'llama\server'))
        $sourceFiles += @((Join-Path $nativeDirectory 'CMakeLists.txt'),
            (Join-Path $nativeDirectory 'tests\test-gpu-exclusive-op.cpp'))
    }
    $snapshotName = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fffffff')
    $beforePath = Join-Path $integrityDirectory ($snapshotName + '-before.json')
    $afterPath = Join-Path $integrityDirectory ($snapshotName + '-after.json')
    $before = Get-GpuFirstSourceSnapshot -SourceRoots $sourceRoots -SourceFiles $sourceFiles
    $before | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $beforePath -Encoding utf8
    $nativeBuildException = $null
    try {
        & $cmake --build $outputDirectory --parallel $Parallel --target @Targets
        $nativeBuildExitCode = $LASTEXITCODE
    } catch {
        # A native-command error preference must not skip the after snapshot.
        $nativeBuildException = $_
        $nativeBuildExitCode = 1
    }
    try {
        $after = Get-GpuFirstSourceSnapshot -SourceRoots $sourceRoots -SourceFiles $sourceFiles
        $after | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $afterPath -Encoding utf8
    } catch {
        Write-GpuFirstTaintMarker -MarkerPath $taintMarker -BeforePath $beforePath -AfterPath $afterPath -SnapshotError $_.Exception.Message
        throw "Cannot verify native inputs after compilation; build marked tainted and installation blocked: $taintMarker"
    }
    Assert-GpuFirstStableSources -Before $before -After $after -MarkerPath $taintMarker -BeforePath $beforePath -AfterPath $afterPath
    if($nativeBuildException) { throw $nativeBuildException }
    if($nativeBuildExitCode -ne 0) { throw 'Native build failed' }
    if($InstallRoot -or $Targets -contains 'llama-server' -or $Targets -contains 'llama-quantize' -or
       $Targets -contains 'test-gpu-exclusive' -or $Targets -contains 'test-server-component') {
        $dependencyRecords = @(Get-GpuFirstNinjaDependencies -Ninja $ninja -OutputDirectory $outputDirectory)
        Assert-GpuFirstHeaderDependencies -BuildRoot $buildDirectory -OutputDirectory $outputDirectory -Records $dependencyRecords
    }
    # Keep the marker through failed retries; remove it only after a complete,
    # successful build with identical before/after source hashes.
    if(Test-Path -LiteralPath $taintMarker -PathType Leaf) { Remove-Item -LiteralPath $taintMarker }
    Copy-Item -LiteralPath $afterPath -Destination (Join-Path $integrityDirectory 'last-successful-sources.json') -Force
} elseif(!$InstallRoot) {
    Write-Output "Configured only. To compile: & '$PSCommandPath' -Build -SkipConfigure -BuildRoot '$buildDirectory' -NativeSource '$nativeDirectory'"
}
if($InstallRoot) {
    if(Test-Path -LiteralPath $taintMarker -PathType Leaf) {
        throw "Installation blocked by tainted build marker: $taintMarker. Run a stable -Build first."
    }
    $dependencyRecords = @(Get-GpuFirstNinjaDependencies -Ninja $ninja -OutputDirectory $outputDirectory)
    Assert-GpuFirstHeaderDependencies -BuildRoot $buildDirectory -OutputDirectory $outputDirectory -Records $dependencyRecords
    Assert-GpuFirstBuiltSources -SnapshotPath (Join-Path $integrityDirectory 'last-successful-sources.json') -SourceRoots $sourceRoots -SourceFiles $sourceFiles
    $installDirectory = [IO.Path]::GetFullPath($InstallRoot)
    & $cmake --install $outputDirectory --prefix $installDirectory --component llama-server
    if($LASTEXITCODE -ne 0) { throw 'CUDA runtime installation failed' }
    & $cmake --install $outputDirectory --prefix $installDirectory --component gpu-first-runtime
    if($LASTEXITCODE -ne 0) { throw 'Base runtime installation failed' }
    Write-Output "Native payload installed at $installDirectory"
}
