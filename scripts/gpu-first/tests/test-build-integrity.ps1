$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
. (Join-Path (Split-Path $PSScriptRoot -Parent) 'build-integrity.ps1')
$fixtureBase = Join-Path $repoRoot '.build\gpu-first-moe\build-deps-selftest'
$fixtureRoot = Join-Path $fixtureBase ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null
function Require([bool]$Condition, [string]$Message) { if(!$Condition) { throw $Message } }
function Rejects([scriptblock]$Action, [string]$Message) {
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    Require $rejected $Message
}
function Put([string]$Path, [string]$Text = '') {
    New-Item -ItemType Directory -Path (Split-Path $Path -Parent) -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Text)
}
try {
    $buildRoot = Join-Path $fixtureRoot 'build'
    $output = Join-Path $buildRoot 'out'
    $zero = Join-Path $output 'obj\old.cxx.obj'
    $valid = Join-Path $output 'obj\valid.cpp.obj'
    $cuda = Join-Path $output 'obj\kernel.cu.obj'
    Put $zero; Put $valid; Put $cuda
    $lines = @(
        'obj/old.cxx.obj: #deps 0, deps mtime 123 (VALID)', '',
        'obj/valid.cpp.obj: #deps 1, deps mtime 124 (VALID)', '    ../native/header.h', '',
        'obj/kernel.cu.obj: #deps 0, deps mtime 125 (VALID)', ''
    )
    $records = @(ConvertFrom-GpuFirstNinjaDependencies -Lines $lines)
    Require ($records.Count -eq 3 -and $records[1].Dependencies[0] -eq '../native/header.h') 'Ninja dependency parsing failed'
    $repair = Repair-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $records
    Require ($repair.RemovedCount -eq 1 -and $repair.CudaZeroRecords -eq 1) 'Repair counted the wrong objects'
    Require (!(Test-Path -LiteralPath $zero) -and (Test-Path -LiteralPath $valid) -and (Test-Path -LiteralPath $cuda)) 'Repair changed CUDA or nonzero dependencies'

    $stubNative = Join-Path $fixtureRoot 'stub-native'
    $stubLines = @()
    foreach($entry in @('server', 'quantize')) {
        $name = if($entry -eq 'server') { 'llama_server' } else { 'llama_quantize' }
        $target = if($entry -eq 'server') { 'llama-server' } else { 'llama-quantize' }
        Put (Join-Path $stubNative "tools/$entry/main.cpp") "int $name(int argc, char ** argv);`nint main(int argc, char ** argv) { return $name(argc, argv); }"
        $relative = "_deps/llama_cpp-build/tools/$entry/CMakeFiles/$target.dir/main.cpp.obj"
        Put (Join-Path $output $relative)
        $stubLines += @("${relative}: #deps 0, deps mtime 130 (VALID)", '')
    }
    $stubRecords = @(ConvertFrom-GpuFirstNinjaDependencies -Lines $stubLines)
    $stubRepair = Repair-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $stubRecords -NativeSource $stubNative
    Require ($stubRepair.RemovedCount -eq 0 -and $stubRepair.HeaderlessStubs -eq 2) 'Known include-free forwarding stubs were rebuilt'
    Put (Join-Path $stubNative 'tools/server/main.cpp') '#include "changed.h"'
    $stubRepair = Repair-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $stubRecords -NativeSource $stubNative
    Require ($stubRepair.RemovedCount -eq 1 -and $stubRepair.HeaderlessStubs -eq 1) 'Changed stub incorrectly bypassed dependency repair'

    Put $zero
    $outside = Join-Path $fixtureRoot 'outside.cpp.obj'
    Put $outside
    $unsafe = @(ConvertFrom-GpuFirstNinjaDependencies -Lines @(
        'obj/old.cxx.obj: #deps 0, deps mtime 123 (VALID)', '',
        '../../outside.cpp.obj: #deps 0, deps mtime 124 (VALID)', ''
    ))
    Rejects { Repair-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $unsafe } 'Escaping object path was accepted'
    Require ((Test-Path -LiteralPath $zero) -and (Test-Path -LiteralPath $outside)) 'Batch validation deleted files before rejecting an unsafe path'
    Rejects { Resolve-GpuFirstDependencyOutput -BuildRoot $buildRoot -OutputDirectory $fixtureRoot } 'Unexpected output root was accepted'

    $sentinelLines = @()
    foreach($name in @('llama-graph.cpp.obj', 'Unity/unity_0_cxx.cxx.obj', 'Unity/unity_1_cxx.cxx.obj', 'llama-model-ram.cpp.obj')) {
        $relative = '_deps/llama_cpp-build/src/CMakeFiles/llama.dir/' + $name
        Put (Join-Path $output $relative)
        $header = if($name -eq 'llama-model-ram.cpp.obj') { 'llama-model-ram.h' } else { 'llama-graph.h' }
        $sentinelLines += @("${relative}: #deps 1, deps mtime 200 (VALID)", "    ../native/src/$header", '')
    }
    $cudaRelative = '_deps/llama_cpp-build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.obj'
    Put (Join-Path $output $cudaRelative)
    $sentinelLines += @("${cudaRelative}: #deps 4, deps mtime 200 (VALID)",
        '    ../native/ggml/include/ggml-cuda.h', '    ../native/ggml/src/ggml-cuda/common.cuh',
        '    ../native/ggml/include/ggml-cuda-upload.h', '    ../native/ggml/include/ggml-cuda-host-op.h', '')
    $sentinels = @(ConvertFrom-GpuFirstNinjaDependencies -Lines $sentinelLines)
    Assert-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $sentinels
    $sentinels[2].Count = 0
    Rejects { Assert-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $sentinels } 'An old zero-dependency Unity object passed the installation check'
    $sentinels[2].Count = 1
    $sentinels[0].Status = 'STALE'
    Rejects { Assert-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $sentinels } 'A stale Ninja record passed the installation check'
    $sentinels[0].Status = 'VALID'
    Rejects { Assert-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records ($sentinels + $records[2]) } 'A zero-dependency CUDA object passed the installation check'
    Require (Test-Path -LiteralPath $cuda) 'CUDA failure removed an object automatically'
    [void]$sentinels[4].Dependencies.Remove('../native/ggml/include/ggml-cuda-upload.h')
    Rejects { Assert-GpuFirstHeaderDependencies -BuildRoot $buildRoot -OutputDirectory $output -Records $sentinels } 'Missing CUDA provider headers passed the installation check'
    $sentinels[4].Dependencies.Add('../native/ggml/include/ggml-cuda-upload.h')

    if($IsWindows) {
        $linkTarget = Join-Path $fixtureRoot 'linked-target'
        $link = Join-Path $output 'linked'
        Put (Join-Path $linkTarget 'escape.cpp.obj')
        New-Item -ItemType Junction -Path $link -Target $linkTarget | Out-Null
        try {
            Rejects { Resolve-GpuFirstDependencyObject -OutputDirectory $output -Object 'linked/escape.cpp.obj' } 'Linked object path was accepted'
        } finally {
            Require ([IO.Path]::GetFullPath($link).StartsWith($output + '\', [StringComparison]::OrdinalIgnoreCase)) 'Unexpected fixture link location'
            Remove-Item -LiteralPath $link -Force
        }
    }

    $source = Join-Path $fixtureRoot 'source'
    $header = Join-Path $source 'layout.h'
    $patch = Join-Path $source 'compat/006-gpu-first-runtime.patch'
    Put $header 'int first;'
    Put $patch '--- old.cpp\n+++ new.cpp\n+int first;'
    $before = Get-GpuFirstSourceSnapshot -SourceRoots @($source)
    Require ($before.Count -eq 2 -and $before[$patch]) 'Compatibility patches were omitted from the source snapshot'
    $builtSnapshot = Join-Path $fixtureRoot 'successful-sources.json'
    $before | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $builtSnapshot -Encoding utf8
    Assert-GpuFirstBuiltSources -SnapshotPath $builtSnapshot -SourceRoots @($source)
    Assert-GpuFirstStableSources -Before $before -After $before -MarkerPath (Join-Path $fixtureRoot 'clean.json')
    Put $header 'int changed;'
    Rejects { Assert-GpuFirstBuiltSources -SnapshotPath $builtSnapshot -SourceRoots @($source) } 'Install-only accepted sources changed after the successful build'
    $after = Get-GpuFirstSourceSnapshot -SourceRoots @($source)
    $marker = Join-Path $fixtureRoot 'tainted.json'
    Rejects { Assert-GpuFirstStableSources -Before $before -After $after -MarkerPath $marker } 'Source changes passed the build guard'
    $taint = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    Require ($taint.install_blocked -and $taint.changed_files.Count -eq 1) 'Source taint metadata is incomplete'
    Put $header 'int first;'
    Put $patch '--- old.cpp\n+++ new.cpp\n+int changed;'
    Rejects { Assert-GpuFirstBuiltSources -SnapshotPath $builtSnapshot -SourceRoots @($source) } 'Install-only accepted a changed compatibility patch'
    $patchAfter = Get-GpuFirstSourceSnapshot -SourceRoots @($source)
    $patchMarker = Join-Path $fixtureRoot 'patch-tainted.json'
    Rejects { Assert-GpuFirstStableSources -Before $before -After $patchAfter -MarkerPath $patchMarker } 'Compatibility patch changes passed the build guard'
    $patchTaint = Get-Content -LiteralPath $patchMarker -Raw | ConvertFrom-Json
    Require ($patchTaint.install_blocked -and $patchTaint.changed_files.Count -eq 1 -and $patchTaint.changed_files[0] -eq $patch) 'Changed compatibility patch was not identified in taint metadata'
    Put $patch '--- old.cpp\n+++ new.cpp\n+int first;'
    Put (Join-Path $source 'added.h') 'int added;'
    Rejects { Assert-GpuFirstBuiltSources -SnapshotPath $builtSnapshot -SourceRoots @($source) } 'Install-only ignored new source inputs'
    Rejects { Assert-GpuFirstBuiltSources -SnapshotPath (Join-Path $fixtureRoot 'missing-snapshot.json') -SourceRoots @($source) } 'Install-only accepted a missing build snapshot'
    Write-Output 'PASS: Ninja VALID/header/CUDA checks, bounded repair, strict stub recognition, path safety and install-only source integrity.'
} finally {
    $absolute = (Resolve-Path -LiteralPath $fixtureRoot -ErrorAction Stop).ProviderPath
    Require ($absolute.StartsWith([IO.Path]::GetFullPath($fixtureBase).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) 'Unexpected self-test cleanup path'
    Assert-GpuFirstNoReparsePath -Path $absolute
    Remove-Item -LiteralPath $absolute -Recurse -Force
}
