# Pure helpers: importing this file never configures, builds or installs anything.
function Get-GpuFirstSourceSnapshot {
    param([string[]]$SourceRoots, [string[]]$SourceFiles = @())
    $extensions = @('.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.hxx','.cuh','.cu',
        '.inl','.inc','.ipp','.s','.asm','.def','.rc','.cmake','.in','.rs','.ps1','.py','.patch')
    $paths = @($SourceFiles)
    foreach($sourceRoot in $SourceRoots) {
        if(!(Test-Path -LiteralPath $sourceRoot -PathType Container)) { continue }
        $paths += @(Get-ChildItem -LiteralPath $sourceRoot -File -Recurse | Where-Object {
            $_.Name -eq 'CMakeLists.txt' -or $_.Extension.ToLowerInvariant() -in $extensions
        } | ForEach-Object FullName)
    }
    $snapshot = [ordered]@{}
    foreach($sourcePath in @($paths | Sort-Object -Unique)) {
        $absolutePath = [IO.Path]::GetFullPath($sourcePath)
        $snapshot[$absolutePath] = (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256 -ErrorAction Stop).Hash
    }
    return $snapshot
}

function Compare-GpuFirstSourceSnapshots {
    param($Before, $After)
    foreach($sourcePath in @((@($Before.Keys) + @($After.Keys)) | Sort-Object -Unique)) {
        if($Before[$sourcePath] -ne $After[$sourcePath]) { Write-Output $sourcePath }
    }
}

function Write-GpuFirstTaintMarker {
    param([string]$MarkerPath, [string[]]$ChangedFiles, [string]$BeforePath,
          [string]$AfterPath, [string]$SnapshotError = '')
    $marker = [ordered]@{
        reason = 'Native build inputs changed during compilation or could not be verified'
        detected_utc = [DateTime]::UtcNow.ToString('o')
        cleaned_before_retry = $false
        changed_files = @($ChangedFiles)
        before_snapshot = $BeforePath
        after_snapshot = $AfterPath
        snapshot_error = $SnapshotError
        install_blocked = $true
    }
    $marker | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $MarkerPath -Encoding utf8
}

function Assert-GpuFirstStableSources {
    param($Before, $After, [string]$MarkerPath, [string]$BeforePath = '', [string]$AfterPath = '')
    $changed = @(Compare-GpuFirstSourceSnapshots -Before $Before -After $After)
    if($changed.Count) {
        Write-GpuFirstTaintMarker -MarkerPath $MarkerPath -ChangedFiles $changed -BeforePath $BeforePath -AfterPath $AfterPath
        throw "Native sources changed while compiling ($($changed.Count) files). Build marked tainted and installation blocked: $MarkerPath. The next build will clean host and CUDA objects once."
    }
}

function ConvertFrom-GpuFirstNinjaDependencies {
    param([string[]]$Lines)
    $record = $null
    foreach($line in $Lines) {
        if($line -match '^(?<object>.+): #deps (?<count>[0-9]+), deps mtime \S+ \((?<status>[^)]+)\)$') {
            if($record) { Write-Output $record }
            $record = [pscustomobject]@{
                Object = $Matches.object
                Count = [uint64]::Parse($Matches.count)
                Status = $Matches.status
                Dependencies = [Collections.Generic.List[string]]::new()
            }
        } elseif([string]::IsNullOrWhiteSpace($line)) {
            if($record) { Write-Output $record; $record = $null }
        } elseif($record -and $line -match '^\s+(.+)$') {
            $record.Dependencies.Add($Matches[1].Trim())
        }
    }
    if($record) { Write-Output $record }
}

function Assert-GpuFirstNoReparsePath {
    param([string]$Path, [string]$StopAt = '')
    $cursor = [IO.Path]::GetFullPath($Path)
    while($cursor) {
        if(Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
            if($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Dependency repair refuses a linked path: $cursor"
            }
        }
        if($StopAt -and $cursor.Equals($StopAt, [StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = Split-Path $cursor -Parent
        if(!$parent -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
}

function Resolve-GpuFirstDependencyOutput {
    param([string]$BuildRoot, [string]$OutputDirectory)
    $expected = [IO.Path]::GetFullPath((Join-Path $BuildRoot 'out')).TrimEnd('\','/')
    $absolute = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\','/')
    if(!$absolute.Equals($expected, [StringComparison]::OrdinalIgnoreCase) -or
       !(Test-Path -LiteralPath $absolute -PathType Container)) {
        throw "Dependency repair requires the existing BuildRoot/out directory: $absolute"
    }
    Assert-GpuFirstNoReparsePath -Path $absolute
    return (Resolve-Path -LiteralPath $absolute -ErrorAction Stop).ProviderPath.TrimEnd('\','/')
}

function Resolve-GpuFirstDependencyObject {
    param([string]$OutputDirectory, [string]$Object)
    $local = $Object.Replace('/', [IO.Path]::DirectorySeparatorChar).Replace('\', [IO.Path]::DirectorySeparatorChar)
    $absolute = if([IO.Path]::IsPathRooted($local)) { [IO.Path]::GetFullPath($local) }
                else { [IO.Path]::GetFullPath((Join-Path $OutputDirectory $local)) }
    $prefix = $OutputDirectory.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    if(!$absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
       [IO.Path]::GetRelativePath($OutputDirectory, $absolute).Contains(':')) {
        throw "Dependency object escapes the intended build output: $Object"
    }
    Assert-GpuFirstNoReparsePath -Path $absolute -StopAt $OutputDirectory
    if(Test-Path -LiteralPath $absolute) {
        if(!(Test-Path -LiteralPath $absolute -PathType Leaf)) { throw "Dependency object is not a file: $absolute" }
        return (Resolve-Path -LiteralPath $absolute -ErrorAction Stop).ProviderPath
    }
    return $null
}

function Get-GpuFirstNinjaDependencies {
    param([string]$Ninja, [string]$OutputDirectory)
    $lines = @(& $Ninja -C $OutputDirectory -t deps 2>&1 | ForEach-Object { $_.ToString() })
    if($LASTEXITCODE -ne 0) { throw 'Cannot inspect Ninja header dependencies' }
    return @(ConvertFrom-GpuFirstNinjaDependencies -Lines $lines)
}

function Test-GpuFirstHeaderlessStub {
    param([string]$NativeSource, [string]$OutputDirectory, [string]$ObjectPath)
    if(!$NativeSource) { return $false }
    $relative = [IO.Path]::GetRelativePath($OutputDirectory, $ObjectPath).Replace('\','/').ToLowerInvariant()
    $stubs = @{
        '_deps/llama_cpp-build/tools/server/cmakefiles/llama-server.dir/main.cpp.obj' = @('tools/server/main.cpp', 'llama_server')
        '_deps/llama_cpp-build/tools/quantize/cmakefiles/llama-quantize.dir/main.cpp.obj' = @('tools/quantize/main.cpp', 'llama_quantize')
    }
    if($stubs.ContainsKey($relative)) {
        $source = Join-Path $NativeSource $stubs[$relative][0]
        if(!(Test-Path -LiteralPath $source -PathType Leaf)) { return $false }
        $entry = $stubs[$relative][1]
        $actual = [regex]::Replace([IO.File]::ReadAllText($source), '\s+', '')
        return $actual -ceq "int${entry}(intargc,char**argv);intmain(intargc,char**argv){return${entry}(argc,argv);}"
    }
    if($relative -eq '_deps/llama_cpp-build/ggml/src/cmakefiles/ggml-cpu.dir/ggml-cpu/hbm.cpp.obj') {
        $source = Join-Path $NativeSource 'ggml/src/ggml-cpu/hbm.cpp'
        $cache = Join-Path $OutputDirectory 'CMakeCache.txt'
        $build = Join-Path $OutputDirectory 'build.ninja'
        if(!(Test-Path -LiteralPath $source -PathType Leaf) -or !(Test-Path -LiteralPath $cache -PathType Leaf) -or
           !(Test-Path -LiteralPath $build -PathType Leaf) -or $env:CL -or $env:_CL_) { return $false }
        $options = [IO.File]::ReadAllText($build)
        return [IO.File]::ReadAllText($cache) -match '(?m)^GGML_CPU_HBM:BOOL=OFF\r?$' -and
               $options -notmatch 'GGML_USE_CPU_HBM|(?:^|\s)(?:/FI|-include)' -and
               (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -eq 'E5CF5C8327F99CCB58461D8416E27BC38DA01D9BD450E48F3DF69BD34A81A47C'
    }
    return $false
}

function Repair-GpuFirstHeaderDependencies {
    param([string]$BuildRoot, [string]$OutputDirectory, [object[]]$Records, [string]$NativeSource = '')
    $output = Resolve-GpuFirstDependencyOutput -BuildRoot $BuildRoot -OutputDirectory $OutputDirectory
    $candidates = [Collections.Generic.List[string]]::new()
    $cudaZero = 0
    $stubs = 0
    foreach($record in $Records) {
        if($record.Count -ne 0) { continue }
        if($record.Object -match '\.cu\.obj$') { ++$cudaZero; continue }
        if($record.Object -notmatch '\.(c|cc|cpp|cxx)\.obj$') { continue }
        $path = Resolve-GpuFirstDependencyObject -OutputDirectory $output -Object $record.Object
        if($path) {
            if(Test-GpuFirstHeaderlessStub -NativeSource $NativeSource -OutputDirectory $output -ObjectPath $path) { ++$stubs }
            else { $candidates.Add($path) }
        }
    }
    # Validate the complete set before removing any object. A stale locale prefix
    # can leave old zero-header records marked VALID even after CMake is repaired.
    $removed = @($candidates | Sort-Object -Unique)
    foreach($path in $removed) { Remove-Item -LiteralPath $path -Force -ErrorAction Stop }
    return [pscustomobject]@{ RemovedObjects = $removed; RemovedCount = $removed.Count; CudaZeroRecords = $cudaZero; HeaderlessStubs = $stubs }
}

function Assert-GpuFirstHeaderDependencies {
    param([string]$BuildRoot, [string]$OutputDirectory, [object[]]$Records)
    $output = Resolve-GpuFirstDependencyOutput -BuildRoot $BuildRoot -OutputDirectory $OutputDirectory
    foreach($record in @($Records | Where-Object { $_.Object -match '\.cu\.obj$' -and $_.Count -eq 0 })) {
        throw "Native installation blocked: CUDA object has zero recorded header dependencies: $($record.Object). CUDA objects were preserved; repair the NVCC dependency output and explicitly rebuild them."
    }
    $sentinels = @(
        @{ Label = 'llama graph'; Pattern = '[/\\]llama\.dir[/\\]llama-graph\.cpp\.obj$'; Headers = @('[/\\]llama-graph\.h$') },
        @{ Label = 'model architecture Unity'; Pattern = '[/\\]llama\.dir[/\\]Unity[/\\]unity_[0-9]+_cxx\.cxx\.obj$'; Headers = @('[/\\]llama-graph\.h$') },
        @{ Label = 'model RAM cache'; Pattern = '[/\\]llama\.dir[/\\]llama-model-ram\.cpp\.obj$'; Headers = @('[/\\]llama-model-ram\.h$') },
        @{ Label = 'CUDA backend'; Pattern = '[/\\]ggml-cuda\.dir[/\\]ggml-cuda\.cu\.obj$'; Headers = @(
            '[/\\]ggml-cuda\.h$', '[/\\]ggml-cuda[/\\]common\.cuh$',
            '[/\\]ggml-cuda-upload\.h$', '[/\\]ggml-cuda-host-op\.h$') }
    )
    foreach($sentinel in $sentinels) {
        $matchingRecords = @($Records | Where-Object { $_.Object -match $sentinel.Pattern })
        $healthy = $false
        foreach($record in $matchingRecords) {
            $path = Resolve-GpuFirstDependencyObject -OutputDirectory $output -Object $record.Object
            $headersPresent = $true
            foreach($header in $sentinel.Headers) {
                $headersPresent = $headersPresent -and @($record.Dependencies | Where-Object { $_ -match $header }).Count -gt 0
            }
            if($path -and $record.Count -gt 0 -and $record.Status -eq 'VALID' -and $headersPresent) {
                $healthy = $true
            } else {
                throw "Native installation blocked: $($sentinel.Label) object has missing header dependencies: $($record.Object)"
            }
        }
        if(!$healthy) { throw "Native installation blocked: $($sentinel.Label) has no verified header dependencies. Check MSVC /showIncludes prefix and rebuild stale host objects." }
    }
}

function Assert-GpuFirstBuiltSources {
    param([string]$SnapshotPath, [string[]]$SourceRoots, [string[]]$SourceFiles = @())
    if(!(Test-Path -LiteralPath $SnapshotPath -PathType Leaf)) {
        throw 'Native installation blocked: no successful source snapshot exists. Run a verified -Build first.'
    }
    $built = Get-Content -LiteralPath $SnapshotPath -Raw -ErrorAction Stop | ConvertFrom-Json -AsHashtable -ErrorAction Stop
    if($built -isnot [Collections.IDictionary] -or !$built.Count) { throw 'Native installation blocked: invalid successful source snapshot.' }
    $current = Get-GpuFirstSourceSnapshot -SourceRoots $SourceRoots -SourceFiles $SourceFiles
    $changed = @(Compare-GpuFirstSourceSnapshots -Before $built -After $current)
    if($changed.Count) {
        throw "Native installation blocked: $($changed.Count) source inputs differ from the last successful build. Run a verified -Build before copying the payload."
    }
}
