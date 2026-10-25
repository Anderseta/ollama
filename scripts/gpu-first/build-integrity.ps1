# Pure helpers: importing this file never configures, builds or installs anything.
function Get-GpuFirstSourceSnapshot {
    param([string[]]$SourceRoots, [string[]]$SourceFiles = @())
    $extensions = @('.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.hxx','.cuh','.cu',
        '.inl','.inc','.ipp','.s','.asm','.def','.rc','.cmake','.in','.rs','.ps1','.py')
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
