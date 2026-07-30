$src = "C:\Users\max\Projects\quantProbe\Hy3-colibri-int4"
$dst = "C:\Users\max\Projects\quantProbe\Hy3-colibri-int4-cfse"
$cfse = "C:\Users\max\Projects\colibri-hy3\c\cfse_pack.exe"
$env:PATH = "C:\Users\max\scoop\apps\msys2\2026-06-11\mingw64\bin;C:\Users\max\scoop\apps\msys2\2026-06-11\usr\bin;$env:PATH"

Get-ChildItem "$src\out-*.safetensors" | ForEach-Object {
    $f = $_.FullName
    $bn = $_.Name
    $dstFile = Join-Path $dst $bn
    if (Test-Path $dstFile) {
        $hdr = [System.IO.File]::ReadAllBytes($dstFile)[8..200]
        $hdrStr = [System.Text.Encoding]::ASCII.GetString($hdr)
        if ($hdrStr -match '"cfse"') {
            Write-Host "SKIP $bn (already CFSE)"
            return
        }
    }
    Write-Host "CONVERT $bn ..."
    & $cfse $f $dstFile 2>&1 | Write-Host
}
Write-Host "ALL DONE"
