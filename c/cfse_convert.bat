@echo off
setlocal enabledelayedexpansion
set CFSE=C:\Users\max\Projects\colibri-hy3\c\cfse_pack.exe
set SRC=C:\Users\max\Projects\quantProbe\Hy3-colibri-int4
set DST=C:\Users\max\Projects\quantProbe\Hy3-colibri-int4-cfse
set PATH=C:\Users\max\scoop\apps\msys2\2026-06-11\mingw64\bin;%PATH%
for %%f in ("%SRC%\out-*.safetensors") do (
  set dstfile="%DST%\%%~nxf"
  if not exist !dstfile! (
    echo CONVERT %%~nxf ...
    "%CFSE%" "%%f" !dstfile! >> "%DST%\convert_log.txt" 2>&1
  ) else (
    findstr /m "cfse" !dstfile! >nul 2>&1
    if errorlevel 1 (
      echo CONVERT %%~nxf ...
      "%CFSE%" "%%f" !dstfile! >> "%DST%\convert_log.txt" 2>&1
    ) else (
      echo SKIP %%~nxf
    )
  )
)
echo ALLDONE
pause
