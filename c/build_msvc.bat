@echo off
REM build_msvc.bat — build colibri C engine on native Windows with MSVC
REM Usage: call build_msvc.bat [target]
REM   target: glm (default), olmoe, iobench, all, clean

setlocal

REM --- Initialize MSVC environment ---
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to initialize MSVC environment.
    echo Install "Desktop development with C++" workload in Visual Studio 2022 Build Tools.
    exit /b 1
)

REM --- Compiler flags ---
set CFLAGS=/std:c17 /openmp /arch:AVX2 /D_FILE_OFFSET_BITS=64 /experimental:c11atomics /W3

REM --- Default target ---
if "%~1"=="" set TARGET=glm
if "%~1"=="clean" goto clean
if "%~1"=="all" goto build_all

echo Building %~1...

if "%~1"=="glm" (
    cl %CFLAGS% /Fe:glm.exe glm.c /link /machine:x64
    if errorlevel 1 exit /b 1
    echo Built glm.exe
) else if "%~1"=="olmoe" (
    cl %CFLAGS% /Fe:olmoe.exe olmoe.c /link /machine:x64
    if errorlevel 1 exit /b 1
    echo Built olmoe.exe
) else if "%~1"=="iobench" (
    cl %CFLAGS% /Fe:iobench.exe iobench.c /link /machine:x64
    if errorlevel 1 exit /b 1
    echo Built iobench.exe
) else (
    echo Unknown target: %~1
    echo Usage: build_msvc.bat [glm|olmoe|iobench|all|clean]
    exit /b 1
)
goto :eof

:build_all
cl %CFLAGS% /Fe:glm.exe glm.c /link /machine:x64
if errorlevel 1 goto :eof
echo Built glm.exe

cl %CFLAGS% /Fe:olmoe.exe olmoe.c /link /machine:x64
if errorlevel 1 goto :eof
echo Built olmoe.exe

cl %CFLAGS% /Fe:iobench.exe iobench.c /link /machine:x64
if errorlevel 1 goto :eof
echo Built iobench.exe
goto :eof

:clean
del /q glm.exe olmoe.exe iobench.exe *.obj 2>nul
echo Cleaned.
goto :eof
