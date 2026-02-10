@echo off
REM ================================================================
REM FlyWire Connectome GPU Simulator - Build Script
REM ================================================================

where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: nvcc not found on PATH.
    echo Please install NVIDIA CUDA Toolkit from:
    echo   https://developer.nvidia.com/cuda-downloads
    exit /b 1
)

echo.
echo CUDA Compiler:
nvcc --version | findstr "release"
echo.

if "%1"=="clean" goto do_clean
if "%1"=="debug" goto do_debug
goto do_release

:do_clean
echo Cleaning build artifacts...
del /q flywire_sim.exe 2>nul
del /q flywire_sim.lib 2>nul
del /q flywire_sim.exp 2>nul
del /q flywire_sim.pdb 2>nul
del /q *.obj 2>nul
echo Done.
exit /b 0

:do_debug
echo Building DEBUG...
nvcc -g -G -lineinfo -arch=sm_86 -o flywire_sim.exe flywire_sim.cu
goto check_result

:do_release
echo Building RELEASE (optimized for RTX 3080 Ti, SM 8.6)...
nvcc -O3 -arch=sm_86 --use_fast_math -o flywire_sim.exe flywire_sim.cu
goto check_result

:check_result
if %errorlevel% neq 0 (
    echo.
    echo BUILD FAILED.
    echo If you see "unsupported gpu architecture sm_86", try sm_80.
    exit /b 1
)

echo.
echo Build successful: flywire_sim.exe
echo.
echo Run with:
echo   flywire_sim.exe
echo   flywire_sim.exe --data flywire_v783.bin
echo   flywire_sim.exe --help
echo.
