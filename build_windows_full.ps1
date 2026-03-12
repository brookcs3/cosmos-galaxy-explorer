# build_windows_full.ps1
# One-shot Windows build for COSMOS Galaxy Explorer
# Prerequisites: Visual Studio Build Tools with C++ workload
# Run from: Developer PowerShell for VS 2022
#
# Usage: .\build_windows_full.ps1

$ErrorActionPreference = "Stop"
$APP_NAME = "cosmos"
$BUILD_DIR = "build_windows"
$DEPS_DIR = "deps_windows"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  COSMOS Galaxy Explorer — Windows Build" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- Verify cl.exe is available ---
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    Write-Host "ERROR: cl.exe not found. Run this from 'Developer PowerShell for VS 2022'." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] cl.exe found: $($cl.Source)" -ForegroundColor Green

# --- Create directories ---
New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $DEPS_DIR | Out-Null

# ============================================================
# STEP 1: Download GLFW (prebuilt MSVC)
# ============================================================
$GLFW_DIR = "$DEPS_DIR\glfw"
if (-not (Test-Path "$GLFW_DIR\lib-vc2022\glfw3.lib")) {
    Write-Host "`n[1/7] Downloading GLFW 3.4..." -ForegroundColor Yellow
    $glfwUrl = "https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip"
    $glfwZip = "$DEPS_DIR\glfw.zip"
    Invoke-WebRequest -Uri $glfwUrl -OutFile $glfwZip
    Expand-Archive -Path $glfwZip -DestinationPath $DEPS_DIR -Force
    $extracted = Get-ChildItem "$DEPS_DIR\glfw-*" -Directory | Select-Object -First 1
    if ($extracted) {
        if (Test-Path $GLFW_DIR) { Remove-Item -Recurse -Force $GLFW_DIR }
        Rename-Item $extracted.FullName $GLFW_DIR
    }
    Remove-Item $glfwZip -Force
    Write-Host "[OK] GLFW ready" -ForegroundColor Green
} else {
    Write-Host "`n[1/7] GLFW already downloaded" -ForegroundColor Green
}

# ============================================================
# STEP 2: Download GLEW (prebuilt)
# ============================================================
$GLEW_DIR = "$DEPS_DIR\glew"
if (-not (Test-Path "$GLEW_DIR\lib\Release\x64\glew32.lib")) {
    Write-Host "`n[2/7] Downloading GLEW 2.2.0..." -ForegroundColor Yellow
    $glewUrl = "https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0-win32.zip"
    $glewZip = "$DEPS_DIR\glew.zip"
    Invoke-WebRequest -Uri $glewUrl -OutFile $glewZip
    Expand-Archive -Path $glewZip -DestinationPath $DEPS_DIR -Force
    $extracted = Get-ChildItem "$DEPS_DIR\glew-*" -Directory | Select-Object -First 1
    if ($extracted) {
        if (Test-Path $GLEW_DIR) { Remove-Item -Recurse -Force $GLEW_DIR }
        Rename-Item $extracted.FullName $GLEW_DIR
    }
    Remove-Item $glewZip -Force
    Write-Host "[OK] GLEW ready" -ForegroundColor Green
} else {
    Write-Host "`n[2/7] GLEW already downloaded" -ForegroundColor Green
}

# ============================================================
# STEP 3: Download GLM (header-only)
# ============================================================
$GLM_DIR = "$DEPS_DIR\glm"
if (-not (Test-Path "$GLM_DIR\glm\glm.hpp")) {
    Write-Host "`n[3/7] Downloading GLM 1.0.1..." -ForegroundColor Yellow
    $glmUrl = "https://github.com/g-truc/glm/releases/download/1.0.1/glm-1.0.1-light.zip"
    $glmZip = "$DEPS_DIR\glm.zip"
    Invoke-WebRequest -Uri $glmUrl -OutFile $glmZip
    Expand-Archive -Path $glmZip -DestinationPath $DEPS_DIR -Force
    Remove-Item $glmZip -Force
    Write-Host "[OK] GLM ready" -ForegroundColor Green
} else {
    Write-Host "`n[3/7] GLM already downloaded" -ForegroundColor Green
}

# ============================================================
# STEP 4: Download Python Embeddable Package
# ============================================================
$PYTHON_DIR = "staging_python_win"
if (-not (Test-Path "$PYTHON_DIR\python.exe")) {
    Write-Host "`n[4/7] Downloading Python 3.12 embeddable..." -ForegroundColor Yellow
    $pyUrl = "https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip"
    $pyZip = "$DEPS_DIR\python-embed.zip"
    Invoke-WebRequest -Uri $pyUrl -OutFile $pyZip
    New-Item -ItemType Directory -Force -Path $PYTHON_DIR | Out-Null
    Expand-Archive -Path $pyZip -DestinationPath $PYTHON_DIR -Force
    Remove-Item $pyZip -Force

    # Enable pip: uncomment 'import site' in python312._pth
    $pthFile = Get-ChildItem "$PYTHON_DIR\python*._pth" | Select-Object -First 1
    if ($pthFile) {
        $content = Get-Content $pthFile.FullName
        $content = $content -replace '#import site', 'import site'
        Set-Content $pthFile.FullName $content
    }

    # Install pip
    Write-Host "  Installing pip..." -ForegroundColor Gray
    $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
    $getPipPath = "$DEPS_DIR\get-pip.py"
    Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath
    & "$PYTHON_DIR\python.exe" $getPipPath --no-warn-script-location 2>&1 | Out-Null
    Remove-Item $getPipPath -Force
    Write-Host "[OK] Python embeddable ready" -ForegroundColor Green
} else {
    Write-Host "`n[4/7] Python embeddable already set up" -ForegroundColor Green
}

# ============================================================
# STEP 5: Install Python dependencies
# ============================================================
Write-Host "`n[5/7] Installing Python deps (numpy, librosa, onnxruntime)..." -ForegroundColor Yellow
& "$PYTHON_DIR\python.exe" -m pip install numpy librosa onnxruntime --no-warn-script-location 2>&1 | ForEach-Object {
    if ($_ -match "Successfully installed|already satisfied") { Write-Host "  $_" -ForegroundColor Gray }
}

# Verify imports
Write-Host "  Verifying imports..." -ForegroundColor Gray
& "$PYTHON_DIR\python.exe" -c "import numpy; import librosa; import onnxruntime; print('All deps OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python dependency verification failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] All Python deps installed and verified" -ForegroundColor Green

# ============================================================
# STEP 6: Compile C++ Code
# ============================================================
Write-Host "`n[6/7] Compiling COSMOS..." -ForegroundColor Yellow

$GLFW_INC = "$GLFW_DIR\include"
$GLFW_LIB = "$GLFW_DIR\lib-vc2022"
$GLEW_INC = "$GLEW_DIR\include"
$GLEW_LIB = "$GLEW_DIR\lib\Release\x64"
$GLM_INC = "$GLM_DIR"

$IMGUI_SOURCES = @(
    "lib\imgui.cpp",
    "lib\imgui_draw.cpp",
    "lib\imgui_tables.cpp",
    "lib\imgui_widgets.cpp",
    "lib\imgui_impl_glfw.cpp",
    "lib\imgui_impl_opengl3.cpp"
)

cl.exe /std:c++17 /EHsc /O2 /MT `
    /I"$GLFW_INC" /I"$GLEW_INC" /I"$GLM_INC" /I"lib" `
    sample.cpp $IMGUI_SOURCES `
    /link /LIBPATH:"$GLFW_LIB" /LIBPATH:"$GLEW_LIB" `
    glfw3.lib glew32.lib opengl32.lib gdi32.lib user32.lib shell32.lib comdlg32.lib `
    /OUT:"$BUILD_DIR\$APP_NAME.exe"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Compilation failed!" -ForegroundColor Red
    exit 1
}
Remove-Item *.obj -ErrorAction SilentlyContinue
Write-Host "[OK] Compiled successfully" -ForegroundColor Green

# ============================================================
# STEP 7: Assemble distribution folder
# ============================================================
Write-Host "`n[7/7] Assembling distribution..." -ForegroundColor Yellow

# Copy GLEW DLL (needed at runtime if dynamically linked)
if (Test-Path "$GLEW_DIR\bin\Release\x64\glew32.dll") {
    Copy-Item "$GLEW_DIR\bin\Release\x64\glew32.dll" "$BUILD_DIR\" -Force
}

# Copy shaders
New-Item -ItemType Directory -Force -Path "$BUILD_DIR\shaders" | Out-Null
Copy-Item "shaders\*" "$BUILD_DIR\shaders\" -Force

# Copy data
New-Item -ItemType Directory -Force -Path "$BUILD_DIR\data" | Out-Null
Copy-Item "data\*" "$BUILD_DIR\data\" -Force

# Copy Python
if (Test-Path "$BUILD_DIR\python") { Remove-Item -Recurse -Force "$BUILD_DIR\python" }
Copy-Item -Recurse -Force $PYTHON_DIR "$BUILD_DIR\python"

# Copy backend (analyze_single.py + models)
if (Test-Path "staging_backend") {
    if (Test-Path "$BUILD_DIR\backend") { Remove-Item -Recurse -Force "$BUILD_DIR\backend" }
    Copy-Item -Recurse -Force "staging_backend" "$BUILD_DIR\backend"
    Write-Host "  Backend copied" -ForegroundColor Gray
} else {
    Write-Host "  WARNING: staging_backend/ not found. '+ Add File' will use system Python." -ForegroundColor Yellow
}

Write-Host "[OK] Distribution assembled" -ForegroundColor Green

# --- Summary ---
$totalSize = (Get-ChildItem -Recurse "$BUILD_DIR" | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  BUILD COMPLETE" -ForegroundColor Cyan
Write-Host "  Output: .\$BUILD_DIR\" -ForegroundColor Cyan
Write-Host "  Size: $([math]::Round($totalSize, 0)) MB" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "`nTo create an installer, open installer.iss in Inno Setup and compile it."
Write-Host "Or just zip the $BUILD_DIR folder and distribute directly."
