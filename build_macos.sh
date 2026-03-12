#!/bin/bash
# build_macos.sh — Build and package COSMOS Galaxy Explorer as a macOS .app bundle
# Prerequisites: brew install glfw glew glm
# For full packaging: staging_python_mac/ with numpy, librosa, onnxruntime installed

set -e

APP_NAME="CosmosGalaxy"
APP_DIR="${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

echo "============================================"
echo "  COSMOS Galaxy Explorer — macOS Build"
echo "============================================"

# 1. Create .app directory structure
echo ""
echo "[1/5] Creating app bundle structure..."
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}/shaders"
mkdir -p "${RESOURCES_DIR}/data"

# 2. Compile
echo "[2/5] Compiling..."

IMGUI_SOURCES="lib/imgui.cpp lib/imgui_draw.cpp lib/imgui_tables.cpp lib/imgui_widgets.cpp lib/imgui_impl_glfw.cpp lib/imgui_impl_opengl3.cpp"

# Try to static-link GLFW for portability
GLFW_STATIC="/opt/homebrew/lib/libglfw3.a"
GLEW_STATIC="/opt/homebrew/lib/libGLEW.a"

if [ -f "$GLFW_STATIC" ] && [ -f "$GLEW_STATIC" ]; then
    echo "  Using static GLFW + GLEW"
    g++ -std=c++17 -O2 -Wall -Wno-deprecated -Wno-unused-function \
        -I/opt/homebrew/include -Ilib \
        -DGL_SILENCE_DEPRECATION \
        sample.cpp ${IMGUI_SOURCES} \
        "${GLFW_STATIC}" "${GLEW_STATIC}" \
        -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
        -framework CoreAudio -framework AudioToolbox -framework CoreFoundation \
        -o "${MACOS_DIR}/cosmos"
else
    echo "  Static libs not found, using dynamic linking"
    g++ -std=c++17 -O2 -Wall -Wno-deprecated -Wno-unused-function \
        -I/opt/homebrew/include -Ilib -L/opt/homebrew/lib \
        -DGL_SILENCE_DEPRECATION \
        sample.cpp ${IMGUI_SOURCES} \
        -lglfw -lGLEW \
        -framework OpenGL \
        -framework CoreAudio -framework AudioToolbox -framework CoreFoundation \
        -o "${MACOS_DIR}/cosmos"
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi
echo "  Compiled successfully"

# 3. Copy resources
echo "[3/5] Copying resources..."
cp shaders/*.vert shaders/*.frag "${RESOURCES_DIR}/shaders/"
cp data/*.json "${RESOURCES_DIR}/data/"

# 4. Create Info.plist
cat > "${CONTENTS_DIR}/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>cosmos</string>
    <key>CFBundleIdentifier</key>
    <string>com.cosmos.galaxy-explorer</string>
    <key>CFBundleName</key>
    <string>COSMOS Galaxy Explorer</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# 5. Bundle Python + backend (if staging dirs exist)
echo "[4/5] Bundling Python environment and backend..."

if [ -d "staging_python_mac" ]; then
    echo "  Copying bundled Python..."
    cp -R staging_python_mac "${RESOURCES_DIR}/python"
else
    echo "  WARNING: staging_python_mac/ not found"
    echo "  The '+ Add File' feature requires a bundled Python."
    echo "  See package_instructions.md for setup."
fi

if [ -d "staging_backend" ]; then
    echo "  Copying backend (analyze_single.py + ONNX models)..."
    cp -R staging_backend "${RESOURCES_DIR}/backend"
else
    echo "  WARNING: staging_backend/ not found"
    echo "  The '+ Add File' feature requires the analysis backend."
    echo "  See package_instructions.md for setup."
fi

# 6. Ad-hoc codesign
echo "[5/5] Code signing..."
codesign --force --deep -s - "${APP_DIR}" 2>/dev/null || echo "  codesign skipped (not required for local use)"

# Summary
echo ""
echo "============================================"
TOTAL_SIZE=$(du -sh "${APP_DIR}" | cut -f1)
echo "  BUILD COMPLETE"
echo "  Output: ./${APP_DIR}"
echo "  Size: ${TOTAL_SIZE}"
echo "============================================"
echo ""
echo "To run:  open ./${APP_DIR}"
echo "To distribute: zip -r ${APP_NAME}.zip ${APP_DIR}"
