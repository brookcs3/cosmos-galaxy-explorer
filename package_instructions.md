# COSMOS Galaxy Explorer — Packaging Instructions

The app bundles a standalone Python environment so end users don't need to install Python, pip, or any dependencies.

## Quick Reference

| Component | macOS | Windows |
|-----------|-------|---------|
| Python env | `staging_python_mac/` | auto-downloaded by build script |
| Backend | `staging_backend/` | `staging_backend/` |
| Build script | `./build_macos.sh` | `.\build_windows_full.ps1` |
| Output | `CosmosGalaxy.app/` | `build_windows/` |

## 1. Prepare the Backend (Both OS)

Create `staging_backend/` with the analysis script and ONNX models:

```
staging_backend/
├── analyze_single.py
├── dict.json
├── class_to_taxonomy.json
└── models/
    └── models/
        ├── model_v5s.onnx          (35MB — instrument classifier)
        ├── crest.onnx              (4MB)
        ├── wet_dry.onnx            (4MB)
        ├── sat_clean.onnx          (4MB)
        ├── centroid.onnx           (4MB)
        ├── by_inst.onnx            (4MB)
        ├── wet_dry_classifier.onnx (3MB)
        ├── sat_clean_classifier.onnx (3MB)
        └── os_loop_classifier.onnx (3MB)
```

Source files are at: `ml-ops/galaxy-semantic-backbone/`

Note: `analyze_single.py` resolves model paths relative to its own location (`../models/models/`). When copying to staging_backend, adjust the `MODELS_DIR` path in the script to use `os.path.dirname(__file__) + "/models/models/"` so it works from the bundled location.

## 2. macOS Build

### Prepare bundled Python:
```bash
# Download standalone Python 3.12 (indygreg builds)
curl -LO https://github.com/indygreg/python-build-standalone/releases/download/20241206/cpython-3.12.8+20241206-aarch64-apple-darwin-install_only_stripped.tar.gz
mkdir staging_python_mac
tar xzf cpython-*.tar.gz -C staging_python_mac --strip-components=1

# Install deps into bundled Python
./staging_python_mac/bin/pip install numpy librosa onnxruntime
```

### Build:
```bash
chmod +x build_macos.sh
./build_macos.sh
```

Output: `CosmosGalaxy.app/` — double-click to run or `open CosmosGalaxy.app`

## 3. Windows Build

Run from **Developer PowerShell for VS 2022**:

```powershell
.\build_windows_full.ps1
```

The script automatically downloads GLFW, GLEW, GLM, Python embeddable, and installs numpy + librosa + onnxruntime. You only need to provide `staging_backend/`.

Output: `build_windows/` folder with `cosmos.exe` and all deps.

### Optional: Create installer
Install [Inno Setup](https://jrsoftware.org/isinfo.php), open `installer.iss`, compile → `CosmosGalaxy_Setup.exe`

## 4. Python Dependencies

The bundled Python needs exactly these packages:
- `numpy` — numerical arrays
- `librosa` — audio loading + mel spectrogram
- `onnxruntime` — ONNX model inference

Total size: ~150-200MB (much smaller than torch-based projects)

## Without Bundled Python

If `staging_python_mac/` or `staging_backend/` are missing, the app still works — the 44K point cloud, all filtering, audio playback, and UI work fine. Only the "+ Add File" injection feature is disabled (it falls back to looking for system `python3`).
