# COSMOS Galaxy Explorer

Native C++/OpenGL port of the COSMOS Galaxy Explorer — a real-time interactive visualization of ~44,000 audio samples mapped into perceptual feature space using conditioned Variational Autoencoders.

Each point in the galaxy represents an audio sample positioned by its sonic characteristics across 5 perceptual dimensions. Click any point to hear it. Drag across the cloud to scrub-play samples in sequence.

## Features

- **44,280-point cloud** rendered as a GPU-accelerated 2D scatter plot with family-based coloring (12 instrument families)
- **5 VAE head views** with animated transitions between perceptual dimensions:
  - **Crest** — Transient character (punchy ↔ sustained)
  - **Wet/Dry** — Effect amount (dry ↔ ambient)
  - **Sat/Clean** — Saturation (warm ↔ clean)
  - **Centroid** — Spectral brightness (dark ↔ bright)
  - **Instrument** — Instrument family embedding
- **GPU color picking** — hover and click detection via FBO with unique color IDs per point
- **Drag-to-scrub audio** — click and drag across points to play samples sequentially
- **Live sample injection** — "+ Add File" button runs audio through the ONNX analysis pipeline and hot-loads the result as a magenta marker point
- **ImGui UI** — search, instrument/tag filtering, family legend, detail panel with all head coordinates
- **Custom cursor** with spring-physics pulsating glow effect

## Screenshot

![COSMOS Galaxy Explorer](https://user-images.githubusercontent.com/placeholder/cosmos-screenshot.png)

## Requirements

- macOS (tested on Apple Silicon M4 Max)
- OpenGL 4.1 (Metal backend on macOS)
- [Homebrew](https://brew.sh) packages:

```bash
brew install glfw glew glm
```

For the live sample injection feature ("+ Add File"):
- Python 3 with `librosa`, `numpy`, `onnxruntime`
- The ONNX model files from the COSMOS analysis pipeline (9 models, ~64MB total)

## Build & Run

```bash
make
./cosmos
```

## Controls

| Input | Action |
|-------|--------|
| **1–5** | Switch VAE head view (animated transition) |
| **Scroll** | Zoom in/out |
| **Drag background** | Pan camera |
| **Click point** | Select & play audio |
| **Drag on point** | Scrub-play (plays each point the cursor crosses) |
| **O** | Toggle one-shots visibility |
| **L** | Toggle loops visibility |
| **Esc** | Reset view, clear filters, deselect |
| **H** | Print controls to terminal |
| **Q** | Quit |

## Architecture

```
sample.cpp          — Main application (~1900 lines, single-file)
shaders/
  pointcloud.vert   — Point cloud vertex shader
  pointcloud.frag   — Point cloud fragment shader (family color lookup)
  picking.vert      — GPU picking vertex shader
  picking.frag      — GPU picking fragment shader (unique color per point)
data/
  galaxy_viz_data.json  — 44,280 samples with 5-head coordinates
  galaxy_tags.json      — 154 tags mapped to samples
lib/
  imgui*            — Dear ImGui (immediate-mode GUI)
  json.hpp          — nlohmann/json (header-only JSON parser)
  miniaudio.h       — miniaudio (header-only audio playback)
Makefile            — Build with g++ -std=c++17
```

## Data Format

Each sample in `galaxy_viz_data.json` is a 20-element array:

```
[cosmos_id, name, instrument, family, type(1=oneshot/2=loop),
 crest_x, crest_y, crest_c,
 wetdry_x, wetdry_y, wetdry_c,
 satclean_x, satclean_y, satclean_c,
 centroid_x, centroid_y, centroid_c,
 instrument_x, instrument_y,
 wav_path]
```

Coordinates are produced by conditioned VAEs trained on mel spectrograms with perceptual conditioning scalars. The instrument head uses a 13-dimensional family vector as conditioning input.

## Live Sample Injection

The "+ Add File" button opens a native file picker, then runs `analyze_single.py` which:

1. Extracts a mel spectrogram (8192 bands via librosa)
2. Classifies instrument family (model_v5s.onnx, 122 sigmoid outputs → 13 family vector)
3. Runs 3 classifier heads (wet/dry, sat/clean, oneshot/loop)
4. Runs 5 conditioned VAEs to produce (x, y) galaxy coordinates per head
5. Returns JSON which is parsed and injected as a live magenta point

Injected points are in-memory only — they are not written to the data file.

## Dependencies (vendored in lib/)

- [Dear ImGui](https://github.com/ocornut/imgui) v1.90+ — Immediate-mode GUI
- [nlohmann/json](https://github.com/nlohmann/json) — Header-only JSON
- [miniaudio](https://github.com/mackron/miniaudio) — Header-only audio

## License

Internal tool — not for redistribution.
