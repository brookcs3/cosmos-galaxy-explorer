# Technical Handoff Report: Project AI-STEM-Separator (Mad Scientist Edition)

### 1. The 5.5-Phase Gated Pipeline Architecture

Project AI-STEM-Separator implements a **Decision-Based Generative Architecture**. Unlike legacy subtractive source separation (e.g., Demucs, Spleeter) which uses masking to filter audio, this pipeline utilizes initial separations as a "structural blueprint." We treat the output of the initial stage as a degraded prior, using conditional latent diffusion to "hallucinate" pristine, phase-coherent audio from scratch.

The architecture is deployed as a modular, 5.5-phase gated pipeline:

- **Phase 1 (The Anchor): SCNet Separator.** A discriminative prior that blindly decomposes the stereo mix into the "Essential Six" fixed energy buckets. This provides the comprehensive structural map for the generative process.
- **Phase 2 (The Compressor): Audio VAE Encoder.** A continuous 1D Variational Autoencoder (DAC-style) that compresses 44.1kHz stereo audio 2048x into phase-aware latents (64-channel). This prevents phase collapse and preserves the physical timbre of the source.
- **Phase 2.5 (The Forensic Scanner): Semantic Router.** The "Gatekeeper" and **Semantic Spine**. It embeds user prompts via T5 and scans the latent physics of all 6 stems to identify the "best harmonic candidate."
- **Phase 3 (The Guide): LiLAC ControlNet.** A lightweight adapter (Lightweight Latent ControlNet) that wraps frozen DiT blocks. It extracts structural features (phrasing, transient attacks, pitch bends) from the Phase 1 latent to enforce the original musical "groove."
- **Phase 4 (The Brain): DiT Generator.** A 1.1B parameter Diffusion Transformer that "hallucinates" pristine texture onto the structural skeleton provided by LiLAC. It uses Flow Matching and Global RoPE.
- **Phase 5 (The Voice): VAE Decoder.** A HiFi-GAN style neural vocoder that converts hallucinated latents back into transparent, high-fidelity 44.1kHz stereo waveforms.

**The "5.5" Designation:** Phase 2.5 is not a sequential gate but a non-linear **Semantic Spine**. It is the only component that handles text embeddings, building the semantic representation consumed by Phase 3 (for structural feature weighting) and Phase 4 (as cross-attention keys for acoustic conditioning).

\--------------------------------------------------------------------------------

### 2. The 'Essential Six' Structural Invariant

To unify heterogeneous datasets (MUSDB18-HQ and MoisesDB), we enforce a strict 6-stem ontology. This contract ensures a fixed input/output tensor shape across the entire pipeline.

| Slot       | Stem Category | Source Data Mapping                 | Loss Masking Rule              |
| ---------- | ------------- | ----------------------------------- | ------------------------------ |
| **Slot 0** | Drums         | MoisesDB: drums, a-tonal percussion | Ground Truth (No Masking)      |
| **Slot 1** | Bass          | MoisesDB: bass                      | Ground Truth (No Masking)      |
| **Slot 2** | Guitar        | MoisesDB: guitar                    | Zero-Padding + Masking (MUSDB) |
| **Slot 3** | Piano         | MoisesDB: piano (grand)             | Zero-Padding + Masking (MUSDB) |
| **Slot 4** | Vocals        | MoisesDB: vocals                    | Ground Truth (No Masking)      |
| **Slot 5** | Other         | MoisesDB: keys, strings, wind, etc. | Ground Truth (No Masking)      |

**Ontology Logic:**

- **MoisesDB Collapse:** Over 30 granular tags are programmatically collapsed into these six categories.
- **MUSDB18 Expansion:** Standard 4-stem data is expanded to 6. Missing slots (Guitar/Piano) are zero-padded.
- **Silence vs. Absence:** **No Loss Masking is used for naturally occurring silence.** Silence is a valid "True Negative" signal. Masking is reserved exclusively for "Missing Labels" (e.g., when a guitar is in the mix but no isolated stem exists in the source dataset).

\--------------------------------------------------------------------------------

### 3. Phase 1: SCNet Modernization & Ensemble Strategy

Phase 1 employs a modernized SCNet (Sparse Compression Network) within a specialized ensemble strategy to maximize fidelity across different instrument domains:

- The "Surgical Stitch" Ensemble:
  - **SCNet-6s:** Used for **rhythm-focused stems** (Drums, Bass) to maintain phase coherence and low-end transient punch.
  - **BSRoformer (Fallback):** Leveraged for **melodic stems** (Vocals, Guitar) to ensure superior mid-range harmonic resolution.
- **Boolean Silence Gate:** Implemented via a `WeightedRandomSampler` and `is_silent` metadata. This ensures balanced batch representation, preventing the model from "forgetting" how to generate underrepresented stems like Piano or Guitar.
- Modernized Training Parameters:
  - **N_FFT:** **4096**
  - **Hop Length:** **1024**
  - **Sample Rate:** **44100 Hz**
  - **Input:** Stereo Complex Spectrograms (4-channel: L-Real, L-Imag, R-Real, R-Imag).

\--------------------------------------------------------------------------------

### 4. Phase 2: Audio VAE Training Protocol

The VAE serves as our codec, trained with a **"Tabula Rasa"** approach. Fine-tuning from Stable Audio Open was rejected to optimize the latent space specifically for our 6-stem resynthesis requirements.

- **Milestone A (0–460K steps):** Joint encoder+decoder training using Multi-Resolution STFT (MR-STFT) and KL loss (1e-4 weight).
- **Milestone B (460K–1.1M steps):** Frozen encoder. Decoder-only adversarial training using Multi-Period (MPD) and Multi-Scale (MSD) discriminators to achieve perceptual "air" and high-frequency realism.
- Compression & Bias:
  - **Ratio:** 2048x (Immutable strides: [4, 8, 8, 8]).
  - **Snake Activations:** Periodic inductive bias to preserve phase info. **Alpha initialization is set to 5–50**specifically for audio, deviating from the standard 0.5.

\--------------------------------------------------------------------------------

### 5. Phase 2.5: Waves COSMOS (The Semantic Spine)

Phase 2.5 acts as the **"Rosetta Stone"** of the project, providing a supervised bridge between latents and semantics. We utilize 40,000 matched pairs where Phase 2 VAE latents are mapped directly to metadata from the Waves COSMOS database.

**The 14-Float Semantic Coordinate Schema:** The Galaxy multi-head encoder provides acoustic grounding through five parallel VAE-style heads. Each head (except Instrument) outputs (x, y, c) where 'c' represents confidence/KL-divergence.

1. **Crest:** Transient character (punchy vs. sustained) — 3 floats.
2. **Wet/Dry:** Reverb amount — 3 floats.
3. **Sat/Clean:** Saturation/warmth — 3 floats.
4. **Centroid:** Spectral brightness — 3 floats.
5. **Instrument:** 13-family classification — 2 floats (x, y only).

**Triple Training Signal:** COSMOS metadata informs the pipeline at three critical points:

- **Phase 2.5:** Direct label ↔ latent alignment.
- **Phase 3:** Structural feature weighting (weighting LiLAC based on instrument family).
- **Phase 4:** DiT conditioning (teaching the generator exactly "what a harmonica sounds like" in terms of centroid and crest).

**Standalone COSMOS Feature Extractor:** A self-contained Python module at `ml-ops/phase-2_5_router/cosmos_feature_extractor/` replicates the full COSMOS analysis pipeline outside of the Waves app. It bundles all 9 ONNX models (~67MB), `dict.json` (122-class instrument mapping), and the enrichment SQLite (~40MB) locally — no dependency on the Waves install path or parent directory traversal.

The module runs any WAV through the complete pipeline: `process_data` → `classify_instrument` (model_v5s.onnx, 122 classes → 13 families) → `run_classifiers` (3 binary classifiers) → `run_vae_heads` (5 VAE encoders) → JSON output with all 14 galaxy floats + instrument classification. Invoked via `python -m cosmos_feature_extractor <wav_path>` or programmatically via `from cosmos_feature_extractor import analyze`.

```
cosmos_feature_extractor/
├── models/          ← all ONNX models + enrichment DB bundled
├── __init__.py      ← public API
├── __main__.py      ← CLI entry point
├── constants.py     ← feature vector layout, model paths
├── extractor.py     ← process_data, classifiers, VAE heads, analyze()
├── instrument.py    ← classify_instrument, family vector mapping
└── PIPELINE.md      ← ASCII flowchart of the full pipeline
```

\--------------------------------------------------------------------------------

### 6. The Data Model: Manifest vs. Sidecar

We maintain a strict boundary between the codec's operational data and the semantic enrichment layer.

| Feature         | Slim Phase 2 Manifest (Codec)               | Rich COSMOS Sidecar (Semantic)                               |
| --------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Primary Use** | VAE Dataloader / Audio Training             | Router / DiT Conditioning                                    |
| **Join Key**    | **cosmos_id** (sample_t._id)                | **cosmos_id** (Primary Join Key)                             |
| **Key Fields**  | `file_path`, `category`, `duration`, `acrc` | `instrument_id`, `galaxy_*` coords (14), `tags`, `bpm`, `key`, `scale` |

**Amendment Pattern:** Data quality is enforced via `COALESCE(amended_X, X)`, prioritizing human corrections.

- *Example:* A "Cello Sustained B3" sample classified as "Violin" by ML is corrected in the sidecar. The pipeline will automatically prefer the human "Cello" label for training the semantic spine.

\--------------------------------------------------------------------------------

### 7. Hardware Configuration: The Novus Cluster (GH200)

Training is executed on the **OSU Novus HPC cluster** using **NVIDIA GH200 Grace Hopper Superchips**.

- **Specifications:** 480GB Unified Memory (96GB HBM3 + 384GB LPDDR5X).
- **Software Environment:** NVIDIA NGC Containers (`nvidia/pytorch:24.12-py3`) deployed via **enroot/pyxis**.
- **Critical Memory Flag:** `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` is required to enable unified memory spillover.
- **Workflow Discovery:** **Slurm commands (sbatch/srun) are host-only.** They must be executed from the cluster login node, not from within the active container environment.
- **ABI Compatibility:** `torchaudio` must be built from source inside the NGC 24.12 container to ensure compatibility with the pre-installed PyTorch version.

\--------------------------------------------------------------------------------

### 8. Current Sprint Status & Completion Outlook

As of the March 2026 session logs, we are in **Sprint 2**.

[!CAUTION] **Status Alert: Phase 1 Performance** Phase 1 (SCNet) has reached **Epoch 230** with a best SDR of **-0.9368**. This is currently **0.3 dB behind** the historical peak of **-0.6124** (Epoch ~166). Training is continuing to recover and exceed this baseline.



EDIT NEW BEST IS  -.04

\--------------------------------------------------------------------------------

### 9. Critical AI Operational Corrections

All engineers must adhere to the following **Operational Directives**:

- **Verify, Don't Assert:** Never state file locations or system states from memory. Always perform a directory check or verification first.
- **Naming Convention (The "DIRTY" Rule):** The term **"DIRTY" (Category 4)** is strictly reserved for offline degraded training data. Never call live Phase 1 candidate stems "DIRTY." Refer to them as **CATEGORY V** or **STEM_PHASE1_OUT**.
- **Silence is Signal:** Do not use loss masking for silent stems. The model must learn silence as a valid state via the Boolean Silence Gate.
- **Container Integrity:** Always build `torchaudio` from source when using the NGC 24.12 container to prevent ABI version mismatches.
- **Destructive Ops:** Never `rm -rf` a directory without verifying alternative copies. If it is the only copy, it is "precious." Use `mv` where possible.