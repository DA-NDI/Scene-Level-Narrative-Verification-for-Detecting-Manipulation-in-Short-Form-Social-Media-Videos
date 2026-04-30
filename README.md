# Scene-Level Narrative Verification for Detecting Manipulation in Short-Form Social Media Videos

**Master's Thesis** | Ukrainian Catholic University | Andrii Hupalo, 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code, results, and documentation for detecting **narrative manipulation (cheapfakes)** in short-form social media videos through **scene-level cross-modal consistency verification**.

**Key Contribution:** A modular CV+LLM pipeline combining:
- Scene segmentation with adaptive thresholding
- Multimodal feature extraction (OCR, ASR, visual alignment)
- Zero-shot reasoning via multimodal LLMs (Gemini, Qwen)
- Pixel-level + narrative-level fusion for robust detection

### Results Summary

| Model | Dataset | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| **Gemini 3.1-flash-lite (Config F)** | FakeTT | 0.600 | 0.558 | 0.960 | **0.706** |
| **Gemini 3.1-flash-lite (Config F)** | FMNV | 0.643 | 0.857 | 0.500 | 0.632 |
| **Hybrid Fusion (Pixel + Narrative)** | FakeTT | 0.630 | 0.589 | 0.860 | 0.699 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/video-manipulation-detection.git
cd video-manipulation-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
export GEMINI_API_KEY="your-key"
export QWEN_API_KEY="your-key"  # Optional
```

### Data Setup

**1. FakeTT Dataset (43GB)**
- Download: https://github.com/ICTMCG/FakingRecipe
- Extract to: `./fakett/FakeTT_DATA_OPENSOURCE/`
- Structure:
  ```
  fakett/FakeTT_DATA_OPENSOURCE/
  ├── data.json          # Video metadata + labels
  └── video/
      ├── video_001.mp4
      ├── video_002.mp4
      └── ...
  ```

**2. FMNV Dataset (1.3GB)**
- Download: [Google Drive Link](https://drive.google.com/file/d/1w0UfaC1jjCJff5SP3uCEFHx_8PtG7biE/view)
- Extract to: `./FMNV/`
- Structure:
  ```
  FMNV/
  ├── data.json
  ├── ocr_dataset.json
  ├── caption_data.json
  └── videos/
  ```

### Run Evaluation

```bash
# FakeTT evaluation (100 balanced videos)
python fakett_pipeline_eval_vlm.py
# Output: fakett_pipeline_results_vlm.json

# FMNV evaluation
python fmnv_eval/fmnv_pipeline_eval_v2.py
# Output: fmnv_results.json

# Qwen2.5-VL comparison (100 videos)
python fakett_qwen_eval.py
# Output: fakett_qwen_results.json

# Pixel-level + narrative fusion
python deepfake/eval_fusion.py
# Output: fusion_results.json
```

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusions
│
├── thesis_ucu/                        # Master's thesis (LaTeX)
│   ├── master-thesis-template.tex     # Main document
│   ├── Thesis.cls                     # Document class
│   ├── bibliography.bib               # References
│   ├── chapters/
│   │   ├── 1_introduction.tex
│   │   ├── 2_related_work.tex
│   │   ├── 3_theoretical_background.tex
│   │   ├── 4_proposed_solution.tex
│   │   ├── 5_experiments_and_results.tex
│   │   └── 6_conclusions.tex
│   ├── appendices/
│   ├── figures/                       # Thesis figures (PNG, JPG)
│   └── COMPILATION_FILES.txt
│
├── Pipeline Code (Core)
│   ├── fakett_pipeline_eval.py        # FakeTT evaluation
│   ├── fakett_pipeline_eval_vlm.py    # FakeTT with visual descriptions
│   ├── fakett_qwen_eval.py            # Qwen2.5-VL evaluation
│   ├── ocr_clip_whisper_extraction.py # Multimodal feature extraction
│   ├── clip_cosine_sim.py             # Visual-text alignment scoring
│   ├── gemini_api_reasoning.py        # Gemini reasoning engine
│   │
│   ├── deepfake/
│   │   ├── deepfake_detector.py       # Pixel-level detection (ViT)
│   │   ├── narrative_manipulation.py  # Narrative-level detection
│   │   └── eval_fusion.py             # Hybrid fusion evaluation
│   │
│   └── fmnv_eval/
│       ├── fmnv_pipeline_eval.py      # FMNV baseline
│       ├── fmnv_pipeline_eval_v2.py   # FMNV improved
│       ├── fmnv_configs_h_i_j.py      # Configs H, I, J
│       └── run_configs.sh             # Batch runner
│
├── Results & Analysis
│   ├── ablation_results/              # FakeTT ablations (A-J)
│   │   ├── fakett_ablation_A.json
│   │   ├── fakett_ablation_F.json     # Best config (F1=0.706)
│   │   └── ...
│   ├── ablation_results_fmnv/         # FMNV results
│   ├── evaluation_results/            # Fusion evaluation
│   ├── fakett_pipeline_results_vlm.json
│   ├── COMPLETE_RESULTS_A_TO_I.md     # Summary table
│   └── threshold_sensitivity_summary.md
│
├── Utilities
│   ├── adaptivedetector.py            # Scene detection fix (threshold=3.0)
│   ├── pipeline_analysis.py           # Result analysis
│   ├── pipeline_eval.py               # Evaluation harness
│   ├── json_checker.py                # Result validation
│   └── eda_plots.py                   # Visualization
│
└── Documentation
    ├── QWEN_SETUP.md                  # Qwen setup guide
    ├── PAPERS_SUMMARY_FOR_THESIS.md
    ├── COMPREHENSIVE_THESIS_PLAN.md
    └── COMPLETE_RESULTS_A_TO_I.md     # Detailed results table
```

## Pipeline Architecture

### Stream A: Pixel-Level Detection
```
Video → ViT Face Detector → Artifact Score (0-1)
```
- High scores (0.9+) on AI-generated faces (e.g., Trump Gaza deepfake)
- Near-zero on general scene synthesis

### Stream B: Narrative-Level Cross-Modal Consistency
```
Video → Scene Detection → {OCR, ASR, Visual-Text Alignment}
      → Multimodal LLM (Gemini) → Narrative Score (0-100)
```
- 4-stage pipeline: segmentation → feature extraction → reasoning → aggregation
- Detects recontextualisation, misleading captions, audio-visual misalignment

### Fusion Module
```
M_fusion = w_a * M_pixel + w_n * M_narrative
```
- Weighted voting combines both streams
- Best F1 = 0.699 (vs Config F narrative-only = 0.706)

## Key Findings

### 1. Scene Detection Fix
**Bug:** PySceneDetect's `AdaptiveDetector` threshold misconfigured at 27.0
- **Issue:** 69% of TikTok videos returned zero detected scenes
- **Fix:** Correct threshold = 3.0
- **Result:** Zero-scene rate reduced to 21%

### 2. Model Generation vs. Capacity
- **Gemini 3.x > 2.5** regardless of parameter count
- 3.1-flash-lite (cheapest) outperforms 2.5-pro (expensive)
- F1 improvement: +5.4% generation gap

### 3. Task-Definition Mismatch
- FakeTT annotation schema (footage authenticity) ≠ pipeline target (narrative consistency)
- Mann-Whitney U=1222, p=0.51: systematic mismatch, not pipeline failure
- Motivates need for cheapfake-specific benchmarks

### 4. Modality Ablation
| Config | Modalities | F1 |
|--------|-----------|-----|
| A | Text only | 0.341 |
| B | Text + CLIP | 0.482 |
| C | Text + CLIP + Visual | **0.706** |
| F | Gemini 3.1-flash-lite | **0.706** |

## Usage Examples

### Run on FakeTT (50 real + 50 fake)
```python
python fakett_pipeline_eval_vlm.py
# Returns: Accuracy=0.600, Precision=0.558, Recall=0.960, F1=0.706
```

### Run on FMNV (explicit manipulation classes)
```python
python fmnv_eval/fmnv_pipeline_eval_v2.py
# Returns: Accuracy=0.643, Precision=0.857, Recall=0.500, F1=0.632
```

### Compare Gemini vs Qwen
```bash
# Gemini (thesis baseline)
python fakett_pipeline_eval_vlm.py

# Qwen2.5-VL (new experiment)
python fakett_qwen_eval.py

# Hybrid (pixel + narrative)
python deepfake/eval_fusion.py
```

### Customize pipeline
```python
from fakett_pipeline_eval_vlm import *

# Modify settings
N_REAL = 100
N_FAKE = 100
MAX_SCENES_FOR_ANALYSIS = 6
model = genai.GenerativeModel("gemini-pro-vision")  # Different model

# Run
main()
```

## Datasets

| Dataset | Size | Videos | License | Download |
|---------|------|--------|---------|----------|
| **FakeTT** | 43GB | 2,000+ | CC-BY-NC | [GitHub](https://github.com/ICTMCG/FakingRecipe) |
| **FMNV** | 1.3GB | 500+ | CC-BY | [Google Drive](https://drive.google.com/file/d/1w0UfaC1jjCJff5SP3uCEFHx_8PtG7biE/view) |
| **Proprietary** | ~9GB | 150 | NDA | Contact author |

## Requirements

See `requirements.txt`:
```
google-generativeai>=0.4.0
opencv-python>=4.8.0
easyocr>=1.7.0
scenedetect>=0.6.0
Pillow>=10.0.0
torch>=2.0.0
transformers>=4.30.0
torch-cv>=0.1.0
```

## Installation (Requirements)

```bash
pip install -r requirements.txt

# For GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: For Qwen local inference
pip install qwen-vl
```

## Results Files

- **`ablation_results/fakett_ablation_*.json`** — Config A-J results (11 experiments)
- **`ablation_results_fmnv/fmnv_*.json`** — FMNV F, G results
- **`evaluation_results/Fusion_3.1_Lite_results.json`** — Hybrid fusion results
- **`fakett_pipeline_results_vlm.json`** — Best FakeTT config (F1=0.706)

## Citation

```bibtex
@mastersthesis{hupalo2026scene,
  author   = {Hupalo, Andrii},
  title    = {Scene-Level Narrative Verification for Detecting Manipulation 
             in Short-Form Social Media Videos},
  school   = {Ukrainian Catholic University},
  year     = {2026},
  address  = {Lviv, Ukraine},
  note     = {Faculty of Applied Sciences}
}
```

## References

- **FakeForensics++**: Rössler et al., ICCV 2019
- **FakeTT**: Bu et al., ACM MM 2024
- **FakeSV-VLM**: Wang et al., arXiv 2024
- **FakingRecipe**: Bu et al., ACM MM 2024 (creative process modeling)
- **CLIP**: Radford et al., ICML 2021
- **Whisper**: Radford et al., arXiv 2022

## License

MIT License — See LICENSE file

## Author

**Andrii Hupalo**
- Email: hupalo.pn@ucu.edu.ua
- Affiliation: Ukrainian Catholic University, Faculty of Applied Sciences
- Supervisor: Philip Shurpik (LetsData, Ukraine)

## Acknowledgments

- Philip Shurpik (thesis supervision & LetsData infrastructure)
- Anonymous industry partner (proprietary TikTok dataset)
- StopFake & VoxCheck (fact-checking organizations using this work)
- UCU Faculty of Applied Sciences

## Contributing

This is an academic thesis repository. For questions or corrections:
1. Open an issue with detailed description
2. Include which dataset/experiment is affected
3. Provide reproduction steps if applicable

## Related Work

- **Thesis Document**: See `thesis_ucu/master-thesis-template.pdf`
- **Symposium Slides**: See `master_seminar/` directory
- **Paper**: FakingRecipe (https://github.com/ICTMCG/FakingRecipe)
- **Community**: Join discussions at https://github.com/ICTMCG/FakeVideoForensics

---

**Last Updated**: April 30, 2026  
**Thesis Status**: Submitted to UCU  
**Code Status**: Production-ready (v1.0)
