# FMNV Evaluation Suite

**Location:** `/dyplom/fmnv_eval/`

Evaluation scripts for the FMNV (Fake Multimodal News Video) dataset.

---

## 📁 Files

- **`fmnv_pipeline_eval.py`** — Gemini API evaluation (Configs A–G, Mac)
- **`fmnv_gemma_eval.py`** — Gemma-4 local evaluation (Config H, PC)
- **`FMNV_EVALUATION_PLAN.md`** — Complete guide with timelines and integration plan
- **`README.md`** (this file)

---

## 🚀 Quick Start

### On Mac (Gemini A–G)

**1. Set API key:**
```bash
export GEMINI_API_KEY="your-key-here"
```

**2. Run Config F (recommended):**
```bash
cd dyplom
python fmnv_eval/fmnv_pipeline_eval.py --config F
```

**3. Run all A–G:**
```bash
python fmnv_eval/fmnv_pipeline_eval.py --all
```

**4. Compare results:**
```bash
python fmnv_eval/fmnv_pipeline_eval.py --compare
```

### On PC (Gemma-4 H)

**1. Pull model:**
```bash
ollama pull gemma4:27b
```

**2. Run evaluation:**
```bash
cd dyplom
python fmnv_eval/fmnv_gemma_eval.py
```

---

## 📊 Dataset

- **98 videos** locally available
- **60 fake / 38 real** labels
- **Classes:** fa, fv, fc, ft (fake audio, video, context, text)
- **Source:** https://github.com/DennisIW/FMNV

---

## 💾 Output

Results saved to:
```
dyplom/ablation_results_fmnv/
├── fmnv_A.json
├── fmnv_B.json
├── ...
├── fmnv_G.json
└── fmnv_H.json (from PC)
```

---

## ⏱️ Timelines

| Config | Time | Device |
|--------|------|--------|
| F (quick test) | ~20 min | Mac |
| A–G (full) | ~2.5 hrs | Mac |
| H | ~30–45 min | PC |

---

## 📖 Details

See **`FMNV_EVALUATION_PLAN.md`** for:
- Detailed execution plan
- Cost estimates
- Thesis integration strategy
- Expected outcomes

---

**Status:** ✅ Ready to run
