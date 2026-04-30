# FMNV Dataset Evaluation Plan
**Date:** 28 April 2026  
**Status:** Scripts ready to run  
**Dataset:** FMNV (98 videos available locally, 60 false + 38 true)

---

## 📋 FMNV Dataset Overview

**Name:** Fake Multimodal News Video  
**Total in dataset:** 2,393 videos  
**Locally available:** 98 videos  
**Label distribution (local):** 60 fake / 38 real

**Source:** https://github.com/DennisIW/FMNV

### What's Included:
- ✅ Video files (.mp4)
- ✅ OCR extracted text
- ✅ CLIP ViT features (pre-extracted)
- ✅ Motion regions (HDF5)
- ✅ Captions and metadata
- ✅ Label annotations (false/true)

---

## 🚀 QUICK START

### **On Mac (Your current machine)**

**Step 1: Set up API key**
```bash
export GEMINI_API_KEY="your-key-here"
```

**Step 2: Run Config F (recommended)**
```bash
cd dyplom_v2
python fmnv_pipeline_eval.py --config F
```

**Step 3: Run all Gemini configs (A–G)**
```bash
python fmnv_pipeline_eval.py --all
```

**Step 4: Compare results**
```bash
python fmnv_pipeline_eval.py --compare
```

### **On PC (Your RTX 3090)**

**Prerequisites:**
```bash
ollama pull gemma4:27b
```

**Run Config H:**
```bash
cd dyplom_v2
python fmnv_gemma_eval.py
```

---

## 📊 Execution Plan

### **Phase 1: Mac (Gemini A–G) — ~2–3 hours**

| Config | Model | Est. Time | Status |
|--------|-------|-----------|--------|
| A | 2.5-flash (text-only) | 15 min | ⏳ Ready |
| B | 2.5-flash (vision) | 20 min | ⏳ Ready |
| C | 2.5-flash + CLIP | 25 min | ⏳ Ready |
| D | 2.5-flash-lite | 18 min | ⏳ Ready |
| E | 3-flash-preview | 20 min | ⏳ Ready |
| **F** | **3.1-flash-lite** | **20 min** | ⏳ Ready |
| G | 2.5-pro | 25 min | ⏳ Ready |
| **Total** | — | **~143 min** | **~2.5 hrs** |

**Recommended approach:**
1. Start with **Config F** (best cost-benefit)
2. Run **all A–G** in sequence overnight or in parallel if time allows

### **Phase 2: PC (Gemma-4 H) — ~30–45 min**

| Config | Model | Est. Time | Status |
|--------|-------|-----------|--------|
| H | Gemma-4:27b (local) | 30–45 min | ⏳ Ready (after PC setup) |

---

## 📁 Output Structure

Results will be saved to:
```
dyplom_v2/ablation_results_fmnv/
├── fmnv_A.json
├── fmnv_B.json
├── fmnv_C.json
├── fmnv_D.json
├── fmnv_E.json
├── fmnv_F.json
├── fmnv_G.json
└── fmnv_H.json (from PC)
```

Each file contains:
```json
{
  "config": "F",
  "model": "gemini-3.1-flash-lite-preview",
  "metrics": {
    "n": 98,
    "accuracy": 0.xxx,
    "precision": 0.xxx,
    "recall": 0.xxx,
    "f1": 0.xxx,
    "tp": X, "tn": X, "fp": X, "fn": X
  },
  "results": [
    {
      "video_id": "...",
      "gold_label": "fake/real",
      "pred_label": "fake/real",
      "pred_score": 0–100,
      "correct": true/false,
      "scene_count": X,
      "ocr_length": X,
      "audio_length": X,
      "reasoning": "..."
    },
    ...
  ]
}
```

---

## 🎯 Integration into Thesis

After running FMNV evaluation, I will:

1. **Create new Section 5.8** in Chapter 5: "Evaluation on FMNV Dataset"
   - Results comparison table (A–H)
   - Discussion of FakeTT vs. FMNV performance
   - Cross-dataset generalization analysis

2. **Add new Figure** showing FMNV confusion matrices and score distributions

3. **Update Conclusions** (Chapter 6) with generalization findings

4. **Add Future Work** note about FMNV-specific tuning

**Estimated impact:** ~2–3 pages added to thesis

---

## ⚠️ Known Considerations

1. **Video availability:** 98/2393 videos have local files. Others are YouTube references.
   - Not a blocker for evaluation
   - Makes dataset smaller and faster to evaluate

2. **Label format:** FMNV uses "false"/"true" (not "fake"/"real")
   - Scripts handle this internally
   - No user action needed

3. **Compute time:**
   - Mac (Gemini): Fast, API-based
   - PC (Gemma-4): Slower but free (local GPU)

4. **API costs (Mac only):**
   - Gemini API calls at ~$0.05/video
   - Total: ~$5 for 98 videos
   - Consider budget before running all configs

---

## 🔄 Running on Mac Now

**Recommended order:**

### **Option 1: Just Config F (Quick validation)**
```bash
python fmnv_pipeline_eval.py --config F
# Time: ~20 minutes
# Cost: ~$0.50
# Gives you: Best model results on FMNV
```

### **Option 2: All Gemini Configs (Complete evaluation)**
```bash
python fmnv_pipeline_eval.py --all
# Time: ~2.5 hours
# Cost: ~$5
# Gives you: Full A–H comparison on FMNV (when H added from PC)
```

### **Option 3: One-by-one as time allows**
```bash
# Run individually throughout the day:
python fmnv_pipeline_eval.py --config A
python fmnv_pipeline_eval.py --config B
python fmnv_pipeline_eval.py --config C
# ... etc
```

---

## 📋 Checklist

### Mac (do now):
- [ ] Set GEMINI_API_KEY environment variable
- [ ] Run `python fmnv_pipeline_eval.py --config F` (test)
- [ ] If successful, run `python fmnv_pipeline_eval.py --all` (full)
- [ ] Verify results in `ablation_results_fmnv/`

### PC (later):
- [ ] Pull Gemma-4: `ollama pull gemma4:27b`
- [ ] Run `python fmnv_gemma_eval.py`
- [ ] Copy result file back to Mac: `fmnv_H.json`

### Integration:
- [ ] I'll add FMNV section to thesis (Chapter 5.8)
- [ ] I'll create comparison plots
- [ ] I'll update conclusions
- [ ] I'll regenerate thesis PDF

---

## 🎓 Expected Outcomes

After FMNV evaluation:

1. **Generalization findings:**
   - How well does FakeTT-tuned pipeline work on different dataset?
   - Is Config F still best on FMNV?
   - Any dataset-specific insights?

2. **Comparative analysis:**
   - FakeTT (2,393 balanced TikTok videos) vs. FMNV (98 news videos)
   - Different modality distributions
   - Different manipulation patterns

3. **Thesis narrative:**
   - Strengthens thesis: "Our pipeline generalizes beyond FakeTT"
   - Adds practical validation on real-world news data
   - Motivates cheapfake-specific benchmark development

---

## 📞 Next Steps

1. **Decide:** Do you want to run FMNV now, or focus on thesis completion first?
2. **If YES:** Run Config F first (~20 min) to validate setup
3. **If YES to all:** Start `--all` config run in background

Let me know! ✅

---
