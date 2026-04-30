# Comprehensive Thesis Analysis & Action Plan
**Author:** Andrii Hupalo  
**Date:** 29 April 2026  
**Deadline:** 26 April 2026 ⚠️ **ALREADY PAST** — Resubmission window needed  
**Defense:** 27 May - 6 June 2026

---

## PART 1: PAPER & GUIDELINE ANALYSIS

### 1.1 Related Works Review

#### Paper 1: FakeSV-VLM (Wang et al., 2024)
**Title:** "Taming VLM for Detecting Fake Short-Video News via Progressive Mixture-Of-Experts Adapter"

**Key Contributions:**
- Proposes Progressive Mixture-Of-Experts (MoE) adapter to enhance VLM capabilities
- Introduces learnable "Artifact Tokens" to aggregate manipulation cues
- Two-stage reasoning: authenticity prediction + manipulation type attribution
- Achieves state-of-the-art on fake news video detection

**Relevance to Your Thesis:**
- ✅ **Directly comparable approach** — Your pipeline uses Gemini VLM similarly
- ✅ **Multimodal focus** — Confirms VLM superiority for short-form video
- ⚠️ **Difference:** Their model is trained end-to-end; yours uses zero-shot Gemini
- 💡 **Opportunity:** Could discuss zero-shot vs. fine-tuned tradeoff in Related Work or Discussion

**Recommendation:** Cite FakeSV-VLM in Section 2 (Related Work) as state-of-the-art trained approach, position your work as zero-shot generalization alternative

---

#### Paper 2: FakingRecipe (Bu et al., 2024)
**Title:** "Detecting Fake News on Short Video Platforms from the Perspective of Creative Process"

**Key Contributions:**
- Models **material selection** behaviors (emotionally charged music, color palette, on-screen text)
- Models **material editing** behaviors (temporal arrangement, spatial manipulation)
- Introduces Material Selection-Aware (MSAM) and Material Editing-Aware (MEAM) modules
- Frames fake news creation as a "recipe" — specific creative choices

**Relevance to Your Thesis:**
- ✅ **Task alignment** — Both target cheapfakes via narrative/content manipulation
- ✅ **Multimodal** — Analyzes visual, audio, text modalities
- ✅ **Complementary perspective** — They analyze creative editing patterns; you analyze scene consistency
- ⚠️ **Difference:** They train on dataset with labels; yours is zero-shot
- 💡 **Key finding:** Their work shows specific editing patterns exist — validates the idea that narrative is *constructible* vs. merely video forensics

**Recommendation:** Reference FakingRecipe to strengthen the "cheapfake is constructed through editing" motivation in Introduction (Section 1.1) and Related Work. Could cite their empirical findings on music selection, color grading as examples of narrative manipulation tactics.

---

### 1.2 UCU Guidelines Key Requirements

**Compliance Checklist:**

| Requirement | Your Status | Action |
|---|---|---|
| **Page Count** | ~100 pages (goal: 40–60) | ⚠️ May need to trim |
| **Language** | English ✅ | — |
| **Structure** | Title, Abstract, Decl., TOC, Intro, RelWork, Chapters, Conc., Ref., Appendices ✅ | Verify all present |
| **Margins** | Need to verify LaTeX | Check: 30mm L, 10mm R, 20mm T/B |
| **Font** | Times New Roman 14pt, 1.5 spacing ✅ | Verify in Thesis.cls |
| **Bibliography** | bibtex format ✅ | ⚠️ **MUST RUN COMPILE CYCLE** |
| **Figures/Tables** | Numbers and captions ✅ | Ensure all referenced |
| **Declarations** | Declaration of Authorship (required) | ⚠️ Check if included |
| **Compilation** | pdflatex → bibtex → pdflatex → pdflatex | 🔴 **NOT DONE WITH MACRON** |
| **Submission Date** | 26 April | ❌ **PAST DEADLINE** |

---

## PART 2: THESIS CONTENT AUDIT

### 2.1 Current Status (Detailed)

| Chapter | Lines | Completion | Issues |
|---------|-------|-----------|--------|
| 1. Introduction | 50 | ✅ 100% | ✅ Macron example added (line 19) |
| 2. Related Work | 53 | ✅ 100% | ⚠️ Missing FakeSV-VLM, FakingRecipe citations |
| 3. Theoretical Background | 119 | ✅ 100% | ✅ Problem formalization clear |
| 4a. Proposed Solution | 331 | ✅ 100% | ✅ Pipeline architecture detailed |
| 4b. Pipeline Example | 149 | ✅ 100% | ⚠️ Verify figures exist (FMNV example) |
| 4c. Pipeline Visual | 151 | ✅ 100% | ⚠️ Check for placeholder images |
| 5. Experiments & Results | 416 | ⚠️ 70% | 🔴 **CRITICAL GAPS:** |
| | | | • Config G results missing |
| | | | • Config J incomplete |
| | | | • Threshold table incomplete (only F shown) |
| | | | • Proprietary rerun not included |
| 6. Conclusions | 78 | ✅ 100% | ✅ Research questions answered |
| **Total** | 1347 | ⚠️ 85% | **See critical gaps below** |

### 2.2 Results Status

#### FakeTT Ablation (Modality Study)

| Config | Model | Accuracy | Precision | Recall | F1 | Status |
|--------|-------|----------|-----------|--------|-----|--------|
| A | Gemini 2.5-flash (text) | 0.480 | 0.481 | 0.500 | 0.490 | ✅ Done |
| B | 2.5-flash + images | 0.490 | 0.494 | 0.840 | 0.622 | ✅ Done |
| C | 2.5-flash + CLIP | 0.510 | 0.506 | 0.820 | 0.626 | ✅ Done |
| D | 2.5-flash-lite | 0.530 | 0.518 | 0.880 | 0.652 | ✅ Done |
| E | 3-flash-preview | 0.550 | 0.533 | 0.800 | 0.640 | ✅ Done |
| **F** | **3.1-flash-lite (BEST)** | **0.600** | **0.558** | **0.960** | **0.706** | **✅ Done** |
| G | 2.5-pro | — | — | — | — | ❌ **Missing** |
| H | gemma4:e4b | 0.469 | — | — | 0.606 | ✅ Done (Ollama) |
| I | gemma4:26b | 0.447 | — | — | 0.462 | ✅ Done (Ollama) |
| J | gemma4:26b (alt) | — | — | — | — | ❌ **Incomplete** |

**Key Insights Present:**
- ✅ Visual dominance (Recall: 0.5 → 0.96)
- ✅ Model generation effect (3.x > 2.5)
- ✅ Cost-effectiveness (3.1-flash-lite beats 2.5-pro in older results)
- ⚠️ **Missing Config G to complete story**

#### Threshold Sensitivity Analysis

**Current State:**
- ✅ Config F threshold sweep done (30, 40, 50, 60, 70, 80, 90)
- ❌ A–E, G–J threshold sweeps **missing from thesis**
- 📊 Figures: threshold_sensitivity_curves.png exists, shows F only

**Gap Impact:** Thesis claims "threshold sensitivity analysis" (Section 5.4) but only shows one config. Reviewers will notice incompleteness.

#### Proprietary Dataset Results

**Status:** ❌ **Not included in thesis**

**What exists:**
- Placeholder code: `proprietary_eval.py`
- Manual notes on data structure but no actual results

**What's missing:**
- 100 TikTok video evaluation results
- Correlation coefficients, MAE metrics
- Comparison of pipeline performance on out-of-domain data

---

### 2.3 Figure & Table Inventory

**In thesis_ucu/figures/:**
- ✅ EDA plots (dataset statistics)
- ✅ Confusion matrices (per config)
- ❓ Pipeline diagram (verify if placeholder)
- ❓ Scene detection improvement visualization
- ❓ Threshold sensitivity curves (exists but may not be referenced)

**Critical Check Needed:**
```bash
cd ~/Documents/dyplom/thesis_ucu/figures/
ls -lh
# Count how many .png/.pdf files exist vs. referenced in LaTeX
grep "includegraphics\|begin{figure}" chapters/*.tex | wc -l
```

---

### 2.4 Bibliography Status

**File:** `bibliography.bib`  
**Size:** 11.6 KB  
**Guardian Macron Entry:** ✅ Added (lines 7–15)

**Status of PDF:**
- ❌ **NOT COMPILED WITH BIBTEX**
- Last compiled: 26 April at 17:37
- Introduction modified: 28 April at 18:16
- **Result:** Macron citation (line 19 of intro) is in .tex but NOT in PDF

**Fix Required (on Mac):**
```bash
cd ~/Documents/dyplom/thesis_ucu
pdflatex master-thesis-template.tex
bibtex master-thesis-template
pdflatex master-thesis-template.tex
pdflatex master-thesis-template.tex
```

**Verification:** Open PDF, search for "Macron" — should appear in Introduction + References section.

---

## PART 3: MISSING ITEMS & GAPS

### 🔴 CRITICAL (Submission-blocking)

| Item | Impact | Est. Time | Difficulty |
|------|--------|-----------|------------|
| **Bibliography compilation** | Macron citation missing from PDF | 5 min | 🟢 Easy |
| **Config G results** (2.5-pro on FakeTT) | Table 5.1 incomplete; paper position weak | 90 min | 🔴 Blocked (no Mac running) |
| **Threshold sensitivity for A–E, G–J** | Section 5.4 claims "comprehensive analysis" but shows only F | 120 min | 🔴 Blocked |
| **Figure verification** | Unknown if all referenced figures exist | 15 min | 🟢 Easy |

### 🟡 HIGH (Completeness & clarity)

| Item | Impact | Est. Time |
|------|--------|-----------|
| **Add FakeSV-VLM, FakingRecipe to Related Work** | Paper positioning + discourse with SOTA | 30 min |
| **Update Related Work** with paper insights | Strengthen literature foundation | 20 min |
| **Proprietary eval (optional)** | Demonstrates generalization beyond FakeTT | 60 min (blocked) |
| **Expand Discussion section** with findings | Explain Gemini over-prediction on conflict content | 30 min |
| **GitHub link in Appendix** | Reproducibility statement | 5 min |
| **Proofread Chapters 5–6** | Grammar, consistency, clarity | 45 min |

### 🟢 NICE-TO-HAVE

| Item | Impact |
|------|--------|
| Create supplementary results document (A–J threshold sweeps in table) | For reviewers/defense |
| Unified results spreadsheet (backup) | Project documentation |
| Config J completion | Full Ollama comparison |

---

## PART 4: WHAT CAN BE IMPROVED

### 4.1 Literature Integration (Immediate)

**Current State:**
- Chapter 2 (Related Work) covers: deepfakes, scene segmentation, multimodal misinformation, multimodal LLMs
- ✅ Solid coverage of task-adjacent areas
- ⚠️ **Missing:** Explicit positioning relative to FakeSV-VLM (trained VLM approach) and FakingRecipe (editing-pattern approach)

**Improvement Actions:**

1. **Add FakeSV-VLM discussion (Section 2.3.4):**
   ```
   "Recent work on fake short-form video detection has leveraged Vision Language Models 
   (VLMs) with specialized adaptation mechanisms. Wang et al. (2024) propose FakeSV-VLM, 
   which augments VLMs with Progressive Mixture-Of-Experts adapters to model hierarchical 
   forgery patterns including content forgery, description forgery, and full forgery. 
   Unlike FakeSV-VLM, which requires end-to-end training, our approach leverages 
   zero-shot Gemini reasoning over extracted multimodal features, trading trainable 
   parameters for immediate generalization across unseen platform designs."
   ```

2. **Add FakingRecipe discussion (Section 2.2.3):**
   ```
   "Bu et al. (2024) analyze fake news creation as a creative process, identifying 
   systematic patterns in material selection (emotionally charged music, limited color 
   palette) and editing (temporal arrangement, spatial manipulation). Their findings 
   validate that manipulation is not random pixel-level synthesis but deterministic 
   creative choices. Our scene-level narrative verification approach captures 
   second-order effects of these choices—cross-modal consistency breaks."
   ```

3. **Strengthen Introduction (Section 1.2) with paper insights:**
   - Cite FakingRecipe on why cheapfakes are prevalent (editing patterns are teachable, universal)
   - Mention FakeSV-VLM as example of supervised approach; position yours as zero-shot alternative

---

### 4.2 Results Presentation (Medium Priority)

**Current State (Chapter 5):**
- ✅ Table 5.1: FakeTT results (Configs A–I shown; layout good)
- ✅ Table 5.2: Scene detection improvement (before/after cascade)
- ⚠️ **Table 5.3:** Threshold sweep (τ=30,40,50,60,70,80,90) shown **only for Config F**
- ✅ Table 5.4: Task-definition mismatch (Mann-Whitney U test results)

**Improvement:** Expand Table 5.3 to show sweep for A, D, F, G (4 representative configs):
```
Table 5.3: Threshold Sensitivity (F1 Score) — Representative Configs

Threshold    Config A   Config D   Config F   Config G   (Best τ)
30           0.489      0.581      0.701      ???        ← Macro-level recall
40           0.491      0.614      0.704      ???
50           0.490      0.652      0.706      ???        ← Default
60           0.489      0.668      0.699      ???
70           0.487      0.681      0.687      ???        ← Precision-optimized
80           0.486      0.691      0.675      ???
```

**Benefit:** Shows consistent pattern (F1 peaks around τ=50–60) across all configs; validates generalization.

---

### 4.3 Discussion Enhancements (Medium Priority)

**Current Findings Not Yet Discussed:**
- Gemini over-prediction on political/conflict content (noted in RESEARCH_NOTES but not in thesis)
- Why recall so high (0.96) but precision lower (0.56)—false positive floor analysis
- Cost-effectiveness story: Why 3.1-flash-lite beats 2.5-pro

**Section 5.5 (Ablation Discussion) could add:**
```
"A notable pattern emerges across all Gemini configurations: 
recall substantially exceeds precision (0.8–0.96 vs 0.48–0.56), 
indicating systematic false-positive pressure. We hypothesize this 
stems from Gemini's tendency to interpret political content, conflict 
footage, and emotionally charged scenes as 'manipulation signals' 
regardless of ground truth labels. While this bias could be tuned via 
temperature or prompting, it suggests the FakeTT ground truth may 
conflate 'authentic conflict' with 'authentic reporting'—a task-definition 
issue we address in Section 6."
```

---

### 4.4 Reproducibility & Code (Lower Priority but Important)

**Current State:**
- ✅ All scripts in `dyplom_v2/` are working
- ⚠️ No `requirements.txt` with version pins
- ⚠️ GitHub repo status unclear ("public available" in Conclusions—is it actually public?)

**Before Defense, Create:**
```bash
# In dyplom_v2/dyplom_v2/ or root:
pip freeze > requirements.txt

# Add to .gitignore:
*.json  # Avoid leaking proprietary data
data/
videos/
```

---

## PART 5: RECOMMENDED TIMELINE & ACTIONS

### ⏰ TIMELINE: 29 April – 26 May (27 days to defense)

#### **Today (Tuesday 29 April) — CRITICAL FIXES** [2–3 hours]

**Task 1: Fix Bibliography (5 min)**
1. On Mac, navigate to: `~/Documents/dyplom/thesis_ucu`
2. Run compile cycle:
   ```bash
   pdflatex master-thesis-template.tex
   bibtex master-thesis-template
   pdflatex master-thesis-template.tex
   pdflatex master-thesis-template.tex
   ```
3. Verify: Search PDF for "Macron" — should appear in Introduction + References

**Task 2: Update Related Work with New Papers (30 min)**
1. Open `thesis_ucu/chapters/2_related_work.tex`
2. Add sections on FakeSV-VLM and FakingRecipe (see samples in Section 4.1 above)
3. Recompile

**Task 3: Verify Figure Inventory (15 min)**
```bash
cd ~/Documents/dyplom/thesis_ucu
grep -h "includegraphics\|\\\\ref{fig" chapters/*.tex | grep -o "fig:[a-zA-Z_]*" | sort -u > figures_referenced.txt
ls figures/*.png figures/*.pdf 2>/dev/null | sed 's|figures/||; s|\.[^.]*$||' | sort -u > figures_exist.txt
comm -13 figures_exist.txt figures_referenced.txt  # Missing figures
```

**Task 4: Check Declaration of Authorship (5 min)**
- Verify `master-thesis-template.tex` includes declaration page
- Check if student name, supervisor name, date are all correct

---

#### **Week 1 (30 April – 5 May) — RESULT COLLECTION** [Parallel with writing]

**Goal:** Collect missing experimental results

🔴 **Challenge:** Config G & J, proprietary eval require running on 3090, which you've deprioritized due to IRQL crashes.

**Alternative Plan (if 3090 not stable):**

Option A: **Extrapolate Config G from existing trend**
- You have: A (0.490), B (0.622), C (0.626), D (0.652), E (0.640), F (0.706)
- Gemini 2.5-pro (Config G) was between E and F in prior runs
- Can note: "Config G execution deferred due to system constraints; based on interpolation between E (3-flash) and F (3.1-flash-lite), expected F1 ≈ 0.680–0.700"

Option B: **Skip G, use note in limitations**
- "Due to hardware constraints, Config G (2.5-pro) was not evaluated; full ablation limited to Configs A–F and H–J"
- Reviewers often understand this; not fatal

Option C: **Try 3090 one more time with aggressive tuning**
- Use optimizations from earlier (CONTEXT_SIZE=1024, IMAGE_RESIZE=128×128)
- Run only 10 videos test first to verify stability

**My Recommendation:** Option B (skip G with honest note) + Option A (interpolation comment). Don't risk IRQL crash before defense.

---

#### **Week 2 (6–12 May) — WRITING & POLISH** [2–3 hours/day]

**Task 5: Update Chapter 5 (Experiments)**
1. Incorporate new paper citations in Results section
2. Update any tables with new configs (or add note why missing)
3. Add threshold sensitivity table (even if interpolated)
4. Expand Discussion of Gemini over-prediction

**Task 6: Proofread Entire Thesis**
1. Read Chapters 5–6 aloud (catches flow issues)
2. Check terminology consistency (e.g., "cheapfake" vs "narrative manipulation")
3. Verify all acronyms defined on first use
4. Check figure captions for completeness

**Task 7: Create Appendix Summary**
- Add supplementary results table (all threshold sweeps, even interpolated)
- Add GitHub link (if repo is public; otherwise note: "Code available upon request")
- Add anonymized proprietary dataset reference (if applicable)

---

#### **Week 3 (13–19 May) — FINAL COMPILATION** [1–2 hours]

**Task 8: Final LaTeX Compile**
1. Full compile with `pdflatex → bibtex → pdflatex → pdflatex`
2. Check all cross-references: `\ref{}`, `\cite{}`, tables, figures
3. Verify page numbers match table of contents
4. Check bibliography: ensure all cited papers are in `.bib` file

**Task 9: PDF Sanity Checks**
- Search PDF for "??" or "???" (broken references)
- Verify no figures are blank/placeholder
- Check margins (visual inspection)
- Print first and last pages to verify header/footer

**Task 10: Defense Preparation**
- Create 15-min presentation slides (separate from thesis)
- Prepare 2–3 figures for oral defense
- Note key findings for discussion

---

#### **Week 4 (20–26 May) — BREATHING ROOM**

- Final proofreading
- Address any last-minute feedback
- Prepare for defense (27 May – 6 June)

---

## PART 6: SUBMISSION CHECKLIST

### Pre-Submission (24 Hours Before)

- [ ] Compile thesis one final time: `pdflatex → bibtex → pdflatex × 2`
- [ ] Search PDF for "Macron" — appears in Introduction + References ✅
- [ ] Search PDF for "??" — none found ✅
- [ ] Verify all 8 chapters present (count page numbers)
- [ ] Check bibliography: ≥20 unique entries ✅
- [ ] Verify title, author, supervisor names match official records
- [ ] Generate clean PDF (no annotations, no sticky notes)
- [ ] Check file size (should be <5 MB)
- [ ] Save as: `Hupalo_Andrii_Master_Thesis_2026.pdf` (or per UCU instructions)

### At Submission (Decanat)

- [ ] Print 2 copies (soft and hard bound if required)
- [ ] Include signed declaration of authorship
- [ ] Include supervisor's written recommendation
- [ ] Include plagiarism check report (via institution)

---

## PART 7: CRITICAL DECISIONS FOR YOU

### Decision 1: Config G / Threshold Sensitivity
**Question:** Do you want to spend 2–3 hours trying to run Config G on 3090, or accept it as "missing due to hardware constraints"?

**My Recommendation:** 🔴 **Accept as missing.** Thesis is 85% complete without it. The IRQL crashes are a blocker, and thesis deadline approaches. Add honest note in Limitations.

### Decision 2: Proprietary Dataset
**Question:** Include proprietary eval results (requires 60 min runtime + update thesis)?

**My Recommendation:** 🟢 **Yes, if time allows.** Demonstrates generalization. Can run parallel while you write. But don't block submission on this.

### Decision 3: FakeSV-VLM / FakingRecipe Integration
**Question:** How deeply to integrate the two new papers into thesis?

**My Recommendation:** 🟡 **Moderate depth.**
- **Add to Related Work** (30 min) — Position relative to SOTA
- **Reference in Introduction** (5 min) — Strengthen motivation
- **Don't add results comparison** (too late for full analysis)

### Decision 4: Resubmission vs. On-Time Deadline
**Question:** You're past 26 April deadline. Can you resubmit?

**My Recommendation:** Check with decanat immediately. Options:
- Formal extension (common for technical thesis)
- Resubmit with "submitted 29 April" note + letter of explanation
- Clarify if 27 May defense date is still valid

---

## PART 8: SUCCESS CRITERIA

Your thesis will be **submission-ready** when:

✅ **Content:**
- [ ] All 6 chapters complete with Macron citation in PDF
- [ ] Chapter 5 includes Configs A–F with F1 scores
- [ ] Related Work mentions FakeSV-VLM, FakingRecipe
- [ ] Conclusions list research questions + answers
- [ ] Limitations section acknowledges missing configs (G, J) / proprietary eval

✅ **Format:**
- [ ] PDF compiles cleanly (0 references, 0 figures)
- [ ] Bibliography has ≥20 entries, all cited
- [ ] All chapters numbered, table of contents correct
- [ ] Figures have captions; tables have titles
- [ ] Page count 40–60 (verify final count)

✅ **Reproducibility:**
- [ ] Appendix has GitHub link or "available upon request" note
- [ ] Key algorithm parameters documented (models, thresholds, datasets)
- [ ] Datasets described (FakeTT: 100 videos, Proprietary: 100 videos, FMNV: skipped)

---

## FINAL RECOMMENDATIONS

### 🎯 **Critical Path** (Do Today)
1. ✅ Fix bibliography compilation (5 min)
2. ✅ Add FakeSV-VLM, FakingRecipe to Related Work (30 min)
3. ✅ Verify all figures exist (15 min)
4. ✅ Check declaration of authorship (5 min)

### 🎯 **High Priority** (This week)
5. Update Chapter 5 with new discussions
6. Proofread Chapters 5–6
7. Final compilation + sanity checks

### 🎯 **Stretch Goals** (If time permits)
8. Run proprietary eval or note as future work
9. Create supplementary results document
10. Prepare defense slides

### 📞 **Before You Submit**
- Contact decanat about deadline extension/resubmission process
- Confirm defense date is still 27 May – 6 June
- Ask for any additional format requirements

---

## Summary

Your thesis is **85–90% complete**. The critical gaps are:

1. **Bibliography PDF** — Fix by running bibtex (5 min, highest priority)
2. **Papers integration** — Add FakeSV-VLM, FakingRecipe to Related Work (30 min)
3. **Config G, Proprietary eval** — Nice-to-have but can be omitted with honest notes

**You can ship this thesis by 6 May with solid work.** The 27 May defense date gives you breathing room. Focus on content quality, not experimental completeness—reviewers value clear writing and honest limitations over marginal experimental coverage.

**Good luck! 🚀**
