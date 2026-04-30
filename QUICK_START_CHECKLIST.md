# Thesis Submission Checklist — Quick Start

**Status:** Thesis is 85% complete. Critical gaps: Bibliography PDF compilation, papers integration.  
**Deadline:** 26 April (PASSED) → Need to contact decanat for resubmission  
**Defense:** 27 May - 6 June 2026  

---

## 🔴 TODAY (29 April) — 1 HOUR FIX SESSION

### Task 1: Compile Bibliography (5 min)
```bash
cd ~/Documents/dyplom/thesis_ucu
pdflatex master-thesis-template.tex
bibtex master-thesis-template
pdflatex master-thesis-template.tex
pdflatex master-thesis-template.tex
```
✅ Verify: PDF contains "Macron" in Introduction + References

### Task 2: Add FakeSV-VLM & FakingRecipe to Chapter 2 (30 min)
Edit: `thesis_ucu/chapters/2_related_work.tex`

**Add section on FakeSV-VLM** (after Section 2.3.3):
```latex
\subsection{Vision Language Models with Specialized Adaptation}

Recent advances in VLM-based fake video detection demonstrate the 
effectiveness of specialized architectural adaptations. Wang \etal (2024) 
propose FakeSV-VLM, augmenting Vision Language Models with Progressive 
Mixture-Of-Experts (MoE) adapters to model hierarchical forgery patterns. 
Unlike FakeSV-VLM, which requires end-to-end training on labeled data, 
our approach leverages zero-shot Gemini reasoning, prioritizing 
generalization across novel platforms and content types.
```

**Add section on FakingRecipe** (Section 2.2.3):
```latex
Bu \etal (2024) analyze fake news video creation from the creative process 
perspective, identifying systematic patterns in material selection and 
editing. Their findings validate that manipulation consists of deterministic 
creative choices—emotionally charged audio selection, limited color palettes, 
temporal reordering—rather than random artifacts. Our scene-level analysis 
captures the downstream effects of these choices as cross-modal contradictions.
```

### Task 3: Check Figures (15 min)
```bash
cd ~/Documents/dyplom/thesis_ucu
# List what figures are referenced in LaTeX
grep -h "includegraphics" chapters/*.tex

# List what figures actually exist
ls figures/

# Cross-check for mismatches
```

### Task 4: Update Introduction (10 min)
Edit `chapters/1_introduction.tex` → Add note after Macron paragraph:
```latex
Similar patterns of deliberate manipulation emerge across academic literature...
[Cite FakingRecipe]. Recent work on VLM-based detection [FakeSV-VLM] 
demonstrates...
```

---

## 📋 THIS WEEK (30 Apr - 5 May)

- [ ] **Task #14:** Add FakeSV-VLM, FakingRecipe to Related Work ✅ (done today)
- [ ] **Task #15:** Verify all figures exist
- [ ] **Task #16:** Expand Chapter 5 with new discussions
- [ ] **Task #19:** Contact decanat about resubmission

---

## ✍️ NEXT WEEK (6-12 May) — WRITING SPRINT

- [ ] **Task #17:** Proofread Chapters 5-6
- [ ] **Task #18:** Final LaTeX compilation
- [ ] **Task #20:** Create supplementary defense materials

---

## 📤 BEFORE SUBMISSION (24 hours before deadline)

- [ ] Bibliography compiles (search for "Macron" in PDF) ✅
- [ ] All figures referenced exist ✅
- [ ] Chapter 5 includes FakeSV-VLM, FakingRecipe discussion ✅
- [ ] No broken cross-references ("??" in PDF)
- [ ] Page count is 40-60 pages
- [ ] PDF file size < 5 MB
- [ ] Saved as: `Hupalo_Andrii_Master_Thesis_2026.pdf`

---

## 🎯 KEY DECISIONS

**Config G Results?**  
→ Skip. Note as "hardware constraints". Thesis is complete without it.

**Proprietary Dataset?**  
→ Optional. Can be future work. Focus on writing quality.

**Deadline?**  
→ Contact decanat TODAY. Confirm if you can submit 29 April, verify 27 May defense is still valid.

---

## 📊 THESIS STATUS SUMMARY

| Component | Status | Action |
|-----------|--------|--------|
| Bibliography PDF | ❌ Not compiled | Compile today (5 min) |
| FakeSV-VLM/FakingRecipe | ❌ Missing | Add to Related Work (30 min) |
| Chapter 5 Results | ✅ Complete (A-I) | Note missing G, J in text |
| Figures | ⚠️ Unknown | Verify today (15 min) |
| Proofread | ❌ Incomplete | Do next week (2 hours) |
| Declaration | ⚠️ Check | Verify in LaTeX |
| Final compile | ❌ Not done | Do before submission |

---

## 📞 CONTACT DECANAT TODAY

Send email:
```
Subject: Master's Thesis Submission — Deadline Clarification

Dear [Decanat Staff],

I am finalizing my master's thesis "Scene-Level Narrative Verification..." 
and will submit on [29 April 2026].

My original deadline was 26 April, but I have encountered technical 
challenges [IRQL GPU driver issues] that delayed final results.

Could you please confirm:
1. Can I submit on 29 April with an explanation letter?
2. Is the 27 May - 6 June defense date still valid?
3. Required documents: signed declaration, supervisor letter, plagiarism report?

Thank you,
Andrii Hupalo
Email: hupalo.pn@ucu.edu.ua
```

---

## ⏱️ TIME ESTIMATE

| Task | Time | Priority |
|------|------|----------|
| Bibliography compile | 5 min | 🔴 Today |
| Add papers to Related Work | 30 min | 🔴 Today |
| Verify figures | 15 min | 🔴 Today |
| Contact decanat | 10 min | 🔴 Today |
| Expand Chapter 5 | 60 min | 🟡 This week |
| Proofread Chapters 5-6 | 120 min | 🟡 Next week |
| Final compilation | 30 min | 🟡 Next week |

**Total:** ~3.5 hours to submission-ready  
**Available time:** 27 days to defense

---

## 🚀 YOU'VE GOT THIS

Your thesis is solid. The remaining gaps are:
1. **Fix bibliography PDF** (5 min) — Highest ROI task
2. **Integrate recent papers** (30 min) — Strengthen literature
3. **Polish & proofread** (2 hours) — Final quality pass

The 27 May defense gives you breathing room. Focus on **content quality, not experimental completeness**. Honest limitations are better than missing results.

Good luck! 🎓
