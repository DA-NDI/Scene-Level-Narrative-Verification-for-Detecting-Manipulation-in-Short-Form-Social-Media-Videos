# Two Reference Papers — Summary for Thesis Integration

Use this to quickly draft text for Related Work section.

---

## Paper 1: FakeSV-VLM (Wang et al., 2024)

**Full Citation:**  
Wang, J., Wang, Y., Cheng, L., Zhong, Z. (2024). "FakeSV-VLM: Taming VLM for Detecting Fake Short-Video News via Progressive Mixture-Of-Experts Adapter." HefeiUniversity of Technology, China.

**Venue:** (appears to be workshop/arXiv paper)

**Key Idea:**
- Uses Vision Language Models with **Progressive Mixture-Of-Experts (MoE) Adapter** for fake news detection
- Introduces **learnable "Artifact Tokens"** to aggregate manipulation cues
- Two-stage reasoning pipeline:
  1. Overall authenticity prediction
  2. Manipulation type attribution (content forgery vs. description forgery vs. both)

**Technical Approach:**
- Input: Video keyframes + text description
- Visual encoder + text tokenizer → Multimodal features
- Features + Artifact Tokens → Early LLM layers
- Progressive MoE Adapter → Two-stage classification
- Key innovation: Learnable parameters to adapt pre-trained VLMs

**Results:**  
Achieves state-of-the-art on fake short-form video detection (exact metrics not in abstract).

**How to Cite in Your Thesis:**

### For Related Work Section 2.3 (Multimodal LLMs):
```latex
\subsubsection{Vision Language Models with Specialized Adaptation}

Recent work has explored augmenting Vision Language Models with specialized 
architectural components for improved fake video detection. Wang \etal (2024) 
propose FakeSV-VLM, which enhances VLMs through Progressive Mixture-Of-Experts 
(MoE) adapters and learnable Artifact Tokens to model hierarchical forgery 
patterns. Their two-stage reasoning framework explicitly distinguishes between 
content forgery, description forgery, and full manipulation.

In contrast to FakeSV-VLM, which requires supervised training on labeled data, 
our approach leverages the zero-shot reasoning capabilities of Gemini across 
multimodal features. This design choice prioritizes rapid generalization across 
novel platforms and content types at the cost of task-specific parameter 
tuning~\cite{fakesv-vlm}.
```

### For Introduction (Motivation):
```latex
State-of-the-art VLM-based approaches~\cite{fakesv-vlm} demonstrate that 
specialized architectural design—such as Progressive Mixture-Of-Experts 
adapters—can significantly improve fake news detection. However, such 
training-based methods require large labeled datasets and domain-specific 
annotation, limiting their applicability to rapidly evolving platforms like 
TikTok where manipulation tactics constantly evolve.
```

---

## Paper 2: FakingRecipe (Bu et al., 2024)

**Full Citation:**  
Bu, Y., Sheng, Q., Cao, J., Qi, P., Wang, D., Li, J. (2024). "FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process." *Proceedings of the 32nd ACM International Conference on Multimedia (MM'24)*, October 28-November 1, 2024, Melbourne, Australia. ACM. DOI: 10.1145/3664647.3680663

**Venue:** ACM Multimedia 2024 (top-tier conference)

**Key Idea:**
- Analyzes fake news creation as a **"recipe"—a systematic sequence of creative choices**
- Identifies two levels of manipulation:
  1. **Material Selection:** Selection of emotionally charged audio, limited color palette, specific text presentation
  2. **Material Editing:** Temporal arrangement, spatial manipulation, duration control

**Technical Approach:**
- Dual-branch architecture:
  1. **Material Selection-Aware Modeling (MSAM):** Extracts multimodal features via content attention (sentiment resonance between audio & text)
  2. **Material Editing-Aware Modeling (MEAM):** Models spatial and temporal editing patterns
- Finds that fake videos exhibit:
  - More emotionally charged music than real videos
  - More limited color palettes
  - Less dynamic on-screen text presentation

**Results:**  
Empirically validates that fake news contains distinctive patterns—not random pixel-level synthesis but **deterministic creative choices**.

**How to Cite in Your Thesis:**

### For Related Work Section 2.2 (Cheapfake Detection):
```latex
\subsubsection{Narratives as Creative Constructs}

Beyond pixel-level forensics, recent work frames fake news creation as 
a creative process characterized by systematic material selection and 
editing choices. Bu \etal (2024) analyze fake short-form video creation 
through this lens, identifying empirical patterns such as: preference for 
emotionally charged audio, restricted color palettes, and subdued on-screen 
text dynamics. Critically, their analysis demonstrates that fake news 
consists not of random digital artifacts but of **deterministic creative 
decisions**~\cite{fakingrecipe}.

This validates our hypothesis that narrative-level inconsistencies—rather 
than pixel-level synthesis artifacts—should be the focus of detection 
systems. When a video's editing patterns conflict with its textual claims, 
a multimodal reasoner should detect such cross-modal contradictions.
```

### For Introduction (Problem Framing):
```latex
Fake news video creation is not random pixel-level synthesis but a 
structured creative process~\cite{fakingrecipe}. Adversaries deliberately 
select materials—emotionally charged music, specific color grading—and 
arrange them temporally to reinforce false narratives. Our approach targets 
these **narrative-level** inconsistencies rather than low-level digital 
artifacts, offering robustness against novel synthesis techniques.
```

### For Related Work Introduction (Transition):
```latex
Unlike deepfake detection (which seeks pixel-level anomalies), cheapfake 
detection must understand the narrative layer—the intentional arrangement 
of authentic but decontextualized material to mislead~\cite{paris2019deepfakes}. 
Recent work~\cite{fakingrecipe} frames this as a creative process, where 
manipulators employ systematic tactics (audio selection, color grading, 
temporal sequencing) to construct false narratives. Scene-level analysis 
offers a middle ground between pixel forensics and semantic understanding.
```

---

## Bibliography Entries

Add these to `bibliography.bib`:

```bibtex
@inproceedings{fakesv-vlm,
  author    = {Wang, Junxi and Wang, Yaxiong and Cheng, Lechao and Zhong, Zhun},
  title     = {{FakeSV-VLM}: {T}aming {VLM} for {D}etecting {F}ake {S}hort-{V}ideo {N}ews 
              via {P}rogressive {M}ixture-{O}f-{E}xperts {A}dapter},
  year      = {2024},
  note      = {HefeiUniversity of Technology, China}
}

@inproceedings{fakingrecipe,
  author    = {Bu, Yuyan and Sheng, Qiang and Cao, Juan and Qi, Peng and Wang, Danding and Li, Jintao},
  title     = {{F}aking{R}ecipe: {D}etecting {F}ake {N}ews on {S}hort {V}ideo {P}latforms 
              from the {P}erspective of {C}reative {P}rocess},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia (MM'24)},
  year      = {2024},
  pages     = {1--10},
  address   = {Melbourne, VIC, Australia},
  publisher = {ACM},
  doi       = {10.1145/3664647.3680663}
}
```

---

## Integration Checklist

Use this checklist as you edit thesis:

- [ ] **Related Work Section 2.2** — Add FakingRecipe narrative discussion
- [ ] **Related Work Section 2.3.4** — Add FakeSV-VLM VLM adaptation discussion
- [ ] **Introduction Section 1.2** — Reference both papers in motivation
- [ ] **Bibliography.bib** — Add both entries
- [ ] **Chapter 5 (Discussion)** — Optional: cite FakingRecipe when discussing false positive patterns
- [ ] **Recompile** — After editing, run full LaTeX → BibTeX → LaTeX cycle

---

## Quick Copy-Paste Snippets

If you need to add quickly without writing from scratch:

### Snippet 1: FakeSV-VLM Intro
```
Recent advances in VLM-based misinformation detection demonstrate the power 
of architectural specialization. Wang et al.\ (2024) propose FakeSV-VLM, 
which augments Vision Language Models with learnable Artifact Tokens and 
Progressive Mixture-Of-Experts adapters. While their approach achieves 
state-of-the-art results through supervised training, our zero-shot design 
prioritizes generalization across rapidly evolving platforms.
```

### Snippet 2: FakingRecipe Intro
```
Fake news video creation follows a creative "recipe"---systematic choices 
in material selection (emotionally charged audio, limited color palettes) 
and editing (temporal reordering, spatial manipulation). Bu et al.\ (2024) 
empirically validate that these patterns are distinguishable, providing 
evidence that narrative-level analysis is viable for manipulation detection.
```

### Snippet 3: Motivation Connection
```
Together, these works suggest that fake news detection should focus on 
narrative consistency rather than pixel-level forensics. While FakeSV-VLM 
achieves this through supervised learning and FakingRecipe through creative 
process modeling, we pursue zero-shot cross-modal reasoning to enable rapid 
adaptation to novel platforms and manipulation tactics.
```

---

## Why These Papers Matter for Your Thesis

1. **FakeSV-VLM** → Shows that VLM specialization works; your zero-shot approach is an alternative
2. **FakingRecipe** → Validates that narrative manipulation is systematic; supports your core hypothesis
3. **Both** → Positions your work as complementary, not derivative

They strengthen your Related Work by showing your approach sits at a unique intersection: **zero-shot reasoning (like FakeSV-VLM), focused on narrative consistency (like FakingRecipe), applied to short-form social media (both papers)**.

---

## Estimated Time to Integrate

- **FakeSV-VLM text:** 10 minutes (write + cite)
- **FakingRecipe text:** 10 minutes (write + cite)
- **Bibliography entries:** 2 minutes (copy-paste + format)
- **Recompile & verify:** 5 minutes

**Total:** ~30 minutes to fully integrate both papers into thesis.
