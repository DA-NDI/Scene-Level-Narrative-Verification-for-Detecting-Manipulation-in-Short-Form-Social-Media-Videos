# Symposium 2026 Slide Blueprint

## Constraints Extracted From the Course Guideline
- Target format: progress/expose-style talk for 10-20 minutes.
- Recommended density: 1 simple slide about 1 minute.
- Keep each slide focused on one idea.
- Prefer visual explanation over long text.
- Keep backup detail off the main narrative and use it for Q&A.

## Recommended Deck Structure (15 minutes, 14 slides)

| # | Slide Title | Purpose | Visual Direction | Source Asset |
|---|---|---|---|---|
| 1 | Scene-Level Narrative Verification | Hook + thesis in one slide | Strong title with one-line problem/solution split | MS-AMLV-2026-CR-Submission-No-31/source/macron_deepfake.jpg |
| 2 | Why Current Detectors Fail | Show research gap | Comparison panel: pixel-level vs narrative-level manipulation | custom diagram |
| 3 | Research Questions | Clarify scope and evaluation intent | 4-card layout (RQ1-RQ4) | samplepaper.tex |
| 4 | Dataset and Constraints | Build credibility and transparency | KPI strip + NDA box + language distribution | samplepaper.tex |
| 5 | Problem Formulation | Formalize target output | Equation + simple pipeline mini-diagram | samplepaper.tex |
| 6 | Hybrid CV+LLM Pipeline | Core method for near-term results | Full architecture figure | screens/detecting_narrative_units.jpg + redraw |
| 7 | End-to-End Transformer Alternative | Position strategic second track | Modular block diagram with pros/cons | samplepaper.tex |
| 8 | Pilot Setup and Protocol | Explain what was actually tested | Two-column: FakeTT and proprietary pilot | supplementary_material.tex |
| 9 | Pilot Results: FakeTT | Show objective baseline | Confusion matrix + score distribution | MS-AMLV-2026-CR-Submission-No-31/source/figures/fakett_eval.jpg |
|10 | Pilot Results: Proprietary Subset | Show weakly-supervised signal quality | Scatter + error distribution | MS-AMLV-2026-CR-Submission-No-31/source/figures/proprietary_eval.jpg |
|11 | Failure Modes and Lessons | Demonstrate scientific maturity | 4 failure cards with icons | supplementary_material.tex |
|12 | Next 8 Weeks Plan | Show feasibility and execution discipline | Timeline with milestones | samplepaper.tex (Table 1) |
|13 | Contributions and Expected Impact | Reinforce value | 3-column impact: science/industry/society | samplepaper.tex |
|14 | Conclusion + Ask | Memorable close + invite feedback | Return to thesis statement and open questions | none |

## Design System (for Visme or PPT)
- Theme: forensic editorial, light background, high contrast accents.
- Palette:
  - Ink: #0F172A
  - Slate: #334155
  - Teal accent: #0F766E
  - Amber alert: #B45309
  - Soft background: #F8FAFC
- Typography:
  - Headings: Sora Semibold
  - Body: Source Sans 3
  - Numeric callouts: Space Mono
- Layout rules:
  - 12-column grid.
  - Keep text under 40 words per content slide.
  - One key visual per slide.
  - Use progressive reveal only for 2-3 complex slides.

## Ready Slide Copy (Main Narrative)

### Slide 1 - Scene-Level Narrative Verification for Short-Form Videos
- Problem: Pixel-level detectors can miss cheapfakes built from authentic footage.
- Idea: verify narrative consistency between visuals, speech, and on-screen text.
- Goal: scalable detection pipeline for real-world TikTok misinformation.
Speaker note: Open with one sentence: manipulation moved from pixels to narrative framing.

### Slide 2 - Why Existing Forensics Miss Cheapfakes
- Legacy deepfake detection: strong on synthetic artifacts.
- Cheapfake reality: footage is real, meaning is manipulated.
- Gap: no robust scene-level cross-modal consistency verification.
Speaker note: Use one concrete example of true video plus misleading caption.

### Slide 3 - Research Questions
- RQ1: Can scene segmentation isolate narrative units, not only shot cuts?
- RQ2: Hybrid CV+LLM vs end-to-end transformer, which is stronger?
- RQ3: Can enriched metadata enable weak supervision at scale?
- RQ4: Are multimodal LLMs reliable enough for forensic reasoning?
Speaker note: State that each section of the talk maps directly to these RQs.

### Slide 4 - Data Foundation
- Proprietary corpus: about 7.4k multilingual TikTok videos.
- Typical clip: 15-60s, median 28s, 3-8 scene transitions.
- Enrichment: OCR, ASR, visual descriptions, embeddings, weak-reference scores.
- Constraint: NDA, aggregated reporting only.
Speaker note: Emphasize this is real-world noisy data, not clean benchmark-only.

### Slide 5 - Problem Formulation
- For each narrative segment N_i, compute visual/audio/text representations.
- Score consistency with cross-modal similarity:
  C(N_i) = sim(v_i, t_i) + sim(a_i, t_i) + sim(v_i, a_i)
- Predict manipulation score M in [0,1] at video level.
Speaker note: low consistency suggests potential recontextualization.

### Slide 6 - Hybrid CV+LLM Pipeline
- Stage 1: adaptive scene segmentation.
- Stage 2: multimodal extraction (visual, ASR, OCR).
- Stage 3: LLM reasoning per scene.
- Stage 4: weighted aggregation into final manipulation score.
Speaker note: key advantage is interpretability with scene-level justifications.

### Slide 7 - End-to-End Transformer Track
- Joint optimization over visual, audio, and text streams.
- Auxiliary tasks: segmentation + modality consistency.
- Trade-off: potentially better performance vs lower interpretability and higher compute.
Speaker note: present this as a strategic parallel path, not a replacement today.

### Slide 8 - Pilot Evaluation Protocol
- FakeTT pilot: 100 videos, balanced labels (50 fake / 50 real).
- Proprietary pilot: 100 videos with external weak-reference signal.
- Metrics: Accuracy, Precision/Recall/F1, MAE, Pearson correlation.
Speaker note: clarify that proprietary pilot is weak supervision, not final ground truth.

### Slide 9 - Pilot Results on FakeTT
- TP 40, TN 17, FP 33, FN 10.
- Accuracy 0.57, Precision 0.548, Recall 0.80, F1 0.650.
- Main pattern: high recall but elevated false positives.
Speaker note: model is sensitive, but calibration is currently conservative.

### Slide 10 - Pilot Results on Proprietary Subset
- Mean external score: 35.9; mean model score: 62.5.
- MAE: 37.6, Pearson r: 0.354.
- 69/100 samples had scene_count = 0 in current preprocessing.
Speaker note: this is the key engineering bottleneck and immediate optimization target.

### Slide 11 - Failure Modes and What We Learned
- Scene detection failures in short compressed clips.
- OCR noise in low-quality overlays.
- ASR degradation in multilingual/noisy audio.
- Ambiguity between editorial framing and manipulation.
Speaker note: these are tractable pipeline issues, not a failure of the core hypothesis.

### Slide 12 - Execution Plan to Defense
- Weeks 1-2: stabilize segmentation and extraction.
- Weeks 3-4: calibrate fusion and evaluate hybrid pipeline.
- Weeks 5-6: implement transformer baseline.
- Weeks 7-8: ablations, final symposium package.
Speaker note: close this slide with concrete deliverables and risk controls.

### Slide 13 - Contributions and Expected Impact
- Conceptual: shift from artifact detection to narrative verification.
- Technical: dual-track architecture with weakly supervised scaling path.
- Practical: tool direction for detecting real cheapfakes in short-form media.
Speaker note: connect impact to fact-checking organizations and platform integrity.

### Slide 14 - Conclusion and Questions
- Narrative manipulation is the dominant modern threat in social short video.
- Scene-level multimodal verification is feasible and promising.
- Current pilots expose bottlenecks and define a clear path to robust results.
Speaker note: finish with 2 open questions to invite discussion.

## Backup Slides to Prepare (not in main flow)
- Prompt template used for LLM consistency scoring.
- Detailed threshold tuning for scene segmentation.
- Example false positives and false negatives.
- Additional architecture ablations.
- Ethical and dual-use risk considerations.
