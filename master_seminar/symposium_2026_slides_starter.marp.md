---
marp: true
theme: default
paginate: true
size: 16:9
---

# Scene-Level Narrative Verification
## Detecting Manipulation in Short-Form Social Media Videos

**Andrii Hupalo**  
Ukrainian Catholic University

Problem: pixel-level detectors miss cheapfakes built from authentic footage.

---

# Why Existing Forensics Miss Cheapfakes

- Deepfake detectors focus on synthetic artifacts.
- Cheapfakes alter meaning, not necessarily pixels.
- Real footage + misleading caption/narration can bypass artifact checks.

**Research gap:** no robust scene-level cross-modal narrative verification.

---

# Research Questions

- **RQ1:** Can scene segmentation recover narrative units?
- **RQ2:** Hybrid CV+LLM vs end-to-end transformer: what works better?
- **RQ3:** Can enriched metadata support weakly supervised learning?
- **RQ4:** Are multimodal LLMs reliable enough for forensic reasoning?

---

# Data Foundation

- Proprietary corpus: ~7.4k multilingual TikTok videos.
- Duration: 15-60s (median 28s).
- Scene transitions: 3-8 per video (median 4).
- Enrichment: OCR, ASR, visual descriptions, embeddings, weak-reference signals.

**Constraint:** NDA, aggregated reporting only.

---

# Problem Formulation

For each narrative segment $N_i$, extract:
- visual representation $v_i$
- audio representation $a_i$
- text representation $t_i$

Compute consistency:

$$
C(N_i)=sim(v_i,t_i)+sim(a_i,t_i)+sim(v_i,a_i)
$$

Predict video-level manipulation score $M\in[0,1]$.

---

# Hybrid CV+LLM Pipeline

1. Adaptive scene segmentation
2. Multimodal extraction (visual, OCR, ASR)
3. Scene-level LLM consistency reasoning
4. Weighted aggregation to video-level score

Key strength: interpretable scene-level justifications.

---

# End-to-End Transformer Alternative

- Joint multimodal encoding (visual + audio + text)
- Auxiliary tasks: segmentation + consistency learning
- Potentially stronger with enough training data

Trade-off:
- Higher compute
- Lower interpretability

---

# Pilot Evaluation Protocol

**FakeTT pilot**
- 100 videos (50 fake / 50 real)

**Proprietary pilot**
- 100 videos with weak-reference external score

**Metrics**
- Accuracy, Precision, Recall, F1
- MAE, Pearson correlation

---

# Pilot Results: FakeTT

- TP 40, TN 17, FP 33, FN 10
- Accuracy: 0.57
- Precision: 0.548
- Recall: 0.80
- F1: 0.650

Interpretation: strong sensitivity, but over-flagging (false positives).

---

# Pilot Results: Proprietary Subset

- Mean external score: 35.9
- Mean model score: 62.5
- MAE: 37.6
- Pearson $r = 0.354$
- 69/100 samples with scene_count = 0

Main bottleneck: segmentation robustness in short/noisy clips.

---

# Failure Modes

- Scene detection misses in compressed short clips
- OCR noise in low-quality overlays
- ASR errors in multilingual/noisy speech
- Ambiguity in contextual framing

These are engineering bottlenecks, not evidence against the core formulation.

---

# 8-Week Execution Plan

- Weeks 1-2: segmentation and extraction stabilization
- Weeks 3-4: hybrid calibration and evaluation
- Weeks 5-6: transformer baseline implementation
- Weeks 7-8: ablations and symposium final package

---

# Contributions and Impact

- Shift from pixel artifacts to narrative consistency verification
- Dual-track architecture for interpretable and scalable detection
- Practical path for cheapfake detection in real social media settings

---

# Conclusion

- Narrative manipulation is now the dominant threat.
- Scene-level multimodal verification is feasible.
- Pilot results expose clear bottlenecks and next actions.

## Questions?

---

# Back-Up Slide 1: Segmentation Fallback Strategy

```text
						Video Input
								 |
			Primary: PySceneDetect
				(Threshold = 27)
								 |
				scene_count > 0 ?
						/         \
					Yes          No
					 |            |
 Rapid cuts captured    Fallback: Fixed Temporal Windows
					 |            Triggered if scene_count = 0
					 |            (e.g., strict 5-second segments)
						\          /
						 \        /
	 Continuous narrative unit extraction
									|
				 Units passed to LLM reasoning
```

**Primary:** PySceneDetect (Threshold = 27), captures rapid cuts.

**Fallback:** Fixed Temporal Windows, triggered if `scene_count = 0` (e.g., strict 5-second segments).

**Goal:** Continuous narrative unit extraction.

**What to say (Script):**
"When PySceneDetect fails due to hard cuts or artifacts, we don't just discard the video. Our fallback strategy, which we are implementing now, forces a fixed temporal window, for instance 5 seconds. This guarantees we always extract narrative units for the LLM to process."
