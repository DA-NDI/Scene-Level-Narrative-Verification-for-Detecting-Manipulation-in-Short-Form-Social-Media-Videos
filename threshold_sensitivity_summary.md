# Threshold Sensitivity Analysis Summary

## Overview
Analysis of decision thresholds (20–90) on FakeTT dataset for Configs A–G. Each config shows F1-optimal threshold and corresponding precision-recall metrics.

## F1-Optimal Thresholds by Config

| Config | Model | Optimal Threshold | F1 | Precision | Recall | Accuracy | TP | TN | FP | FN |
|--------|-------|-------------------|----|-----------|--------|----------|----|----|----|----|
| **F** | gemini-3.1-flash-lite-preview | 30 | **0.701** | 0.552 | 0.960 | 0.59 | 48 | 11 | 39 | 2 |
| **D** | gemini-2.5-flash-lite | 30 | **0.681** | 0.521 | 0.980 | 0.54 | 49 | 5 | 45 | 1 |
| **B** | gemini-2.5-flash | 60 | 0.630 | 0.519 | 0.800 | 0.53 | 40 | 13 | 37 | 10 |
| **G** | gemini-2.5-pro | 30 | 0.651 | 0.532 | 0.840 | 0.55 | 42 | 13 | 37 | 8 |
| **E** | gemini-3-flash-preview | 20 | 0.662 | 0.538 | 0.860 | 0.56 | 43 | 13 | 37 | 7 |
| **C** | gemini-2.5-flash + CLIP | 20 | 0.622 | 0.494 | 0.840 | 0.49 | 42 | 7 | 43 | 8 |
| **A** | gemini-2.5-flash (no vision) | 20 | 0.544 | 0.484 | 0.620 | 0.48 | 31 | 17 | 33 | 19 |

## Key Findings

### 1. **Config F Achieves Best F1 (0.701)**
- Optimal threshold: 30
- Highest recall (0.960) with reasonable precision (0.552)
- Only 2 false negatives (missed fakes) across 100 samples
- Consistent across thresholds 30–80 (F1 plateau ~0.645)

### 2. **Config D: High Recall Tradeoff (F1=0.681)**
- Optimal threshold: 30
- Nearly perfect recall (0.980) but high FP rate (45 false positives)
- Rapid F1 degradation at thresholds >70

### 3. **Precision-Recall Stability**
- **Configs F, G, E**: Maintain F1 ≥ 0.645 across threshold range 30–70
- **Configs B, C**: Peak F1 at lower thresholds (20–30) with sharp degradation
- **Config A**: Unstable; scores max out at 30, F1 collapses for threshold ≥30

### 4. **Score Distribution Insights**
| Config | Score Mean | Score Std | Min | Max |
|--------|-----------|-----------|-----|-----|
| A | 21.28 | 7.82 | 5 | 30 |
| B | 73.16 | 29.47 | 10 | 100 |
| C | 70.45 | 30.81 | 10 | 100 |
| D | 63.22 | 18.48 | 10 | 90 |
| E | 70.65 | 33.75 | 5 | 100 |
| **F** | **74.30** | **30.63** | 10 | 100 |
| G | 74.10 | 33.49 | 5 | 98 |

**Config A** has drastically lower score range (5–30), indicating weak model discriminability.

### 5. **Optimal Operating Point**
For production deployment, **threshold 30 with Config F** offers:
- **Highest F1**: 0.701
- **High Recall**: 0.960 (catches 96% of fake videos)
- **Reasonable Precision**: 0.552 (1 in 2 positive predictions is correct)
- This tradeoff prioritizes recall (minimizing false negatives = missed fakes)

Alternative: **Threshold 90 with Config F** gives higher precision (0.674) but sacrifices recall (0.62) and F1 (0.646).

## Integration into Thesis

### Section 5.X Recommendation
Add subsection on threshold sensitivity showing:
1. Table above (F1-optimal metrics for all configs)
2. Precision-recall curve visualization (provided in PNG)
3. Interpretation: Config F's robust performance across 30–80 threshold range supports RQ2/RQ3 findings about model generation and modality contributions

### Sentence for Chapter 5
*"Threshold sensitivity analysis (Figure X.X, Table X.X) reveals that Config F maintains F1 ≥ 0.645 across thresholds 30–80, indicating robust decision boundaries. At the F1-optimal threshold of 30, Config F achieves F1=0.701 with 96% recall, compared to Config A's peak F1 of 0.544, demonstrating the critical importance of model architecture (generation and base model selection) in both discrimination ability and calibration."*
