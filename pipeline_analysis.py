"""
Analyze pipeline_results.json — compare our LLM pipeline vs commercial scores.
Run after pipeline_eval.py completes (or partially, to check progress).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

with open('pipeline_results.json') as f:
    results = json.load(f)

# Filter out failed Gemini calls
valid = [r for r in results if r['our_score'] is not None]
print(f"Total results: {len(results)} | Valid (Gemini succeeded): {len(valid)}")

commercial = np.array([r['commercial_score'] for r in valid])
ours       = np.array([r['our_score']        for r in valid])
scenes     = np.array([r['scene_count']      for r in valid])

# ── Statistics ───────────────────────────────────────────────────────
pearson_r, pearson_p = stats.pearsonr(commercial, ours)
spearman_r, spearman_p = stats.spearmanr(commercial, ours)
mae = np.mean(np.abs(commercial - ours))

print(f"\n── Correlation Results ──────────────────────────────")
print(f"Pearson  r = {pearson_r:.3f}  (p={pearson_p:.4f})")
print(f"Spearman r = {spearman_r:.3f}  (p={spearman_p:.4f})")
print(f"MAE (mean absolute error): {mae:.1f} points")

# Binary agreement: threshold at 50
comm_binary = (commercial >= 50).astype(int)
our_binary  = (ours >= 50).astype(int)
agreement   = np.mean(comm_binary == our_binary)
print(f"\nBinary agreement (threshold=50): {agreement*100:.1f}%")

# Per-class
from collections import Counter
tp = np.sum((comm_binary == 1) & (our_binary == 1))
tn = np.sum((comm_binary == 0) & (our_binary == 0))
fp = np.sum((comm_binary == 0) & (our_binary == 1))
fn = np.sum((comm_binary == 1) & (our_binary == 0))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")

# ── Plot 1: Scatter — our score vs commercial ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(commercial, ours, alpha=0.5, color='steelblue', edgecolors='none', s=50)
# Regression line
m, b = np.polyfit(commercial, ours, 1)
x_line = np.linspace(0, 100, 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2, label=f'Fit (r={pearson_r:.2f})')
ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.4, label='Perfect agreement')
ax.set_xlabel("Commercial Score (d-tiktok)", fontsize=12)
ax.set_ylabel("Our Pipeline Score (Gemini)", fontsize=12)
ax.set_title(f"Our Pipeline vs Commercial Score\n(n={len(valid)}, Pearson r={pearson_r:.2f})", fontsize=13)
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# ── Plot 2: Error distribution ────────────────────────────────────────
ax2 = axes[1]
errors = ours - commercial
sns.histplot(errors, bins=25, kde=True, color='steelblue', ax=ax2)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel("Our Score − Commercial Score", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title(f"Score Difference Distribution\nMAE = {mae:.1f} pts", fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_pipeline_comparison.png', dpi=150)
plt.close()
print("\nSaved: plot_pipeline_comparison.png")

# ── Plot 3: Confusion matrix heatmap ─────────────────────────────────
cm = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Authentic', 'Predicted: Manipulated'],
            yticklabels=['Actual: Authentic', 'Actual: Manipulated'])
plt.title(f"Confusion Matrix (threshold=50)\nF1={f1:.2f}, Agreement={agreement*100:.0f}%")
plt.tight_layout()
plt.savefig('plot_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: plot_confusion_matrix.png")

# ── Scene count correlation ───────────────────────────────────────────
sc_r, sc_p = stats.spearmanr(scenes, commercial)
print(f"\nScene count vs commercial score: Spearman r={sc_r:.3f} (p={sc_p:.4f})")

# ── Sample of interesting cases ───────────────────────────────────────
print("\n── Cases where we disagree most (top 5 largest error) ──")
sorted_by_err = sorted(valid, key=lambda r: abs(r['our_score'] - r['commercial_score']), reverse=True)
for r in sorted_by_err[:5]:
    diff = r['our_score'] - r['commercial_score']
    print(f"  id={r['id'][:8]}… commercial={r['commercial_score']} our={r['our_score']} diff={diff:+.0f}")
    print(f"  reasoning: {r['reasoning'][:120]}")
