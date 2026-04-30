"""
EDA plots for the master seminar presentation.
Analyzes each dataset separately (no incorrect merge).
"""
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Load datasets separately ────────────────────────────────────────
with open('a-tiktok_20251105_20251206.json') as f:
    df_a = pd.DataFrame(json.load(f))

with open('d-tiktok_20251129_20251206.json') as f:
    df_d = pd.DataFrame(json.load(f))

print(f"a-tiktok: {len(df_a)} records | d-tiktok: {len(df_d)} records")

# ────────────────────────────────────────────────────────────────────
# PLOT 1: Manipulation score distribution (d-tiktok, full 5727 posts)
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Overview: d-tiktok (n=5,727)", fontsize=14)

ax = axes[0]
sns.histplot(df_d['manipulation_score'].dropna(), bins=20, kde=True,
             color='steelblue', ax=ax)
ax.set_title("Manipulation Score Distribution")
ax.set_xlabel("Manipulation Score")
ax.set_ylabel("Count")

ax2 = axes[1]
sns.histplot(df_d['deepfake_score'].dropna(), bins=15, kde=True,
             color='tomato', ax=ax2)
ax2.set_title("Deepfake Score Distribution")
ax2.set_xlabel("Deepfake Score")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig('plot_score_distribution.png', dpi=150)
plt.close()
print("Saved: plot_score_distribution.png")

# ────────────────────────────────────────────────────────────────────
# PLOT 2: Manipulation score vs video length (d-tiktok)
# ────────────────────────────────────────────────────────────────────
df_d_clean = df_d[['video_length', 'manipulation_score']].dropna()
df_d_clean = df_d_clean[df_d_clean['video_length'] < 300]  # exclude outliers >5min

plt.figure(figsize=(10, 6))
sns.regplot(data=df_d_clean, x='video_length', y='manipulation_score',
            scatter_kws={'alpha': 0.2, 'color': 'steelblue'},
            line_kws={'color': 'red'})
plt.title("Narrative Manipulation Score vs. Video Duration (n=5,727)")
plt.xlabel("Video Length (seconds)")
plt.ylabel("Manipulation Score")
plt.tight_layout()
plt.savefig('plot_manipulation_vs_length.png', dpi=150)
plt.close()
print("Saved: plot_manipulation_vs_length.png")

# ────────────────────────────────────────────────────────────────────
# PLOT 3: Engagement by manipulation level (d-tiktok)
# ────────────────────────────────────────────────────────────────────
df_d['manipulation_level'] = pd.cut(
    df_d['manipulation_score'],
    bins=[-1, 30, 60, 100],
    labels=['Low (0–30)', 'Medium (31–60)', 'High (61–100)']
)
df_eng = df_d[['manipulation_level', 'engagement_rate']].dropna()

plt.figure(figsize=(9, 6))
sns.boxplot(data=df_eng, x='manipulation_level', y='engagement_rate',
            palette=['#2ecc71', '#f39c12', '#e74c3c'])
plt.yscale('log')
plt.title("Engagement Rate by Manipulation Level (log scale, n=5,727)")
plt.xlabel("Manipulation Level")
plt.ylabel("Engagement Rate (log)")
plt.tight_layout()
plt.savefig('plot_engagement_boxplot.png', dpi=150)
plt.close()
print("Saved: plot_engagement_boxplot.png")

# ────────────────────────────────────────────────────────────────────
# PLOT 4: Correlation heatmap (d-tiktok numeric fields)
# ────────────────────────────────────────────────────────────────────
numeric_cols = ['manipulation_score', 'deepfake_score', 'engagement_rate',
                'views_count', 'video_length', 'reactions_count', 'shares_count']
df_corr = df_d[numeric_cols].dropna()

plt.figure(figsize=(9, 7))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f",
            square=True, linewidths=0.5)
plt.title("Correlation Matrix — d-tiktok Video Metrics")
plt.tight_layout()
plt.savefig('plot_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: plot_correlation_heatmap.png")

# ────────────────────────────────────────────────────────────────────
# PLOT 5: a-tiktok actor category breakdown (your labeled subset)
# ────────────────────────────────────────────────────────────────────
cat_counts = df_a['category'].value_counts()

plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#3498db', '#e74c3c']
cat_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title("Actor Category Distribution — a-tiktok (n=1,705)")
plt.xlabel("Actor Category")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
for i, v in enumerate(cat_counts):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_actor_categories.png', dpi=150)
plt.close()
print("Saved: plot_actor_categories.png")

# ────────────────────────────────────────────────────────────────────
# PLOT 6: Manipulation score — Malign vs Neutral actors (a-tiktok)
# ────────────────────────────────────────────────────────────────────
df_a_labeled = df_a[df_a['category'].isin(['Malign actor', 'Neutral actor'])][
    ['category', 'manipulation_score']].dropna()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_a_labeled, x='category', y='manipulation_score',
            palette={'Malign actor': '#e74c3c', 'Neutral actor': '#2ecc71'})
plt.title("Manipulation Score: Malign vs Neutral Actors (a-tiktok)")
plt.xlabel("")
plt.ylabel("Manipulation Score")
plt.tight_layout()
plt.savefig('plot_malign_vs_neutral.png', dpi=150)
plt.close()
print("Saved: plot_malign_vs_neutral.png")

# ── Print summary stats ─────────────────────────────────────────────
print("\n── Dataset Summary ────────────────────────────────")
print(f"d-tiktok manipulation_score mean:   {df_d['manipulation_score'].mean():.1f}")
print(f"d-tiktok manipulation_score median: {df_d['manipulation_score'].median():.1f}")
print(f"d-tiktok high manipulation (>60):   {(df_d['manipulation_score'] > 60).sum()} posts")
print(f"a-tiktok malign actor posts:        {(df_a['category'] == 'Malign actor').sum()}")
print(f"a-tiktok malign avg manip score:    {df_a[df_a['category']=='Malign actor']['manipulation_score'].mean():.1f}")
print(f"a-tiktok neutral avg manip score:   {df_a[df_a['category']=='Neutral actor']['manipulation_score'].mean():.1f}")
