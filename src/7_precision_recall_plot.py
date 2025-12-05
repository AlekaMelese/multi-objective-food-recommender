"""
Additional Visualization: Precision/Recall and Top-K Health Analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"

print("="*80)
print("GENERATING PRECISION/RECALL VISUALIZATION")
print("="*80)

# Load Top-K evaluation results
print("\n[1/2] Loading Top-K evaluation data...")
with open(f"{RESULTS_DIR}/topk_evaluation.json", 'r') as f:
    topk_data = json.load(f)

alpha_values = topk_data['alpha_values']
results = topk_data['results']

# Extract metrics
precision = [r['precision_at_k'] for r in results]
recall = [r['recall_at_k'] for r in results]
f1_scores = [r['f1_score'] for r in results]
health_topk = [r['avg_health_topk'] for r in results]

print("✓ Data loaded")

# Create comprehensive visualization
print("\n[2/2] Creating Precision/Recall plot...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Top-K Recommendation Quality Analysis', fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Precision@10 vs α
ax1 = axes[0]
ax1.plot(alpha_values, precision, 'o-', linewidth=2.5, markersize=8, color='#2E86AB', label='Precision@10')
ax1.set_xlabel('α (Preference Weight)', fontsize=11)
ax1.set_ylabel('Precision@10', fontsize=11)
ax1.set_title('Recommendation Accuracy: Precision@10', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(precision) * 1.2])
for i, (x, y) in enumerate(zip(alpha_values, precision)):
    ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

# Plot 2: Recall@10 vs α
ax2 = axes[1]
ax2.plot(alpha_values, recall, 'o-', linewidth=2.5, markersize=8, color='#A23B72', label='Recall@10')
ax2.set_xlabel('α (Preference Weight)', fontsize=11)
ax2.set_ylabel('Recall@10', fontsize=11)
ax2.set_title('Recommendation Coverage: Recall@10', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max(recall) * 1.2])
for i, (x, y) in enumerate(zip(alpha_values, recall)):
    ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

# Plot 3: F1-Score vs α
ax3 = axes[2]
ax3.plot(alpha_values, f1_scores, 'o-', linewidth=2.5, markersize=8, color='#F18F01', label='F1-Score')
ax3.set_xlabel('α (Preference Weight)', fontsize=11)
ax3.set_ylabel('F1-Score', fontsize=11)
ax3.set_title('Overall Quality: F1-Score', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, max(f1_scores) * 1.2])
for i, (x, y) in enumerate(zip(alpha_values, f1_scores)):
    ax3.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

# Add insight box
insight_text = (
    "Key Insights:\n"
    "• Precision/Recall/F1 increase slightly with α (better preference matching)\n"
    f"• Low Precision/Recall ({precision[0]:.4f}-{precision[-1]:.4f}) due to dataset sparsity (99.57%)"
)
fig.text(0.5, -0.05, insight_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save plot
output_path = f"{PLOTS_DIR}/precision_recall_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nGenerated plot: {output_path}")
print("\nThis plot shows:")
print("  • Precision@10: Accuracy of Top-10 recommendations")
print("  • Recall@10: Coverage of relevant items")
print("  • F1-Score: Harmonic mean of Precision & Recall")
