"""
Step 5: Evaluation & Visualization
Generate Pareto curve and comparison plots
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots"

print("="*80)
print("EVALUATION & VISUALIZATION")
print("="*80)

# Load results
print("\n[1/3] Loading results...")
with open(f"{RESULTS_DIR}/baseline_svd_results.json", 'r') as f:
    baseline = json.load(f)

with open(f"{RESULTS_DIR}/multiobjective_results.json", 'r') as f:
    multiobj = json.load(f)

with open(f"{RESULTS_DIR}/pareto_data.json", 'r') as f:
    pareto = json.load(f)

print("✓ Loaded all results")

# ==============================================================================
# PLOT 1: PARETO CURVE (RMSE vs Health Trade-off)
# ==============================================================================
print("\n[2/3] Generating Pareto curve...")

plt.figure(figsize=(10, 6))
alphas = pareto['alpha_values']
rmse = pareto['rmse_values']
health = pareto['health_values']

plt.plot(rmse, health, 'o-', linewidth=2, markersize=8, color='steelblue')

# Annotate points
for i, alpha in enumerate(alphas):
    plt.annotate(f'α={alpha:.2f}', (rmse[i], health[i]),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('RMSE (Lower is Better)', fontsize=12)
plt.ylabel('Average Health Score (Higher is Better)', fontsize=12)
plt.title('Pareto Curve: Preference vs Health Trade-off', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

pareto_path = f"{PLOTS_DIR}/pareto_curve.png"
plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {pareto_path}")
plt.close()

# ==============================================================================
# PLOT 2: ALPHA COMPARISON
# ==============================================================================
print("\n[3/3] Generating alpha comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# RMSE vs Alpha (removed baseline dashed line)
ax1.plot(alphas, rmse, 'o-', linewidth=2.5, markersize=8, color='coral')
ax1.set_xlabel('α (Preference Weight)', fontsize=11)
ax1.set_ylabel('RMSE (Lower is Better)', fontsize=11)
ax1.set_title('RMSE vs α', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add value annotations to RMSE plot
for i, (x, y) in enumerate(zip(alphas, rmse)):
    ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=9)

# Health vs Alpha (with value annotations)
ax2.plot(alphas, health, 'o-', linewidth=2.5, markersize=8, color='seagreen')
ax2.set_xlabel('α (Preference Weight)', fontsize=11)
ax2.set_ylabel('Avg Health Score (Higher is Better)', fontsize=11)
ax2.set_title('Health Score vs α', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add value annotations to Health plot
for i, (x, y) in enumerate(zip(alphas, health)):
    ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=9)

plt.tight_layout()
alpha_path = f"{PLOTS_DIR}/alpha_comparison.png"
plt.savefig(alpha_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {alpha_path}")
plt.close()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nGenerated plots:")
print(f"  1. {pareto_path}")
print(f"  2. {alpha_path}")
print(f"\nKey Findings:")
print(f"  • Baseline RMSE (α=1.0): {baseline['test_metrics']['RMSE']:.4f}")
print(f"  • Best health (α=0.0): {health[0]:.4f}")
print(f"  • Balanced (α=0.5): RMSE={rmse[2]:.4f}, Health={health[2]:.4f}")
print("\nAll results ready for report!")
