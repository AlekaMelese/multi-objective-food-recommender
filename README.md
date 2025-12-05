# Multi-Objective Food Recommendation System

Balancing User Preference and Health in Recipe Recommendations

## Quick Start

### 1. Run Baseline Model (Preference Only)



This trains a Matrix Factorization (SVD) model that **optimizes only for user ratings**.

### 2. Check Results
```bash
cat "../results/baseline_results.json"
```

You should see the Test RMSE (lower is better) - typically around 0.8-1.0 for this dataset.

---

## What We're Building

Traditional recommenders only care about what users **like**, which can lead to unhealthy recommendations.

This project builds a **multi-objective** system that balances:
- ✅ **User Preference** (predicted rating)
- ✅ **Health Score** (WHO-aligned nutrition score)

### The Trade-off Formula
```
Score(user, recipe) = α · Preference(user, recipe) + (1-α) · Health(recipe)
```

- α = 1.0 → Only preference (baseline)
- α = 0.5 → Balanced
- α = 0.0 → Only health

---

## Dataset

- **281 recipes** with complete nutritional data
- **28,491 user ratings** (1-5 stars)
- **WHO health scores** (0-1, higher is healthier)

Split: 80% train, 10% validation, 10% test

---

## Project Status

- ✅ Data exploration (Step 1)
- ✅ Data preparation (Step 2)
- ✅ Baseline model (Step 3 - SVD)
- ✅ Multi-objective model (Step 4)
- ✅ Evaluation & visualization (Step 5-8)
- ✅ Top-K evaluation: Precision/Recall/Health (Step 6) ← **COMPLETE**

---

## Detailed Instructions

See [RUN_PROJECT.md](RUN_PROJECT.md) for complete terminal commands and explanations.

---

## Results Summary

✅ **Baseline Model (Preference Only)**
- Model: Truncated SVD (50 components with mean-centering)
- Test RMSE: **0.9056** (on 1-5 scale)
- Test MAE: 0.6702
- Significant improvement over NMF (previous RMSE: 3.6491)

✅ **Multi-Objective Model Results**

| α Value | Preference Weight | Health Weight | Test RMSE | Avg Health (Top-10) |
|---------|-------------------|---------------|-----------|---------------------|
| 0.00    | 0%                | 100%          | 1.5681    | 0.9137              |
| 0.25    | 25%               | 75%           | 1.3115    | 0.9137              |
| 0.50    | 50%               | 50%           | 1.0958    | 0.9136              |
| 0.75    | 75%               | 25%           | 0.9490    | 0.9089              |
| 1.00    | 100%              | 0%            | 0.9056    | 0.5868              |

**Key Findings:**
- As preference weight increases (α→1.0), RMSE decreases (better rating prediction) but health quality of recommendations drops dramatically (0.9137 → 0.5868, a 36% reduction)
- Clear Pareto trade-off: Better preference accuracy comes at the cost of health optimization
- Balanced approach (α=0.5) maintains 99.9% of maximum health while achieving moderate prediction performance

✅ **Visualizations**
- Pareto curve showing preference vs health trade-off (plots/pareto_curve.png)
- Alpha comparison plots (plots/alpha_comparison.png)
- Precision/Recall/F1-Score analysis (plots/precision_recall_analysis.png)

---


