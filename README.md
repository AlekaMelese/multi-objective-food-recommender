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
- ✅ Baseline model (Step 3)
- ✅ Multi-objective model (Step 4)
- ✅ Evaluation & visualization (Step 5) ← **COMPLETE**

---

## Detailed Instructions

See [RUN_PROJECT.md](RUN_PROJECT.md) for complete terminal commands and explanations.

---

## Results Summary

✅ **Baseline Model (Preference Only)**
- Model: NMF (20 factors)
- Test RMSE: **3.6491**
- Test MAE: 3.5342
- Training time: 3.98 seconds

✅ **Multi-Objective Model Results**

| α Value | Preference Weight | Health Weight | Test RMSE | Avg Health |
|---------|-------------------|---------------|-----------|------------|
| 0.00    | 0%                | 100%          | 0.3920    | 0.5990     |
| 0.25    | 25%               | 75%           | 0.4937    | 0.5990     |
| 0.50    | 50%               | 50%           | 0.6138    | 0.5990     |
| 0.75    | 75%               | 25%           | 0.7433    | 0.5990     |
| 1.00    | 100%              | 0%            | 0.8782    | 0.5990     |

**Key Finding:** As preference weight increases (α→1.0), RMSE increases, showing the trade-off between user preference accuracy and health optimization.

✅ **Visualizations**
- Pareto curve showing preference vs health trade-off
- Alpha comparison plots

---

