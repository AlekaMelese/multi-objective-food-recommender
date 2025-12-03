# Multi-Objective Food Recommendation - Run Instructions

## Project Structure
```
Project 2/
├── data/                    # Processed datasets (created by step 2)
├── src/                     # Source code
├── models/                  # Trained models (saved here)
├── results/                 # Evaluation results (saved here)
├── plots/                   # Visualizations (saved here)
├── requirements.txt         # Python dependencies
└── RUN_PROJECT.md          # This file
```

## Terminal Commands to Run

### Step 1: Data Exploration
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2"
/scratch/aayalew25/Aleka/alekapy/bin/python 1_explore_data.py
```
**Output:** Console output showing dataset statistics

---

### Step 2: Data Preparation
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2"
/scratch/aayalew25/Aleka/alekapy/bin/python 2_prepare_data.py
```
**Output:**
- `data/recipes.csv` - 281 recipes with nutritional data
- `data/train_ratings.csv` - Training set (80%)
- `data/val_ratings.csv` - Validation set (10%)
- `data/test_ratings.csv` - Test set (10%)
- `data/data_summary.json` - Statistics

---

### Step 3: Baseline Model (SVD - Preference Only)
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2/src"
/scratch/aayalew25/Aleka/alekapy/bin/python 3_baseline_model.py
```
**Output:**
- `models/baseline_svd.pkl` - Trained baseline model
- `results/baseline_results.json` - Performance metrics
- `results/baseline_sample_predictions.json` - Example predictions

**What this does:**
- Trains Matrix Factorization (SVD) model
- Optimizes ONLY for rating prediction (user preference)
- Does NOT consider health scores
- Baseline for comparison

---

### Step 4: Multi-Objective Model (Preference + Health)
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2/src"
/scratch/aayalew25/Aleka/alekapy/bin/python 4_multiobjective_model.py
```
**Output:**
- `models/multiobjective_*.pkl` - Models for different α values
- `results/multiobjective_results.json` - Performance for all α values
- `results/pareto_data.json` - Trade-off curve data

**What this does:**
- Tests multiple α values (0.0, 0.25, 0.5, 0.75, 1.0)
- Formula: Score = α·Preference + (1-α)·Health
- Shows trade-offs between preference and health

---

### Step 5: Evaluation & Visualization
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2/src"
/scratch/aayalew25/Aleka/alekapy/bin/python 5_evaluate_visualize.py
```
**Output:**
- `plots/pareto_curve.png` - Trade-off visualization
- `plots/alpha_comparison.png` - Performance across α values
- `plots/health_improvement.png` - Health score comparison
- `results/evaluation_summary.json` - Complete metrics

**What this does:**
- Generates Pareto curve showing trade-offs
- Compares baseline vs multi-objective
- Creates visualizations for report

---

### Step 6: Top-K Evaluation (Precision, Recall, Health)
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2/src"
/scratch/aayalew25/Aleka/alekapy/bin/python 6_topk_evaluation.py
```
**Output:**
- `results/topk_evaluation.json` - Precision@10, Recall@10, F1-Score, Avg Health (Top-10)

**What this does:**
- Evaluates recommendation quality with Precision/Recall metrics
- Measures healthfulness of Top-10 recommendations
- Shows trade-off between accuracy and health

---

## Quick Run: All Steps at Once
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2/src"

# Run all steps sequentially
/scratch/aayalew25/Aleka/alekapy/bin/python 3_baseline_model_sklearn.py && \
/scratch/aayalew25/Aleka/alekapy/bin/python 4_multiobjective_model.py && \
/scratch/aayalew25/Aleka/alekapy/bin/python 5_evaluate_visualize.py && \
/scratch/aayalew25/Aleka/alekapy/bin/python 6_topk_evaluation.py
```

---

## Expected Timeline
- **Step 3 (Baseline):** ~5 seconds
- **Step 4 (Multi-objective):** ~5 seconds
- **Step 5 (Visualization):** ~2 seconds
- **Step 6 (Top-K Evaluation):** ~30 seconds

**Total:** ~15-20 minutes

---

## Check Your Results
After running all steps, you should have:

### Models (`models/` folder)
- `baseline_svd.pkl`
- `multiobjective_alpha_0.0.pkl` (health only)
- `multiobjective_alpha_0.25.pkl`
- `multiobjective_alpha_0.5.pkl` (balanced)
- `multiobjective_alpha_0.75.pkl`
- `multiobjective_alpha_1.0.pkl` (preference only)

### Results (`results/` folder)
- `baseline_results.json`
- `multiobjective_results.json`
- `evaluation_summary.json`
- `pareto_data.json`

### Visualizations (`plots/` folder)
- `pareto_curve.png`
- `alpha_comparison.png`
- `health_improvement.png`

---

## Troubleshooting

### Issue: "No module named 'surprise'"
**Solution:**
```bash
/scratch/aayalew25/Aleka/alekapy/bin/pip install scikit-surprise
```

### Issue: "FileNotFoundError: data/train_ratings.csv"
**Solution:** Run step 2 first to prepare the data
```bash
cd "/scratch/aayalew25/Food/FoodNew/Project 2"
/scratch/aayalew25/Aleka/alekapy/bin/python 2_prepare_data.py
```

### Issue: Models taking too long
**Solution:** Reduce epochs in the code or use smaller parameter grid

---

## Next: Generate Report
After all steps complete, you'll have all data needed for the 3-4 page report showing:
1. Baseline performance (preference only)
2. Multi-objective results (various α values)
3. Pareto curve (trade-off visualization)
4. Health improvement analysis
5. Ethical implications discussion
