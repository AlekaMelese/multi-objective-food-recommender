"""
Step 4: Multi-Objective Model
Balance user preference AND health using weighted formula:
Score(u,i) = α·Preference(u,i) + (1-α)·Health(i)
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
DATA_DIR = "../data"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

print("="*80)
print("MULTI-OBJECTIVE MODEL: PREFERENCE + HEALTH")
print("="*80)

# ==============================================================================
# 1. LOAD BASELINE MODEL AND DATA
# ==============================================================================
print("\n[1/4] Loading baseline model and data...")

# Load baseline SVD model (better than NMF)
with open(f"{MODELS_DIR}/baseline_svd.pkl", 'rb') as f:
    model_data = pickle.load(f)

U = model_data['U']
Vt = model_data['Vt']
global_mean = model_data['global_mean']
user_to_idx = model_data['user_to_idx']
recipe_to_idx = model_data['recipe_to_idx']

# Reconstruct predicted rating matrix (1-5 scale from baseline)
predicted_matrix = np.dot(U, Vt) + global_mean
# Clip to valid rating range [1, 5]
predicted_matrix = np.clip(predicted_matrix, 1, 5)
# Normalize to [0, 1] for combining with health scores
predicted_matrix_norm = (predicted_matrix - 1) / 4  # Convert 1-5 to 0-1

# Load data
recipes_df = pd.read_csv(f"{DATA_DIR}/recipes.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_ratings.csv")

print(f"✓ Loaded baseline model")
print(f"✓ Predicted matrix: {predicted_matrix.shape}")
print(f"✓ Test set: {len(test_df)} ratings")

# ==============================================================================
# 2. PREPARE HEALTH SCORES
# ==============================================================================
print("\n[2/4] Preparing health scores...")

# Create recipe_id -> health_score mapping
health_scores = {}
for _, row in recipes_df.iterrows():
    health_scores[row['recipe_id']] = row['health_score']

print(f"✓ Loaded health scores for {len(health_scores)} recipes")
print(f"  Health score range: [{min(health_scores.values()):.3f}, {max(health_scores.values()):.3f}]")

# ==============================================================================
# 3. EVALUATE MULTIPLE ALPHA VALUES
# ==============================================================================
print("\n[3/4] Testing multiple α values...")

# Test different trade-off weights
alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
results_by_alpha = {}

for alpha in alpha_values:
    print(f"\n  Testing α = {alpha:.2f} (Pref: {alpha:.0%}, Health: {(1-alpha):.0%})")

    # Calculate multi-objective scores for test set
    predictions = []
    actuals = []

    # NEW: Calculate average health of Top-K recommended items for each test user
    K = 10  # Top-10 recommendations
    test_users = test_df['member_id'].unique()
    recommended_health_scores = []

    for _, row in test_df.iterrows():
        user_id = row['member_id']
        recipe_id = row['recipe_id']
        actual_rating = row['rating']  # Use original 1-5 scale

        # Get predicted preference (from baseline model)
        user_idx = user_to_idx[user_id]
        recipe_idx = recipe_to_idx[recipe_id]
        pred_preference_norm = predicted_matrix_norm[user_idx, recipe_idx]  # Normalized 0-1

        # Get health score (0-1 scale)
        health_score_norm = health_scores[recipe_id]

        # Multi-objective score on normalized scale: α·Preference + (1-α)·Health
        multi_obj_score_norm = alpha * pred_preference_norm + (1 - alpha) * health_score_norm

        # Denormalize to 1-5 scale for RMSE calculation
        multi_obj_score = multi_obj_score_norm * 4 + 1  # Convert 0-1 to 1-5

        # RMSE measures how well multi-objective score predicts actual user ratings
        predictions.append(multi_obj_score)
        actuals.append(actual_rating)

    # Generate Top-K recommendations for each user to calculate average health
    for user_id in test_users:
        if user_id not in user_to_idx:
            continue

        user_idx = user_to_idx[user_id]

        # Score all recipes for this user
        recipe_scores = []
        for recipe_id in recipes_df['recipe_id']:
            if recipe_id not in recipe_to_idx:
                continue

            recipe_idx = recipe_to_idx[recipe_id]
            pred_preference_norm = predicted_matrix_norm[user_idx, recipe_idx]  # Normalized 0-1
            health_score = health_scores[recipe_id]

            # Multi-objective score (on normalized 0-1 scale)
            score = alpha * pred_preference_norm + (1 - alpha) * health_score

            recipe_scores.append({
                'recipe_id': recipe_id,
                'score': score,
                'health_score': health_score
            })

        # Get Top-K recommendations
        recipe_scores.sort(key=lambda x: x['score'], reverse=True)
        top_k = recipe_scores[:K]

        # Average health of Top-K
        avg_health_topk = np.mean([r['health_score'] for r in top_k])
        recommended_health_scores.append(avg_health_topk)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    avg_health = np.mean(recommended_health_scores)  # Average health of Top-K recommendations

    results_by_alpha[alpha] = {
        "alpha": alpha,
        "preference_weight": alpha,
        "health_weight": 1 - alpha,
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "avg_health_score": float(avg_health)
    }

    print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, Avg Health: {avg_health:.4f}")

# ==============================================================================
# 4. SAVE RESULTS
# ==============================================================================
print("\n[4/4] Saving results...")

# Save complete results
output = {
    "model_type": "Multi-Objective (NMF + Health)",
    "formula": "Score(u,i) = α·Preference(u,i) + (1-α)·Health(i)",
    "alpha_values_tested": alpha_values,
    "results": list(results_by_alpha.values())
}

results_path = f"{RESULTS_DIR}/multiobjective_results.json"
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"✓ Saved results to: {results_path}")

# Save Pareto data for plotting
pareto_data = {
    "alpha_values": alpha_values,
    "rmse_values": [results_by_alpha[a]["test_rmse"] for a in alpha_values],
    "health_values": [results_by_alpha[a]["avg_health_score"] for a in alpha_values]
}

pareto_path = f"{RESULTS_DIR}/pareto_data.json"
with open(pareto_path, 'w') as f:
    json.dump(pareto_data, f, indent=2)
print(f"✓ Saved Pareto data to: {pareto_path}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("MULTI-OBJECTIVE MODEL COMPLETE!")
print("="*80)

print("\nTrade-off Summary:")
print(f"{'Alpha':<8} {'Pref%':<8} {'Health%':<10} {'RMSE':<10} {'Avg Health':<12}")
print("-"*60)
for alpha in alpha_values:
    r = results_by_alpha[alpha]
    print(f"{alpha:<8.2f} {alpha*100:<8.0f} {(1-alpha)*100:<10.0f} {r['test_rmse']:<10.4f} {r['avg_health_score']:<12.4f}")

print(f"\nResults saved to: {RESULTS_DIR}/")
print("Next step: Generate visualizations (Pareto curve)")
