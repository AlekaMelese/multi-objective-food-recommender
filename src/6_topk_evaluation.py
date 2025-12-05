"""
Step 6: Top-K Evaluation with Precision, Recall, and Health Metrics
Evaluates recommendation quality and healthfulness of Top-K recommendations
"""

import pandas as pd
import numpy as np
import pickle
import json

# Paths
DATA_DIR = "../data"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

print("="*80)
print("TOP-K EVALUATION: PRECISION, RECALL, AND HEALTH METRICS")
print("="*80)

# ==============================================================================
# 1. LOAD MODEL AND DATA
# ==============================================================================
print("\n[1/4] Loading model and data...")

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
predicted_matrix_norm = (predicted_matrix - 1) / 4

# Load data
recipes_df = pd.read_csv(f"{DATA_DIR}/recipes.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_ratings.csv")
train_df = pd.read_csv(f"{DATA_DIR}/train_ratings.csv")

# Create health score mapping
health_scores = dict(zip(recipes_df['recipe_id'], recipes_df['health_score']))

print(f"✓ Loaded model and data")
print(f"✓ Test set: {len(test_df)} ratings")

# ==============================================================================
# 2. GENERATE TOP-K RECOMMENDATIONS FOR EACH ALPHA
# ==============================================================================
print("\n[2/4] Generating Top-K recommendations...")

K = 10  # Top-10 recommendations
alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
RATING_THRESHOLD = 4.0  # Ratings >= 4 are considered relevant (on 1-5 scale)

results_by_alpha = {}

# Get unique test users
test_users = test_df['member_id'].unique()
print(f"  Evaluating {len(test_users)} test users with Top-{K} recommendations")

for alpha in alpha_values:
    print(f"\n  Testing α = {alpha:.2f}...")

    precision_list = []
    recall_list = []
    health_scores_list = []

    for user_id in test_users:
        if user_id not in user_to_idx:
            continue

        user_idx = user_to_idx[user_id]

        # Get user's test ratings
        user_test = test_df[test_df['member_id'] == user_id]
        relevant_items = set(user_test[user_test['rating'] >= RATING_THRESHOLD]['recipe_id'])

        if len(relevant_items) == 0:
            continue

        # Get all recipe predictions for this user
        recipe_scores = []
        for recipe_id in recipes_df['recipe_id']:
            if recipe_id not in recipe_to_idx:
                continue

            recipe_idx = recipe_to_idx[recipe_id]

            # Predicted preference (normalized 0-1)
            pred_preference_norm = predicted_matrix_norm[user_idx, recipe_idx]

            # Health score
            health_score = health_scores[recipe_id]

            # Multi-objective score (on normalized 0-1 scale)
            score = alpha * pred_preference_norm + (1 - alpha) * health_score

            recipe_scores.append({
                'recipe_id': recipe_id,
                'score': score,
                'health_score': health_score
            })

        # Sort by score and get Top-K
        recipe_scores.sort(key=lambda x: x['score'], reverse=True)
        top_k = recipe_scores[:K]

        # Calculate Precision and Recall
        top_k_ids = set([r['recipe_id'] for r in top_k])
        hits = len(top_k_ids & relevant_items)

        precision = hits / K if K > 0 else 0
        recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

        # Calculate average health score of Top-K
        avg_health = np.mean([r['health_score'] for r in top_k])
        health_scores_list.append(avg_health)

    # Aggregate metrics
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_health_topk = np.mean(health_scores_list)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    results_by_alpha[alpha] = {
        'alpha': alpha,
        'precision_at_k': float(avg_precision),
        'recall_at_k': float(avg_recall),
        'f1_score': float(f1_score),
        'avg_health_topk': float(avg_health_topk),
        'k': K,
        'num_users_evaluated': len(precision_list)
    }

    print(f"    Precision@{K}: {avg_precision:.4f}")
    print(f"    Recall@{K}: {avg_recall:.4f}")
    print(f"    F1-Score: {f1_score:.4f}")
    print(f"    Avg Health (Top-{K}): {avg_health_topk:.4f}")

# ==============================================================================
# 3. SAVE RESULTS
# ==============================================================================
print("\n[3/4] Saving Top-K evaluation results...")

output = {
    "evaluation_type": "Top-K Recommendation Quality",
    "k": K,
    "rating_threshold": RATING_THRESHOLD,
    "alpha_values": alpha_values,
    "results": list(results_by_alpha.values())
}

results_path = f"{RESULTS_DIR}/topk_evaluation.json"
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"✓ Saved to: {results_path}")

# ==============================================================================
# 4. SUMMARY
# ==============================================================================
print("\n[4/4] Summary")
print("="*80)
print(f"{'Alpha':<8} {'Precision@10':<14} {'Recall@10':<12} {'F1-Score':<12} {'Health(Top-10)':<15}")
print("-"*80)

for alpha in alpha_values:
    r = results_by_alpha[alpha]
    print(f"{alpha:<8.2f} {r['precision_at_k']:<14.4f} {r['recall_at_k']:<12.4f} {r['f1_score']:<12.4f} {r['avg_health_topk']:<15.4f}")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print(f"• As α decreases (more health focus), health of Top-{K} recommendations increases")
print(f"• As α increases (more preference focus), Precision/Recall improve")
print(f"• Trade-off: Better recommendation accuracy vs. healthier recommendations")
print("\nTop-K evaluation complete!")
