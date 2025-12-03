"""
Step 3: Baseline Model - Matrix Factorization (NMF with scikit-learn)
Optimize for rating prediction ONLY (no health consideration)
Alternative implementation using sklearn instead of surprise library
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import pickle
import time
from scipy.sparse import csr_matrix

# Paths
DATA_DIR = "../data"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

print("="*80)
print("BASELINE MODEL: MATRIX FACTORIZATION (NMF with scikit-learn)")
print("="*80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n[1/6] Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train_ratings.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val_ratings.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_ratings.csv")

print(f"✓ Train: {len(train_df):,} ratings")
print(f"✓ Val:   {len(val_df):,} ratings")
print(f"✓ Test:  {len(test_df):,} ratings")

# ==============================================================================
# 2. CREATE USER-ITEM MATRIX
# ==============================================================================
print("\n[2/6] Creating user-item matrix...")

# Create mappings for users and recipes
all_users = pd.concat([train_df['member_id'], val_df['member_id'], test_df['member_id']]).unique()
all_recipes = pd.concat([train_df['recipe_id'], val_df['recipe_id'], test_df['recipe_id']]).unique()

user_to_idx = {user: idx for idx, user in enumerate(all_users)}
recipe_to_idx = {recipe: idx for idx, recipe in enumerate(all_recipes)}
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
idx_to_recipe = {idx: recipe for recipe, idx in recipe_to_idx.items()}

print(f"✓ Users: {len(user_to_idx):,}")
print(f"✓ Recipes: {len(recipe_to_idx):,}")

# Create training matrix
train_matrix = np.zeros((len(user_to_idx), len(recipe_to_idx)))
for _, row in train_df.iterrows():
    user_idx = user_to_idx[row['member_id']]
    recipe_idx = recipe_to_idx[row['recipe_id']]
    train_matrix[user_idx, recipe_idx] = row['rating']

print(f"✓ Training matrix shape: {train_matrix.shape}")
print(f"✓ Sparsity: {(train_matrix == 0).sum() / train_matrix.size * 100:.2f}%")

# ==============================================================================
# 3. TRAIN MODEL
# ==============================================================================
print("\n[3/6] Training NMF model...")

# NMF parameters
n_components = 20  # Number of latent factors
max_iter = 200
random_state = 42

model = NMF(
    n_components=n_components,
    init='random',
    random_state=random_state,
    max_iter=max_iter,
    verbose=0
)

start_time = time.time()
W = model.fit_transform(train_matrix)  # User factors
H = model.components_  # Recipe factors
train_time = time.time() - start_time

print(f"✓ Model trained in {train_time:.2f} seconds")
print(f"  User factors (W): {W.shape}")
print(f"  Recipe factors (H): {H.shape}")
print(f"  Latent factors: {n_components}")

# ==============================================================================
# 4. MAKE PREDICTIONS
# ==============================================================================
print("\n[4/6] Making predictions...")

# Reconstruct the rating matrix
predicted_matrix = np.dot(W, H)

# Function to get predictions for a dataframe
def get_predictions(df):
    predictions = []
    actuals = []
    for _, row in df.iterrows():
        user_idx = user_to_idx[row['member_id']]
        recipe_idx = recipe_to_idx[row['recipe_id']]
        pred = predicted_matrix[user_idx, recipe_idx]
        # Clip predictions to valid rating range [1, 5]
        pred = np.clip(pred, 1, 5)
        predictions.append(pred)
        actuals.append(row['rating'])
    return np.array(predictions), np.array(actuals)

# Validation predictions
val_preds, val_actuals = get_predictions(val_df)
val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
val_mae = mean_absolute_error(val_actuals, val_preds)

print(f"\nValidation Performance:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae:.4f}")

# Test predictions
test_preds, test_actuals = get_predictions(test_df)
test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
test_mae = mean_absolute_error(test_actuals, test_preds)

print(f"\nTest Performance:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")

# ==============================================================================
# 5. SAVE MODEL AND RESULTS
# ==============================================================================
print("\n[5/6] Saving model and results...")

# Save model components
model_data = {
    'W': W,  # User factors
    'H': H,  # Recipe factors
    'user_to_idx': user_to_idx,
    'recipe_to_idx': recipe_to_idx,
    'idx_to_user': idx_to_user,
    'idx_to_recipe': idx_to_recipe
}

model_path = f"{MODELS_DIR}/baseline_nmf.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"✓ Saved model to: {model_path}")

# Save results
results = {
    "model_type": "NMF (Non-negative Matrix Factorization)",
    "objective": "Rating prediction only (preference)",
    "parameters": {
        "n_components": n_components,
        "max_iter": max_iter,
        "random_state": random_state
    },
    "training_time_seconds": train_time,
    "validation_metrics": {
        "RMSE": float(val_rmse),
        "MAE": float(val_mae)
    },
    "test_metrics": {
        "RMSE": float(test_rmse),
        "MAE": float(test_mae)
    },
    "dataset_info": {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_users": len(user_to_idx),
        "num_recipes": len(recipe_to_idx),
        "matrix_sparsity": float((train_matrix == 0).sum() / train_matrix.size)
    }
}

results_path = f"{RESULTS_DIR}/baseline_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results to: {results_path}")

# ==============================================================================
# 6. GENERATE SAMPLE PREDICTIONS
# ==============================================================================
print("\n[6/6] Generating sample predictions...")

# Get a random user who has ratings in training set
sample_user_id = train_df['member_id'].sample(1).values[0]
sample_user_idx = user_to_idx[sample_user_id]

# Load recipe data
recipes_df = pd.read_csv(f"{DATA_DIR}/recipes.csv")

# Get predictions for all recipes
recipe_predictions = []
for recipe_id in recipes_df['recipe_id']:
    if recipe_id in recipe_to_idx:
        recipe_idx = recipe_to_idx[recipe_id]
        pred_rating = predicted_matrix[sample_user_idx, recipe_idx]
        pred_rating = np.clip(pred_rating, 1, 5)

        recipe = recipes_df[recipes_df['recipe_id'] == recipe_id].iloc[0]
        recipe_predictions.append({
            'recipe_id': int(recipe_id),
            'predicted_rating': float(pred_rating),
            'title': recipe['title'],
            'health_score': float(recipe['health_score']),
            'calories': float(recipe['calories'])
        })

# Sort by predicted rating
recipe_predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
top_10 = recipe_predictions[:10]

print(f"\nSample: Top-10 recommendations for user {sample_user_id}")
print(f"{'Rank':<5} {'Pred Rating':<12} {'Health':<8} {'Calories':<10} {'Title':<40}")
print("-" * 80)
for i, pred in enumerate(top_10, 1):
    print(f"{i:<5} {pred['predicted_rating']:<12.3f} {pred['health_score']:<8.3f} {pred['calories']:<10.0f} {pred['title'][:40]}")

# Save sample predictions
sample_path = f"{RESULTS_DIR}/baseline_sample_predictions.json"
with open(sample_path, 'w') as f:
    json.dump({
        'user_id': int(sample_user_id),
        'top_10_predictions': top_10
    }, f, indent=2)
print(f"\n✓ Saved sample predictions to: {sample_path}")

print("\n" + "="*80)
print("BASELINE MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nModel saved to: {MODELS_DIR}/baseline_nmf.pkl")
print(f"Results saved to: {RESULTS_DIR}/baseline_results.json")
print(f"\nTest RMSE: {test_rmse:.4f} (lower is better)")
print(f"Test MAE:  {test_mae:.4f} (lower is better)")
print("\nNext step: Build multi-objective model (preference + health)")
