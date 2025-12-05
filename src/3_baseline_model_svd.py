"""
Step 3: Baseline Model - SVD (Truncated SVD)
Better alternative to NMF for collaborative filtering
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import pickle
import time

# Paths
DATA_DIR = "../data"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

print("="*80)
print("BASELINE MODEL: TRUNCATED SVD")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train_ratings.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val_ratings.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_ratings.csv")

print(f"✓ Train: {len(train_df):,} ratings")
print(f"✓ Val:   {len(val_df):,} ratings")
print(f"✓ Test:  {len(test_df):,} ratings")

# Create user-item matrix
print("\n[2/5] Creating user-item matrix...")

all_users = pd.concat([train_df['member_id'], val_df['member_id'], test_df['member_id']]).unique()
all_recipes = pd.concat([train_df['recipe_id'], val_df['recipe_id'], test_df['recipe_id']]).unique()

user_to_idx = {user: idx for idx, user in enumerate(all_users)}
recipe_to_idx = {recipe: idx for idx, recipe in enumerate(all_recipes)}

# Create training matrix - CENTER the ratings (remove global mean)
train_matrix = np.zeros((len(user_to_idx), len(recipe_to_idx)))
train_mask = np.zeros((len(user_to_idx), len(recipe_to_idx)), dtype=bool)

for _, row in train_df.iterrows():
    user_idx = user_to_idx[row['member_id']]
    recipe_idx = recipe_to_idx[row['recipe_id']]
    train_matrix[user_idx, recipe_idx] = row['rating']
    train_mask[user_idx, recipe_idx] = True

# Calculate global mean from training data only
global_mean = train_matrix[train_mask].mean()
print(f"✓ Global mean rating: {global_mean:.2f}")

# Center the matrix (subtract mean from non-zero entries)
train_matrix_centered = train_matrix.copy()
train_matrix_centered[train_mask] -= global_mean

print(f"✓ Training matrix shape: {train_matrix.shape}")
print(f"✓ Sparsity: {(~train_mask).sum() / train_mask.size * 100:.2f}%")

# Train SVD model
print("\n[3/5] Training SVD model...")

n_components = 50  # More components than NMF
random_state = 42

svd = TruncatedSVD(
    n_components=n_components,
    random_state=random_state,
    n_iter=20
)

start_time = time.time()
U = svd.fit_transform(train_matrix_centered)  # User factors
Vt = svd.components_  # Recipe factors (already transposed)
train_time = time.time() - start_time

print(f"✓ Model trained in {train_time:.2f} seconds")
print(f"  User factors (U): {U.shape}")
print(f"  Recipe factors (Vt): {Vt.shape}")
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

# Make predictions
print("\n[4/5] Making predictions...")

# Reconstruct centered matrix then add global mean back
predicted_matrix_centered = np.dot(U, Vt)
predicted_matrix = predicted_matrix_centered + global_mean
predicted_matrix = np.clip(predicted_matrix, 1, 5)

def get_predictions(df):
    predictions = []
    actuals = []
    for _, row in df.iterrows():
        user_idx = user_to_idx[row['member_id']]
        recipe_idx = recipe_to_idx[row['recipe_id']]
        pred = predicted_matrix[user_idx, recipe_idx]
        predictions.append(pred)
        actuals.append(row['rating'])
    return np.array(predictions), np.array(actuals)

val_preds, val_actuals = get_predictions(val_df)
val_rmse = np.sqrt(mean_squared_error(val_actuals, val_preds))
val_mae = mean_absolute_error(val_actuals, val_preds)

print(f"\nValidation Performance:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae:.4f}")

test_preds, test_actuals = get_predictions(test_df)
test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
test_mae = mean_absolute_error(test_actuals, test_preds)

print(f"\nTest Performance:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")

# Save model
print("\n[5/5] Saving model and results...")

model_data = {
    'U': U,
    'Vt': Vt,
    'global_mean': global_mean,
    'user_to_idx': user_to_idx,
    'recipe_to_idx': recipe_to_idx
}

model_path = f"{MODELS_DIR}/baseline_svd.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"✓ Saved model to: {model_path}")

results = {
    "model_type": "Truncated SVD",
    "objective": "Rating prediction (collaborative filtering)",
    "parameters": {
        "n_components": n_components,
        "random_state": random_state,
        "global_mean": float(global_mean)
    },
    "training_time_seconds": train_time,
    "validation_metrics": {
        "RMSE": float(val_rmse),
        "MAE": float(val_mae)
    },
    "test_metrics": {
        "RMSE": float(test_rmse),
        "MAE": float(test_mae)
    }
}

results_path = f"{RESULTS_DIR}/baseline_svd_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results to: {results_path}")

print("\n" + "="*80)
print("SVD BASELINE MODEL COMPLETE!")
print("="*80)
print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE:  {test_mae:.4f}")
print(f"\nCompare with NMF RMSE: 3.6491")
