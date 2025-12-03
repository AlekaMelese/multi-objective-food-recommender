"""
Step 3: Baseline Model - Matrix Factorization (SVD)
Optimize for rating prediction ONLY (no health consideration)
This is the traditional recommendation approach
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise import accuracy
import json
import pickle
import time

# Paths
DATA_DIR = "../data"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

print("="*80)
print("BASELINE MODEL: MATRIX FACTORIZATION (SVD)")
print("="*80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n[1/5] Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train_ratings.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val_ratings.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_ratings.csv")

print(f"✓ Train: {len(train_df):,} ratings")
print(f"✓ Val:   {len(val_df):,} ratings")
print(f"✓ Test:  {len(test_df):,} ratings")

# ==============================================================================
# 2. PREPARE DATA FOR SURPRISE LIBRARY
# ==============================================================================
print("\n[2/5] Preparing data for Surprise library...")

# Define rating scale (1-5)
reader = Reader(rating_scale=(1, 5))

# Load datasets
train_data = Dataset.load_from_df(
    train_df[['member_id', 'recipe_id', 'rating']], reader
).build_full_trainset()

val_data = Dataset.load_from_df(
    val_df[['member_id', 'recipe_id', 'rating']], reader
).build_full_trainset().build_testset()

test_data = Dataset.load_from_df(
    test_df[['member_id', 'recipe_id', 'rating']], reader
).build_full_trainset().build_testset()

print("✓ Prepared train/val/test sets for Surprise")

# ==============================================================================
# 3. HYPERPARAMETER TUNING (Optional - use if time permits)
# ==============================================================================
print("\n[3/5] Training baseline SVD model...")

# Option A: Quick training with default parameters
print("\nUsing default SVD parameters for quick training...")
model = SVD(
    n_factors=20,      # Number of latent factors
    n_epochs=20,       # Number of training epochs
    lr_all=0.005,      # Learning rate
    reg_all=0.02,      # Regularization
    random_state=42
)

# Option B: Hyperparameter tuning (commented out - use if you have time)
"""
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_factors': [10, 20, 50],
    'n_epochs': [10, 20],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(Dataset.load_from_df(train_df[['member_id', 'recipe_id', 'rating']], reader))
model = gs.best_estimator['rmse']
print(f"✓ Best RMSE: {gs.best_score['rmse']:.4f}")
print(f"✓ Best params: {gs.best_params['rmse']}")
"""

# Train model
start_time = time.time()
model.fit(train_data)
train_time = time.time() - start_time

print(f"✓ Model trained in {train_time:.2f} seconds")
print(f"  Factors: {model.n_factors}")
print(f"  Epochs: {model.n_epochs}")

# ==============================================================================
# 4. EVALUATE MODEL
# ==============================================================================
print("\n[4/5] Evaluating model...")

# Predictions on validation set
val_predictions = model.test(val_data)
val_rmse = accuracy.rmse(val_predictions, verbose=False)
val_mae = accuracy.mae(val_predictions, verbose=False)

print(f"\nValidation Performance:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae:.4f}")

# Predictions on test set
test_predictions = model.test(test_data)
test_rmse = accuracy.rmse(test_predictions, verbose=False)
test_mae = accuracy.mae(test_predictions, verbose=False)

print(f"\nTest Performance:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")

# ==============================================================================
# 5. SAVE MODEL AND RESULTS
# ==============================================================================
print("\n[5/5] Saving model and results...")

# Save model
model_path = f"{MODELS_DIR}/baseline_svd.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Saved model to: {model_path}")

# Save results
results = {
    "model_type": "SVD (Matrix Factorization)",
    "objective": "Rating prediction only (preference)",
    "parameters": {
        "n_factors": model.n_factors,
        "n_epochs": model.n_epochs,
        "lr_all": model.lr_all,
        "reg_all": model.reg_all
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
        "num_users": int(train_df['member_id'].nunique()),
        "num_recipes": int(train_df['recipe_id'].nunique())
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

# Get a random user
sample_user_id = train_df['member_id'].sample(1).values[0]
recipes_df = pd.read_csv(f"{DATA_DIR}/recipes.csv")

# Get top-10 recipe predictions for this user
recipe_ids = recipes_df['recipe_id'].tolist()
predictions = []

for recipe_id in recipe_ids:
    pred = model.predict(sample_user_id, recipe_id)
    predictions.append({
        'recipe_id': recipe_id,
        'predicted_rating': pred.est
    })

# Sort by predicted rating
predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
top_10 = predictions[:10]

# Add recipe details
for pred in top_10:
    recipe = recipes_df[recipes_df['recipe_id'] == pred['recipe_id']].iloc[0]
    pred['title'] = recipe['title']
    pred['health_score'] = float(recipe['health_score'])
    pred['calories'] = float(recipe['calories'])

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
print(f"\nModel saved to: {MODELS_DIR}/baseline_svd.pkl")
print(f"Results saved to: {RESULTS_DIR}/baseline_results.json")
print(f"\nTest RMSE: {test_rmse:.4f} (lower is better)")
print("\nNext step: Build multi-objective model (preference + health)")
