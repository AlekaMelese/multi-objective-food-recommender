"""
Step 2: Data Preparation
Prepare dataset for multi-objective recommendation system
- Use 281 recipes with both nutritional data and user ratings
- Normalize ratings and health scores
- Split into train/validation/test sets
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Paths
RECIPES_PATH = "Data/Recipes.csv"
RATINGS_PATH = "Data/Ratings.csv"
OUTPUT_DIR = "FoodNew/Project 2/data"

print("="*80)
print("DATA PREPARATION FOR MULTI-OBJECTIVE RECOMMENDATION")
print("="*80)

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n[1/6] Loading data...")
recipes_df = pd.read_csv(RECIPES_PATH)
ratings_df = pd.read_csv(RATINGS_PATH)

print(f"✓ Loaded {len(recipes_df):,} recipes")
print(f"✓ Loaded {len(ratings_df):,} ratings")

# ==============================================================================
# 2. FILTER RECIPES WITH RATINGS
# ==============================================================================
print("\n[2/6] Filtering recipes with ratings...")

# Get recipe IDs that have ratings
recipes_with_ratings = set(ratings_df['recipe_id'].unique())
print(f"✓ Found {len(recipes_with_ratings)} unique recipes with ratings")

# Filter recipes dataframe
recipes_filtered = recipes_df[recipes_df['recipe_id'].isin(recipes_with_ratings)].copy()
print(f"✓ Filtered to {len(recipes_filtered)} recipes")

# ==============================================================================
# 3. PREPARE RECIPE FEATURES
# ==============================================================================
print("\n[3/6] Preparing recipe features...")

# Select relevant columns (map bracket notation to clean names)
recipe_features = recipes_filtered[[
    'recipe_id',
    'title',
    'calories [cal]',
    'totalFat [g]',
    'saturatedFat [g]',
    'sugars [g]',
    'sodium [mg]',
    'protein [g]',
    'who_score',
    'who_normalized',
    'health_category',
    'tags',
    'ingredients',
    'duration'
]].copy()

# Rename columns to remove brackets
recipe_features.columns = [
    'recipe_id', 'title', 'calories', 'totalFat', 'saturatedFat',
    'sugars', 'sodium', 'protein', 'who_score', 'who_normalized',
    'health_category', 'tags', 'ingredients', 'duration'
]

# Normalize who_score to 0-1 if not already (use who_normalized)
recipe_features['health_score'] = recipe_features['who_normalized']

print(f"✓ Prepared {len(recipe_features)} recipe features")
print(f"✓ Health score range: [{recipe_features['health_score'].min():.3f}, {recipe_features['health_score'].max():.3f}]")

# ==============================================================================
# 4. PREPARE RATINGS DATA
# ==============================================================================
print("\n[4/6] Preparing ratings data...")

# Filter ratings to only include recipes we have
ratings_filtered = ratings_df[ratings_df['recipe_id'].isin(recipes_with_ratings)].copy()

print(f"✓ Filtered to {len(ratings_filtered):,} ratings")
print(f"✓ Unique users: {ratings_filtered['member_id'].nunique():,}")
print(f"✓ Unique recipes: {ratings_filtered['recipe_id'].nunique():,}")

# Normalize ratings to 0-1
ratings_filtered['rating_normalized'] = (ratings_filtered['rating'] - 1) / 4  # 1-5 -> 0-1

print(f"✓ Normalized ratings to [0, 1]")
print(f"  Original range: [{ratings_filtered['rating'].min():.1f}, {ratings_filtered['rating'].max():.1f}]")
print(f"  Normalized range: [{ratings_filtered['rating_normalized'].min():.3f}, {ratings_filtered['rating_normalized'].max():.3f}]")

# ==============================================================================
# 5. SPLIT DATA
# ==============================================================================
print("\n[5/6] Splitting data into train/val/test...")

# Split ratings: 80% train, 10% val, 10% test
# Use random split (could also use time-based if we had timestamps)
train_ratings, temp_ratings = train_test_split(
    ratings_filtered, test_size=0.2, random_state=42
)
val_ratings, test_ratings = train_test_split(
    temp_ratings, test_size=0.5, random_state=42
)

print(f"✓ Train: {len(train_ratings):,} ratings ({len(train_ratings)/len(ratings_filtered)*100:.1f}%)")
print(f"✓ Val:   {len(val_ratings):,} ratings ({len(val_ratings)/len(ratings_filtered)*100:.1f}%)")
print(f"✓ Test:  {len(test_ratings):,} ratings ({len(test_ratings)/len(ratings_filtered)*100:.1f}%)")

# ==============================================================================
# 6. SAVE PROCESSED DATA
# ==============================================================================
print("\n[6/6] Saving processed data...")

# Save recipes (JSON for easy loading)
recipes_json = recipe_features.to_dict(orient='records')
with open(f"{OUTPUT_DIR}/recipes.json", 'w') as f:
    json.dump(recipes_json, f, indent=2)
print(f"✓ Saved recipes to: {OUTPUT_DIR}/recipes.json")

# Save recipes as CSV too
recipe_features.to_csv(f"{OUTPUT_DIR}/recipes.csv", index=False)
print(f"✓ Saved recipes to: {OUTPUT_DIR}/recipes.csv")

# Save ratings splits (CSV for compatibility with recommender libraries)
train_ratings.to_csv(f"{OUTPUT_DIR}/train_ratings.csv", index=False)
val_ratings.to_csv(f"{OUTPUT_DIR}/val_ratings.csv", index=False)
test_ratings.to_csv(f"{OUTPUT_DIR}/test_ratings.csv", index=False)
print(f"✓ Saved train ratings to: {OUTPUT_DIR}/train_ratings.csv")
print(f"✓ Saved val ratings to: {OUTPUT_DIR}/val_ratings.csv")
print(f"✓ Saved test ratings to: {OUTPUT_DIR}/test_ratings.csv")

# Save all ratings (for reference)
ratings_filtered.to_csv(f"{OUTPUT_DIR}/all_ratings.csv", index=False)
print(f"✓ Saved all ratings to: {OUTPUT_DIR}/all_ratings.csv")

# ==============================================================================
# 7. SUMMARY STATISTICS
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n--- Recipe Statistics ---")
print(f"Total recipes: {len(recipe_features)}")
print(f"Health score (WHO):")
print(f"  Mean: {recipe_features['health_score'].mean():.3f}")
print(f"  Std:  {recipe_features['health_score'].std():.3f}")
print(f"  Min:  {recipe_features['health_score'].min():.3f}")
print(f"  Max:  {recipe_features['health_score'].max():.3f}")

print(f"\nNutritional statistics:")
for col in ['calories', 'totalFat', 'saturatedFat', 'sugars', 'sodium', 'protein']:
    print(f"  {col}: mean={recipe_features[col].mean():.1f}, std={recipe_features[col].std():.1f}")

print("\n--- Rating Statistics ---")
print(f"Total ratings: {len(ratings_filtered):,}")
print(f"Unique users: {ratings_filtered['member_id'].nunique():,}")
print(f"Unique recipes: {ratings_filtered['recipe_id'].nunique():,}")
print(f"Ratings per user: {len(ratings_filtered)/ratings_filtered['member_id'].nunique():.1f}")
print(f"Ratings per recipe: {len(ratings_filtered)/ratings_filtered['recipe_id'].nunique():.1f}")
print(f"\nRating distribution:")
print(ratings_filtered['rating'].value_counts().sort_index())

print(f"\nNormalized ratings:")
print(f"  Mean: {ratings_filtered['rating_normalized'].mean():.3f}")
print(f"  Std:  {ratings_filtered['rating_normalized'].std():.3f}")

print("\n--- Data Split ---")
print(f"Train: {len(train_ratings):,} ratings")
print(f"Val:   {len(val_ratings):,} ratings")
print(f"Test:  {len(test_ratings):,} ratings")

# Save summary statistics
summary = {
    "num_recipes": len(recipe_features),
    "num_ratings": len(ratings_filtered),
    "num_users": int(ratings_filtered['member_id'].nunique()),
    "health_score_stats": {
        "mean": float(recipe_features['health_score'].mean()),
        "std": float(recipe_features['health_score'].std()),
        "min": float(recipe_features['health_score'].min()),
        "max": float(recipe_features['health_score'].max())
    },
    "rating_stats": {
        "mean": float(ratings_filtered['rating'].mean()),
        "std": float(ratings_filtered['rating'].std()),
        "min": float(ratings_filtered['rating'].min()),
        "max": float(ratings_filtered['rating'].max())
    },
    "data_split": {
        "train": len(train_ratings),
        "val": len(val_ratings),
        "test": len(test_ratings)
    }
}

with open(f"{OUTPUT_DIR}/data_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Saved summary to: {OUTPUT_DIR}/data_summary.json")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files created:")
print("  - recipes.json / recipes.csv (281 recipes)")
print("  - train_ratings.csv (80% of ratings)")
print("  - val_ratings.csv (10% of ratings)")
print("  - test_ratings.csv (10% of ratings)")
print("  - all_ratings.csv (all ratings)")
print("  - data_summary.json (statistics)")
