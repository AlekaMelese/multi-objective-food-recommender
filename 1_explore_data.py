"""
Step 1: Data Exploration
Explore Recipes.csv and Ratings.csv to understand the data structure
"""

import pandas as pd
import numpy as np
import os

# Paths
RECIPES_PATH = "Data/Recipes.csv"
RATINGS_PATH = "Data/Ratings.csv"

print("="*80)
print("HUMMUS DATASET EXPLORATION")
print("="*80)

# ==============================================================================
# 1. EXPLORE RECIPES DATA
# ==============================================================================
print("\n" + "="*80)
print("1. RECIPES DATA")
print("="*80)

print(f"\nLoading recipes from: {RECIPES_PATH}")
recipes_df = pd.read_csv(RECIPES_PATH)

print(f"\n✓ Loaded {len(recipes_df):,} recipes")
print(f"✓ Number of columns: {len(recipes_df.columns)}")

print("\n--- Column Names ---")
print(recipes_df.columns.tolist())

print("\n--- First 3 rows ---")
print(recipes_df.head(3))

print("\n--- Data Types ---")
print(recipes_df.dtypes)

print("\n--- Missing Values ---")
missing = recipes_df.isnull().sum()
missing_pct = (missing / len(recipes_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Percentage': missing_pct.values
})
print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))

# Check for nutritional fields
print("\n--- Nutritional Fields Check ---")
required_fields = ['calories', 'totalFat', 'saturatedFat', 'sugars', 'sodium', 'protein', 'who_score']
available_fields = []
missing_fields = []

for field in required_fields:
    if field in recipes_df.columns:
        available_fields.append(field)
        non_null = recipes_df[field].notna().sum()
        print(f"✓ {field}: {non_null:,} non-null ({non_null/len(recipes_df)*100:.1f}%)")
    else:
        missing_fields.append(field)
        print(f"✗ {field}: NOT FOUND")

# Check for who_score alternatives
if 'who_score' not in recipes_df.columns:
    print("\n--- Searching for WHO score alternatives ---")
    who_alternatives = [col for col in recipes_df.columns if 'who' in col.lower() or 'health' in col.lower() or 'score' in col.lower()]
    if who_alternatives:
        print(f"Found potential alternatives: {who_alternatives}")
    else:
        print("No who_score or alternative health score field found")

# Statistics for nutritional fields
if available_fields:
    print("\n--- Nutritional Fields Statistics ---")
    print(recipes_df[available_fields].describe())

# Count recipes with complete nutritional data
if available_fields:
    complete_nutrition = recipes_df[available_fields].notna().all(axis=1).sum()
    print(f"\n✓ Recipes with ALL nutritional fields: {complete_nutrition:,} ({complete_nutrition/len(recipes_df)*100:.1f}%)")

# ==============================================================================
# 2. EXPLORE RATINGS DATA
# ==============================================================================
print("\n" + "="*80)
print("2. RATINGS/INTERACTIONS DATA")
print("="*80)

print(f"\nLoading ratings from: {RATINGS_PATH}")
ratings_df = pd.read_csv(RATINGS_PATH)

print(f"\n✓ Loaded {len(ratings_df):,} ratings")
print(f"✓ Number of columns: {len(ratings_df.columns)}")

print("\n--- Column Names ---")
print(ratings_df.columns.tolist())

print("\n--- First 5 rows ---")
print(ratings_df.head())

print("\n--- Data Types ---")
print(ratings_df.dtypes)

print("\n--- Ratings Statistics ---")
# Identify the rating column (might be 'rating', 'score', etc.)
rating_col = None
for col in ['rating', 'score', 'user_rating']:
    if col in ratings_df.columns:
        rating_col = col
        break

if rating_col:
    print(f"\nRating column: '{rating_col}'")
    print(ratings_df[rating_col].describe())
    print(f"\nRating value counts:")
    print(ratings_df[rating_col].value_counts().sort_index())
else:
    print("Could not identify rating column")

# User and recipe statistics
user_col = [col for col in ratings_df.columns if 'user' in col.lower() or 'member' in col.lower()]
recipe_col = [col for col in ratings_df.columns if 'recipe' in col.lower()]

if user_col:
    user_col = user_col[0]
    print(f"\n✓ Unique users: {ratings_df[user_col].nunique():,}")
    print(f"  Avg ratings per user: {len(ratings_df)/ratings_df[user_col].nunique():.1f}")

if recipe_col:
    recipe_col = recipe_col[0]
    print(f"✓ Unique recipes: {ratings_df[recipe_col].nunique():,}")
    print(f"  Avg ratings per recipe: {len(ratings_df)/ratings_df[recipe_col].nunique():.1f}")

# ==============================================================================
# 3. DATA OVERLAP ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("3. DATA OVERLAP ANALYSIS")
print("="*80)

if recipe_col and 'recipe_id' in recipes_df.columns:
    recipes_with_ratings = ratings_df[recipe_col].unique()
    all_recipes = recipes_df['recipe_id'].unique()

    overlap = len(set(recipes_with_ratings) & set(all_recipes))
    print(f"\n✓ Recipes in Recipes.csv: {len(all_recipes):,}")
    print(f"✓ Recipes in Ratings.csv: {len(recipes_with_ratings):,}")
    print(f"✓ Overlap: {overlap:,} ({overlap/len(all_recipes)*100:.1f}% of recipes)")
    print(f"✓ Recipes with ratings: {overlap:,}")

# ==============================================================================
# 4. SAMPLING FEASIBILITY
# ==============================================================================
print("\n" + "="*80)
print("4. SAMPLING FEASIBILITY FOR 1000 RECIPES")
print("="*80)

# Check how many recipes meet our criteria
if available_fields and recipe_col:
    # Recipes with complete nutrition
    recipes_complete = recipes_df[recipes_df[available_fields].notna().all(axis=1)]

    # Recipes with ratings
    recipes_with_ratings_ids = set(ratings_df[recipe_col].unique())
    recipes_complete_with_ratings = recipes_complete[
        recipes_complete['recipe_id'].isin(recipes_with_ratings_ids)
    ]

    print(f"\n✓ Recipes with complete nutritional data: {len(recipes_complete):,}")
    print(f"✓ Recipes with ratings: {len(recipes_with_ratings_ids):,}")
    print(f"✓ Recipes with BOTH complete nutrition AND ratings: {len(recipes_complete_with_ratings):,}")

    if len(recipes_complete_with_ratings) >= 1000:
        print(f"\n✅ FEASIBLE: Can sample 1000 recipes meeting all criteria")
    else:
        print(f"\n⚠️  WARNING: Only {len(recipes_complete_with_ratings)} recipes meet all criteria")
        print(f"   May need to relax some requirements")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
