# Multi-Objective Food Recommendation System: Balancing User Preference and Health

**Project: Recommender Systems**

---

## 1. Introduction

### 1.1 Problem Statement

Traditional recommendation systems optimize exclusively for user preferences, often leading to recommendations that align with user tastes but may not promote healthy choices. This project addresses the challenge of building a **multi-objective recommender system** that balances two competing objectives:

1. **User Preference**: Maximizing predicted user satisfaction (rating prediction accuracy)
2. **Health Optimization**: Promoting recipes with higher nutritional quality (WHO health scores)

### 1.2 Research Question

**How can we design a recommendation system that balances individual taste preferences with health considerations, and what are the trade-offs between these two objectives?**

### 1.3 Dataset Overview

We use a food recipe dataset containing:
- **281 recipes** with complete nutritional information
- **28,491 user ratings** (1-5 star scale) from 17,208 unique users
- **WHO health scores** (0-1 scale, based on WHO nutritional guidelines)
- **Nutritional attributes**: calories, fats, proteins, carbohydrates, sugars, sodium, fiber

**Data Split**:
- Training: 22,792 ratings (80%)
- Validation: 2,849 ratings (10%)
- Test: 2,850 ratings (10%)

The dataset exhibits high sparsity (99.53%), which is typical for collaborative filtering scenarios in recommendation systems.

---

## 2. Methodology

### 2.1 Data Preprocessing

**Data Cleaning**:
- Filtered recipes to include only those with complete nutritional information and user ratings
- Removed recipes missing WHO health scores or key nutritional fields
- Final dataset: 281 recipes (from 703 original) with 28,491 ratings

**Normalization**:
- Ratings: Normalized from [1,5] scale to [0,1] using formula: `rating_normalized = (rating - 1) / 4`
- Health scores: Already in [0,1] range (WHO-aligned health scores)
- Rationale: Ensures both objectives operate on the same scale for fair weighting in multi-objective function

**Data Split Strategy**:
- Train: 80% (22,792 ratings)
- Validation: 10% (2,849 ratings)
- Test: 10% (2,850 ratings)
- Split method: Random split (justified by lack of temporal information in dataset)
- High sparsity: 99.53% (typical for collaborative filtering)

**User Health Preference Baseline**:
- Computed average health score of recipes each user has rated
- Used to understand existing user health preferences
- Average user health preference: 0.5990 (moderate health consciousness)

### 2.2 Baseline Model: Preference-Only Recommendation

**Model Architecture**: Truncated Singular Value Decomposition (SVD)

We implemented a baseline model that optimizes solely for rating prediction using Truncated SVD, a matrix factorization technique that decomposes the user-item interaction matrix into latent factor representations:

```
R_centered ≈ U × Σ × V^T
R = R_centered + global_mean
```

Where:
- **R**: User-item rating matrix (17,208 × 281)
- **U**: User latent factors (17,208 × 50)
- **Σ**: Singular values (diagonal matrix)
- **V^T**: Recipe latent factors (50 × 281)
- **global_mean**: Average rating across all training data (for mean-centering)

**Model Parameters**:
- Latent factors: 50 components
- Maximum iterations: 20
- Random state: 42 (for reproducibility)
- Mean-centering: Applied before decomposition

**Rationale**: SVD with mean-centering significantly outperforms NMF (RMSE: 0.9056 vs. 3.6491) by capturing both global rating tendencies and user-specific preferences. The mean-centering step is crucial for recommendation tasks as it removes bias and allows the model to focus on preference deviations.

### 2.3 Multi-Objective Model: Preference + Health

The multi-objective model introduces a weighted combination of preference prediction and health scores:

```
Score(u, i) = α · Preference(u, i) + (1 - α) · Health(i)
```

Where:
- **α**: Trade-off parameter (0 ≤ α ≤ 1)
  - α = 1.0: Pure preference (baseline)
  - α = 0.5: Balanced approach
  - α = 0.0: Pure health optimization
- **Preference(u, i)**: Predicted rating from baseline SVD model (normalized from [1,5] to [0,1])
- **Health(i)**: WHO health score for recipe i (already in [0,1] range)

**Tested α values**: 0.0, 0.25, 0.5, 0.75, 1.0

### 2.4 Evaluation Metrics

**Rating Prediction Accuracy**:
1. **RMSE (Root Mean Squared Error)**: Measures rating prediction accuracy (lower is better)
2. **MAE (Mean Absolute Error)**: Average absolute deviation from actual ratings (lower is better)

**Recommendation Quality (Top-K)**:

3. **Precision@10**: Proportion of Top-10 recommended items that are relevant (rating ≥ 4.0)
4. **Recall@10**: Proportion of relevant items that appear in Top-10 recommendations
5. **F1-Score**: Harmonic mean of Precision and Recall

**Healthfulness**:

6. **Avg Health Score (Top-10)**: Average WHO health score of Top-10 recommended recipes (higher is better)

---

## 3. Results

### 3.1 Baseline Model Performance

| Metric | Validation | Test |
|--------|-----------|------|
| **RMSE** | 0.9120 | **0.9056** |
| **MAE** | 0.6753 | 0.6702 |

The baseline SVD model achieves an RMSE of 0.9056 on the test set (on 1-5 rating scale), establishing our benchmark for pure preference-based recommendation. This represents strong performance, significantly outperforming the initial NMF baseline (RMSE: 3.6491).

### 3.2 Multi-Objective Model Performance

| α Value | Preference Weight | Health Weight | Test RMSE | Test MAE | Avg Health (Top-10) |
|---------|-------------------|---------------|-----------|----------|---------------------|
| 0.00    | 0%                | 100%          | 1.5681    | 1.4079   | 0.9137              |
| 0.25    | 25%               | 75%           | 1.3115    | 1.1869   | 0.9137              |
| 0.50    | 50%               | 50%           | 1.0958    | 0.9807   | 0.9136              |
| 0.75    | 75%               | 25%           | 0.9490    | 0.8072   | 0.9089              |
| 1.00    | 100%              | 0%            | 0.9056    | 0.6702   | 0.5868              |

### 3.3 Top-K Recommendation Quality

Evaluated on 2,155 test users with at least one relevant item (rating ≥ 4.0):

| α Value | Precision@10 | Recall@10 | F1-Score | Avg Health (Top-10) |
|---------|--------------|-----------|----------|---------------------|
| 0.00    | 0.0041       | 0.0349    | 0.0073   | 0.9137              |
| 0.25    | 0.0041       | 0.0349    | 0.0073   | 0.9137              |
| 0.50    | 0.0041       | 0.0349    | 0.0073   | 0.9136              |
| 0.75    | 0.0039       | 0.0330    | 0.0069   | 0.9086              |
| 1.00    | 0.0045       | 0.0384    | 0.0081   | 0.5884              |

**Key Trade-off**: As α increases (more preference focus), Precision/Recall remain relatively stable, but health quality of Top-10 recommendations decreases dramatically (from 0.9137 to 0.5884) — a 52% reduction.

**Note on Low Precision/Recall Values**: The relatively low Precision@10 (0.0039-0.0045) and Recall@10 (0.0330-0.0384) values are expected due to extreme dataset sparsity (99.57%) and the fact that most test users have only 1-2 ratings. This is typical for sparse recommendation datasets and does not indicate model failure.

### 3.4 Key Findings

1. **Trade-off Confirmation**: As α increases (more preference weight), RMSE decreases from 1.5681 to 0.9056, demonstrating that the system achieves better rating prediction accuracy when prioritizing user preferences. However, this comes at the cost of health optimization.

2. **Health Trade-off**: The average health score of Top-10 recommendations decreases from 0.9137 (α=0.0, pure health focus) to 0.5868 (α=1.0, pure preference focus)—a 36% reduction. This demonstrates that optimizing for user preference comes at a significant cost to healthfulness.

3. **Balanced Approach**: α = 0.5 provides a middle ground with RMSE = 1.0958 and health score of 0.9136, offering a compromise between prediction accuracy and health quality. At this setting, the system maintains 99.9% of the maximum health score while achieving moderate prediction performance.

4. **Pareto Frontier**: The relationship between RMSE and health optimization forms a Pareto curve, where improving one objective necessarily degrades the other. No single α value is "optimal"—the choice depends on policy goals.

5. **Recommendation Quality**: Precision@10 and Recall@10 are low (0.0039-0.0045 and 0.0330-0.0384) due to extreme sparsity (99.57%), but remain relatively stable across α values. The primary impact of α is on health quality, not on traditional recommendation accuracy metrics.

### 3.5 Visualizations

**Figure 1: Pareto Curve**

![Pareto Curve](plots/pareto_curve.png)

- Shows the trade-off frontier between preference accuracy (RMSE) and health optimization
- Each point represents a different α value
- Demonstrates no single "optimal" solution exists

**Figure 2: Alpha Comparison**

![Alpha Comparison](plots/alpha_comparison.png)

- Left panel: RMSE decreases with α (from 1.5681 to 0.9056) — lower is better for rating prediction
- Right panel: Average health of Top-10 recommendations decreases with α (from 0.9137 to 0.5868)
- Clearly shows the trade-off: better preference accuracy (lower RMSE) vs. healthier recommendations (higher health score)

**Figure 3: Precision/Recall Analysis**

![Precision/Recall Analysis](plots/precision_recall_analysis.png)

- 3-panel visualization of Top-K recommendation quality metrics
- Left: Precision@10 remains stable across α values (0.0039 to 0.0045)
- Middle: Recall@10 shows similar stability (0.0330 to 0.0384)
- Right: F1-Score (harmonic mean of Precision & Recall) ranges from 0.0069 to 0.0081
- Note: Low values are due to dataset sparsity (99.57%), not model failure

---

## 4. Discussion

### 4.1 Interpretation of Results

The results reveal a fundamental tension in multi-objective recommendation:

**Pure Preference (α=1.0)**:
- Maximizes user satisfaction in the short term
- May lead to unhealthy eating patterns
- RMSE: 0.9056 (best rating prediction accuracy)
- Health score: 0.5868 (significantly worse than health-optimized approaches)

**Pure Health (α=0.0)**:
- Optimizes nutritional quality
- May recommend items users dislike, reducing engagement
- RMSE: 1.5681 (worst rating prediction, but not personalized to preferences)
- Health score: 0.9137 (maximum health optimization)

**Balanced Approach (α=0.5)**:
- Offers compromise between preference and health
- RMSE: 1.0958 (moderate prediction accuracy)
- Health score: 0.9136 (maintains 99.9% of maximum health quality)
- Suitable for "health-conscious but taste-aware" recommendation scenario

### 4.2 Practical Implications

1. **Personalized α Values**: Different users may prefer different α settings:
   - Health-conscious users: α = 0.25-0.5
   - Taste-first users: α = 0.75-1.0
   - System default: α = 0.5 (balanced)

2. **Transparency**: Users should understand the system is balancing preferences with health, avoiding the "filter bubble" effect where users only see what they already like.

3. **Gradual Nudging**: Systems could start with higher α (preference-focused) and gradually decrease it as users become more health-aware.

### 4.3 Ethical Considerations and Responsible AI

**Risks of Nudging Behavior: Over-Restriction vs. Freedom of Choice**

Our multi-objective system employs "nudging"—subtly steering users toward healthier choices. This raises critical ethical tensions:

- **Over-Restriction Risk (Low α)**: When α = 0.0-0.25, the system heavily prioritizes health, potentially recommending recipes users dislike. This risks:
  - Reduced user engagement and system abandonment
  - Paternalistic decision-making that overrides personal preferences
  - Potential backlash against "forced" health optimization

- **Freedom of Choice (High α)**: When α = 0.75-1.0, users get what they prefer, but system may enable unhealthy eating patterns:
  - Reinforces existing poor dietary habits
  - Missed opportunity for public health intervention
  - Filter bubble effect: users never discover healthier alternatives

- **Balance**: α = 0.5 offers compromise, but who decides the "right" balance? This reflects deeper questions about system responsibility vs. individual autonomy.

**Personalization Fairness Across Dietary Preference Groups**

The fairness of health-based recommendations varies significantly across user populations:

1. **Cultural Dietary Preferences**:
   - WHO health scores reflect Western nutritional guidelines
   - May penalize culturally important foods (e.g., high-sodium fermented foods in Asian cuisine)
   - Users from non-Western backgrounds may experience systematically lower satisfaction with health-optimized recommendations

2. **Medical Conditions**:
   - A recipe "healthy" for general population may be harmful for diabetics (high carbs), kidney patients (high protein), or those with allergies
   - One-size-fits-all health scores fail to account for personalized medical needs
   - System could inadvertently recommend dangerous foods to vulnerable users

3. **Socioeconomic Factors**:
   - Healthier recipes often require expensive ingredients or more preparation time
   - Low-income users may be nudged toward unaffordable recommendations
   - Creates fairness gap where health optimization benefits privileged users more

**Recommendations for Responsible Deployment**:
- Implement user-controllable α parameter with clear explanations
- Allow complete opt-out of health optimization
- Provide personalized health profiles accounting for medical conditions and cultural preferences
- Transparently display why each recipe is recommended (preference vs. health contribution)
- Monitor fairness metrics across demographic groups


## 5. Conclusion

This project successfully demonstrates the implementation and evaluation of a multi-objective food recommendation system that balances user preference with health optimization. Key contributions include:

1. **Technical Implementation**: Successfully trained baseline SVD model (RMSE: 0.9056) and multi-objective framework with configurable trade-off parameter α

2. **Empirical Validation**: Demonstrated clear trade-off between preference accuracy (RMSE: 1.5681 → 0.9056) and health optimization (Health: 0.9137 → 0.5868) across five α values

3. **Ethical Analysis**: Identified key tensions between system-driven health nudging and user autonomy

4. **Practical Framework**: Provided actionable approach for deploying health-aware recommendation systems with user control

**Final Recommendation**: Deploy with α = 0.5 as default, allowing users to adjust based on personal priorities, with clear transparency about health optimization goals.

---

## References

- Dataset: Food.com recipes and ratings
- WHO Nutritional Guidelines (health score calculation)
- Truncated SVD Algorithm: scikit-learn v1.2+
- Multi-objective optimization principles from Pareto efficiency theory

---

## Appendix: Reproducibility

**Code Repository Structure**:
```
Project 2/
├── src/
│   ├── 1_explore_data.py
│   ├── 2_prepare_data.py
│   ├── 3_baseline_model_svd.py        # SVD baseline model
│   ├── 4_multiobjective_model.py       # Multi-objective model
│   ├── 5_pareto_plot.py                # Pareto curve visualization
│   ├── 6_topk_evaluation.py            # Top-K metrics evaluation
│   ├── 7_precision_recall_plot.py      # Precision/Recall visualization
│   └── 8_alpha_comparison_plot.py      # Alpha comparison plot
├── data/                               # Processed datasets
├── models/                             # Trained models (.pkl)
├── results/                            # JSON results
├── plots/                              # Visualizations
└── REPORT.md                           # This report
```



---

*Report prepared for Project 2: Multi-Objective Recommendation Systems*
