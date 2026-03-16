"""
================================================================================
HOUSING PRICE REGRESSION ANALYSIS
================================================================================
A comprehensive machine learning study on regression techniques for predicting
housing prices using the Boston Housing dataset. The notebook covers linear
regression, regularization methods (L1/Lasso, L2/Ridge, Elastic Net), polynomial
features, cross-validation, and hyperparameter tuning.

Educational Objectives:
- Understand the fundamentals of linear regression and cost functions
- Learn about regularization techniques and their mathematical formulations
- Explore feature scaling and polynomial basis functions
- Master cross-validation and model selection strategies
- Apply grid search for hyperparameter optimization

================================================================================
"""
#%%
# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

"""
Setup: Import necessary libraries for data manipulation, visualization,
statistical analysis, machine learning models, and preprocessing.
"""

from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

# Pandas and NumPy for data manipulation
import numpy as np
import pandas as pd
import scipy.stats as st
import copy

# Scikit-learn: regression models
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, ElasticNet,
    LassoCV, LassoLarsCV, RidgeCV
)

# Scikit-learn: preprocessing and feature engineering
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Scikit-learn: decomposition techniques
from sklearn.decomposition import PCA

# Scikit-learn: pipeline and model selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_validate, cross_val_score,
    GridSearchCV, KFold, LeaveOneOut
)

# Scikit-learn: evaluation metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Scikit-learn: feature selection
from sklearn.feature_selection import mutual_info_regression

# Matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns

# Enable inline plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('fivethirtyeight')

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

"""
Configure matplotlib parameters for consistent, professional-looking plots.
Define color palettes and text box properties for annotations.
"""

# Font configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

# Color and image configuration
plt.rcParams['image.cmap'] = 'jet'
colors = list(mcolors.TABLEAU_COLORS.values())
cmap = cm.get_cmap('RdYlGn')

# Text box properties for annotations
bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)


# ============================================================================
# SECTION 2: DATASET LOADING AND EXPLORATION
# ============================================================================

"""
DATASET DESCRIPTION: BOSTON HOUSING PRICE PREDICTION

The Boston Housing dataset contains housing price information for 506 census
tracts in the Boston area. Each record has 13 features describing neighborhood
characteristics and 1 target variable (median house value).

FEATURES:
    1. CRIM      - Per capita crime rate by town
    2. ZN        - Proportion of residential land zoned for lots over 25,000 sq.ft
    3. INDUS     - Proportion of non-retail business acres per town
    4. CHAS      - Charles River dummy variable (1 = bounds river, 0 = otherwise)
    5. NOX       - Nitric oxides concentration (parts per 10 million)
    6. RM        - Average number of rooms per dwelling
    7. AGE       - Proportion of owner-occupied units built prior to 1940
    8. DIS       - Weighted distances to five Boston employment centers
    9. RAD       - Index of accessibility to radial highways
   10. TAX       - Full-value property-tax rate per $10,000
   11. PTRATIO   - Pupil-teacher ratio by town
   12. B         - 1000(Bk - 0.63)^2, where Bk is the proportion of Black residents
   13. LSTAT     - Percentage lower status population

TARGET VARIABLE:
   14. MEDV      - Median value of homes in $1000s (PREDICTION TARGET)
"""
#%%
def get_file(filename):
    """
    Load dataset from local or remote source depending on execution environment.
    
    Parameters:
    -----------
    filename : str
        Name of the file to load
        
    Returns:
    --------
    str
        Full path to the dataset file
    """
    filepath = "./"#"../dataset/"
    url = "https://tvml.github.io/ml2425/dataset/"
    
    try:
        # Check if running in Google Colab
        IS_COLAB = ('google.colab' in str(get_ipython()))
    except:
        IS_COLAB = False
    
    if IS_COLAB:
        import urllib.request
        urllib.request.urlretrieve(url + filename, filename)
        return filename
    else:
        return filepath + filename
#%%

# Load dataset into pandas DataFrame
df = pd.read_csv(get_file('housing.data.txt'), header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
              'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {df.columns.tolist()}")

#%%
# ============================================================================
# DATASET VISUALIZATION AND CORRELATION ANALYSIS
# ============================================================================

"""
PAIRPLOT VISUALIZATION:
This matrix displays the joint distributions of selected features. The diagonal
shows the individual distribution of each feature (represented as kernel density
estimate plots), while the off-diagonal elements show scatter plots of pairs of
features. This visualization helps identify:
- Univariate distributions
- Potential correlations between variables
- Presence of outliers
- Non-linear relationships
"""

# Select key features for visualization
cols = ['LSTAT', 'RM', 'INDUS', 'AGE', 'MEDV']

# Create pairplot (commented out to avoid excessive output)
# fig = sns.pairplot(df[cols], height=4, diag_kind='kde',
#                    plot_kws=dict(color=colors[8]),
#                    diag_kws=dict(shade=True, alpha=0.7, color=colors[0]))
# plt.show()


"""
CORRELATION MATRIX HEATMAP:
The correlation matrix displays the linear correlation coefficient between each
pair of features. The value at position (i, j) represents:
    - Corr(i, j) ∈ [-1, 1]
    - +1 : Perfect positive correlation
    - 0  : No linear correlation
    - -1 : Perfect negative (inverse) correlation

Mathematical definition:
    r(X, Y) = cov(X, Y) / (σ_X * σ_Y)

where cov(X, Y) is the covariance and σ denotes standard deviation.
"""

cm = np.corrcoef(df[cols].values.T)

# Create heatmap visualization
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
            fmt='.2f',
            annot_kws={'size': 10},
            yticklabels=cols,
            xticklabels=cols,
            cmap=cmap,
            ax=ax)
plt.tight_layout()
# plt.show()

#%%
# ============================================================================
# SECTION 3: SINGLE-FEATURE REGRESSION ANALYSIS
# ============================================================================

"""
REGRESSION ON A SINGLE FEATURE:
This section analyzes the relationship between the target variable (MEDV) and
individual input features. We use mutual information to identify the most
informative feature.

MUTUAL INFORMATION:
A measure of dependency between two random variables. It quantifies how much
knowing one variable tells us about another variable. Features with high mutual
information with the target are more predictive.

    MI(X; Y) = ∑∑ p(x,y) * log(p(x,y) / (p(x)*p(y)))
"""

# Calculate mutual information between each feature and target
mi = mutual_info_regression(df[df.columns[:-1]], df[df.columns[-1]])
dmi = pd.DataFrame(mi, index=df.columns[:-1], columns=['MI']).sort_values(
    by='MI', ascending=False)

print("\nMutual Information Ranking:")
print(dmi.head(10))

# Select the most informative feature
feat = dmi.index[0]
print(f"\nMost informative feature: {feat}")

# Prepare data: extract feature and target
X = df[[feat]].values
y = df['MEDV'].values

#%%
# ============================================================================
# SUBSECTION 3.1: BASIC LINEAR REGRESSION
# ============================================================================

"""
LINEAR REGRESSION FUNDAMENTALS:

Mathematical Model:
The linear regression model predicts output as a linear function of input:
    y(w, x) = w0 + w1 * x

Cost Function (Mean Squared Error):
    C(w) = (1/2) * Σ_i [y(w, x_i) - t_i]²

where:
    - w = [w0, w1] are the model parameters (intercept and slope)
    - x_i is the i-th input feature
    - t_i is the i-th target value
    - n is the number of samples

The objective is to find weights w that minimize this cost function. The
solution can be obtained analytically using the normal equation or through
iterative optimization methods.

QUALITY METRICS:

1. Mean Squared Error (MSE):
    MSE = (1/n) * Σ_i [y(w, x_i) - t_i]²
    
    - Penalizes larger errors more heavily (due to squaring)
    - Always non-negative, with 0 indicating perfect predictions
    - Same units as squared target variable

2. Root Mean Squared Error (RMSE):
    RMSE = √MSE
    
    - Brings the metric back to original scale of target variable
    - Interpretable: average magnitude of prediction error

3. R² Score (Coefficient of Determination):
    R² = 1 - (SS_res / SS_tot)
    
    where:
        SS_res = Σ_i (t_i - ŷ_i)²  (residual sum of squares)
        SS_tot = Σ_i (t_i - ȳ)²     (total sum of squares)
    
    - Ranges from 0 to 1 (can be negative for poor models)
    - R² = 1 indicates perfect prediction
    - R² = 0 indicates model performs as well as predicting mean
"""

# Create and train basic linear regression model
model_lr = LinearRegression()
model_lr.fit(X, y)

# Make predictions
y_pred = model_lr.predict(X)

# Calculate metrics
mse = mean_squared_error(y_pred, y)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\n--- Basic Linear Regression (Feature: {feat}) ---")
print(f"Intercept (w0): {model_lr.intercept_:.3f}")
print(f"Slope (w1):     {model_lr.coef_[0]:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
#%%
# Visualization
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey', alpha=0.6,
           label='Data points')
ax.plot(x_range, model_lr.predict(x_range), color=colors[2], linewidth=2,
        label='Regression line')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV (Median House Value in $1000s)', fontsize=11, fontweight='bold')
ax.set_title(f'Linear Regression on Single Feature ({feat})', fontsize=13, fontweight='bold')
ax.text(0.85, 0.9, f'MSE: {mse:.3f}\nR²: {r2:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

#%%
# ============================================================================
# SUBSECTION 3.2: TRAIN/TEST SPLIT EVALUATION
# ============================================================================

"""
TRAIN/TEST SPLIT:

Motivation:
The model's performance on training data does not necessarily reflect its ability
to generalize to unseen data. A model that memorizes training data (overfitting)
will show excellent training performance but poor test performance.

Methodology:
1. Partition the dataset into two disjoint sets:
   - Training set (typically 70-80%): used to learn model parameters
   - Test set (typically 20-30%): used to evaluate generalization

2. Train the model on the training set
3. Evaluate on the test set (which the model has never seen)
4. Compare training and test performance:
   - Similar performance → Good generalization
   - Low training error, high test error → Overfitting
   - High training error, high test error → Underfitting

This gives a realistic estimate of how the model will perform on new data.
"""

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Create pipeline with standardization (optional here but good practice)
pipe_lr = Pipeline([
    ('regression', LinearRegression())
])

# Train on training set
pipe_lr.fit(X_train, y_train)

# Make predictions
y_train_pred = pipe_lr.predict(X_train)
y_test_pred = pipe_lr.predict(X_test)

# Calculate metrics for both sets
mse_train = mean_squared_error(y_train_pred, y_train)
mse_test = mean_squared_error(y_test_pred, y_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

model = pipe_lr.named_steps['regression']
print(f"\n--- Linear Regression with Train/Test Split ---")
print(f"Intercept: {model.intercept_:.3f}")
print(f"Slope: {model.coef_[0]:.3f}")
print(f"Training MSE: {mse_train:.3f}, R²: {r2_train:.3f}")
print(f"Test MSE: {mse_test:.3f}, R²: {r2_test:.3f}")

# Visualization
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X_train, y_train, c=colors[8], edgecolor='xkcd:light grey',
           alpha=0.6, label='Training data')
ax.scatter(X_test, y_test, c=colors[0], edgecolor='black', alpha=0.6,
           label='Test data')
ax.plot(x_range, pipe_lr.predict(x_range), color=colors[2], linewidth=2,
        label='Regression line')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title(f'Linear Regression with Train/Test Split', fontsize=13, fontweight='bold')
ax.text(0.9, 0.9,
        f'Train MSE: {mse_train:.3f}\nTest MSE: {mse_test:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props, ha='right')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

#%%
# ============================================================================
# SUBSECTION 3.3: FEATURE STANDARDIZATION
# ============================================================================

"""
FEATURE STANDARDIZATION (Z-SCORE NORMALIZATION):

Motivation:
Different features often have different scales (units and ranges). Linear
regression solutions can be sensitive to feature scaling, particularly when
using regularization. Standardization ensures:
1. All features have equal influence during optimization
2. Hyperparameters have consistent interpretation
3. Faster convergence in gradient-based optimization

Mathematical Formulation:
For each feature x, compute:
    z = (x - μ) / σ

where:
    μ = mean of feature x = (1/n) * Σ_i x_i
    σ = standard deviation = √[(1/n) * Σ_i (x_i - μ)²]

Result: Standardized feature z has:
    - Mean = 0
    - Standard deviation (variance^0.5) = 1

Implementation:
Use sklearn.preprocessing.StandardScaler which:
1. Computes mean and std from training data
2. Applies transformation: z = (x - training_mean) / training_std
3. Must use SAME transformation for test data (fit on train, transform on test)
"""

# Create pipeline with standardization and regression
pipe_scaled = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

# Fit on training data
pipe_scaled.fit(X_train, y_train)

# Predictions
y_train_pred_scaled = pipe_scaled.predict(X_train)
y_test_pred_scaled = pipe_scaled.predict(X_test)

# Metrics
mse_train_scaled = mean_squared_error(y_train_pred_scaled, y_train)
mse_test_scaled = mean_squared_error(y_test_pred_scaled, y_test)

scaler = pipe_scaled.named_steps['scaler']
model_scaled = pipe_scaled.named_steps['regression']

print(f"\n--- Linear Regression with Standardization ---")
print(f"Scaler - Mean: {scaler.mean_[0]:.3f}, Std Dev: {scaler.scale_[0]:.3f}")
print(f"Intercept: {model_scaled.intercept_:.3f}")
print(f"Slope: {model_scaled.coef_[0]:.3f}")
print(f"Training MSE: {mse_train_scaled:.3f}")
print(f"Test MSE: {mse_test_scaled:.3f}")

# Visualization
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X_train, y_train, c=colors[8], edgecolor='xkcd:light grey',
           alpha=0.6, label='Training data')
ax.scatter(X_test, y_test, c=colors[0], edgecolor='black', alpha=0.6,
           label='Test data')
ax.plot(x_range, pipe_scaled.predict(x_range), color=colors[2], linewidth=2,
        label='Regression line')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title('Linear Regression with Standardized Features',
             fontsize=13, fontweight='bold')
ax.text(0.9, 0.9,
        f'Train MSE: {mse_train_scaled:.3f}\nTest MSE: {mse_test_scaled:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props, ha='right')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()


# ============================================================================
# SUBSECTION 3.4: K-FOLD CROSS-VALIDATION
# ============================================================================

"""
K-FOLD CROSS-VALIDATION:

Problem with Single Train/Test Split:
The performance estimate depends on the random choice of which samples go into
training vs. test sets. This introduces high variance in the performance metric.

Solution: K-Fold Cross-Validation
1. Partition dataset into k roughly equal-sized disjoint subsets (folds)
2. For each fold i:
   - Use fold i as test set
   - Use remaining k-1 folds as training set
   - Train model and compute test error
3. Final performance = average of all k error scores
4. Also report standard deviation to quantify uncertainty

Advantages:
- Uses all data for both training and validation
- Reduces variance in performance estimate
- Better utilization of limited data
- More reliable assessment of generalization

Common values: k = 5 or k = 10
Trade-off: larger k → better estimate but more computation

Mathematical notation:
For k-fold CV, the cross-validation error is:
    CV_error = (1/k) * Σ_j Error_j
    
where Error_j is the test error in fold j.
"""

# Perform K-fold cross-validation manually (for educational purposes)
pipe_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_models = []
cv_mse_scores = []

print(f"\n--- 5-Fold Cross-Validation ---")

for fold_idx, (train_idx, test_idx) in enumerate(k_fold.split(X)):
    # Get train/test splits for this fold
    X_fold_train, X_fold_test = X[train_idx], X[test_idx]
    y_fold_train, y_fold_test = y[train_idx], y[test_idx]
    
    # Train on this fold
    pipe_cv.fit(X_fold_train, y_fold_train)
    
    # Evaluate on test fold
    y_fold_pred = pipe_cv.predict(X_fold_test)
    fold_mse = mean_squared_error(y_fold_pred, y_fold_test)
    cv_mse_scores.append(fold_mse)
    
    # Store trained model
    cv_models.append(copy.deepcopy(pipe_cv))
    
    print(f"Fold {fold_idx + 1}: MSE = {fold_mse:.3f}")

cv_mse_mean = np.mean(cv_mse_scores)
cv_mse_std = np.std(cv_mse_scores)

print(f"\nCross-validation results:")
print(f"Mean MSE: {cv_mse_mean:.3f} (± {cv_mse_std:.3f})")

# Visualization: plot predictions from all folds
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey', alpha=0.6,
           label='Data points')

# Plot prediction line from each fold
for i, model in enumerate(cv_models):
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax.plot(x_range, model.predict(x_range), color=colors[i % 7],
            linewidth=1, alpha=0.7)

ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title('Linear Regression with 5-Fold Cross-Validation',
             fontsize=13, fontweight='bold')
ax.legend(['Data'] + [f'Fold {i+1}' for i in range(len(cv_models))],
          loc='upper left', fontsize=9)
plt.tight_layout()
# plt.show()


# Using sklearn's cross_val_score for efficiency
print(f"\n--- Using sklearn cross_val_score ---")
pipe_cv_sklearn = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

cv_scores = cross_val_score(
    estimator=pipe_cv_sklearn, X=X, y=y,
    cv=5, scoring='neg_mean_squared_error'
)

# Convert negative MSE back to positive
cv_mse_scores_sklearn = -cv_scores

print(f"MSE scores for each fold: {cv_mse_scores_sklearn}")
print(f"Mean CV MSE: {cv_mse_scores_sklearn.mean():.3f}")
print(f"Std CV MSE: {cv_mse_scores_sklearn.std():.3f}")


# ============================================================================
# SECTION 4: REGULARIZED REGRESSION METHODS
# ============================================================================

"""
REGULARIZATION:

Problem with Unregularized Regression:
Without constraints on model complexity, regression can overfit to training data,
especially when:
1. Number of features is large relative to number of samples
2. Features are highly correlated
3. Using high-degree polynomial features

Regularization adds a penalty term to the cost function that discourages
overly complex models. The three main approaches:

1. L1 Regularization (Lasso)
2. L2 Regularization (Ridge)
3. Elastic Net (combination of L1 and L2)

All follow the form:
    C_total(w) = C_data(w) + λ * C_regularization(w)

where:
    - C_data is the original cost (MSE)
    - λ (lambda/alpha) is the regularization strength (hyperparameter)
    - C_regularization penalizes model complexity

Trade-off parameter λ:
    - λ = 0: No regularization (standard regression)
    - λ → ∞: Strong regularization (model becomes very simple)
    - Optimal λ: Found via cross-validation
"""


# ============================================================================
# SUBSECTION 4.1: LASSO REGRESSION (L1 REGULARIZATION)
# ============================================================================

"""
LASSO REGRESSION (Least Absolute Shrinkage and Selection Operator):

Cost Function:
    C(w) = (1/2n) * Σ_i [y(w, x_i) - t_i]² + (λ/2) * Σ_j |w_j|

where:
    - First term: Mean Squared Error (data fit)
    - Second term: L1 penalty = sum of absolute values of weights
    - λ: Regularization strength (hyperparameter to tune)

Key Properties:
1. Encourages sparsity: many coefficients become exactly zero
2. Performs automatic feature selection (zero coefficients = excluded features)
3. Useful when you suspect many features are irrelevant
4. More interpretable models (fewer non-zero coefficients)

Intuition:
The absolute value penalty |w_j| creates a "sharp corner" at w_j = 0,
making the optimizer more likely to set coefficients exactly to zero.

Interpretation of λ:
    - λ = 0: Standard linear regression (no penalty)
    - Small λ: Light penalty (few coefficients zero)
    - Large λ: Heavy penalty (many coefficients zero, simpler model)
    - λ → ∞: All coefficients → 0 (constant model)

Trade-off:
As λ increases:
    - Training error increases (less flexible)
    - Test error may first decrease (less overfitting) then increase
    - Optimal λ found where test error is minimized
"""

# LASSO with fixed hyperparameter
alpha_lasso = 0.5

pipe_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Lasso(alpha=alpha_lasso, max_iter=10000))
])

pipe_lasso.fit(X, y)

# Cross-validation evaluation
cv_scores_lasso = cross_val_score(
    estimator=pipe_lasso, X=X, y=y,
    cv=10, scoring='neg_mean_squared_error'
)

mse_lasso = mean_squared_error(pipe_lasso.predict(X), y)
mse_cv_lasso = -cv_scores_lasso.mean()

print(f"\n--- LASSO Regression (L1 Regularization) ---")
print(f"Alpha: {alpha_lasso}")
print(f"Training MSE: {mse_lasso:.3f}")
print(f"CV Mean MSE: {mse_cv_lasso:.3f} (± {-cv_scores_lasso.std():.3f})")

# Visualization
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey', alpha=0.6,
           label='Data points')
ax.plot(x_range, pipe_lasso.predict(x_range), color=colors[2], linewidth=2,
        label='LASSO fit')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title(f'LASSO Regression (L1 Regularization, λ={alpha_lasso})',
             fontsize=13, fontweight='bold')
ax.text(0.9, 0.9,
        f'Train MSE: {mse_lasso:.3f}\nCV MSE: {mse_cv_lasso:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props, ha='right')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
# plt.show()


# ============================================================================
# SUBSECTION 4.2: RIDGE REGRESSION (L2 REGULARIZATION)
# ============================================================================

"""
RIDGE REGRESSION:

Cost Function:
    C(w) = (1/2n) * Σ_i [y(w, x_i) - t_i]² + (λ/2) * Σ_j w_j²

where:
    - First term: Mean Squared Error (data fit)
    - Second term: L2 penalty = sum of squared weights
    - λ: Regularization strength (hyperparameter to tune)

Key Properties:
1. Shrinks coefficients toward zero but NEVER to exactly zero
2. All features remain in the model (no feature selection)
3. Particularly effective for:
       - Multicollinearity (correlated features)
       - When you want to keep all features
       - Large number of features with small effects
4. More stable than LASSO for prediction

Intuition:
The squared penalty w_j² is smooth (no sharp corners), so the optimizer
will shrink coefficients gradually but won't set them to zero.

Comparison with LASSO (L1 vs L2):
    LASSO (L1):          Ridge (L2):
    - Sparse solutions    - Non-sparse solutions
    - Feature selection   - Keep all features
    - Sharp penalty       - Smooth penalty
    - Better for         - Better for
      interpretability     multicollinearity

Geometric Interpretation:
In weight space, L1 penalty creates a diamond-shaped constraint region,
while L2 creates a circular constraint region. This explains why L1
tends to produce corner solutions (some w_j = 0) while L2 doesn't.
"""

# Ridge with fixed hyperparameter
alpha_ridge = 10.0

pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Ridge(alpha=alpha_ridge))
])

pipe_ridge.fit(X, y)

# Cross-validation evaluation
cv_scores_ridge = cross_val_score(
    estimator=pipe_ridge, X=X, y=y,
    cv=10, scoring='neg_mean_squared_error'
)

mse_ridge = mean_squared_error(pipe_ridge.predict(X), y)
mse_cv_ridge = -cv_scores_ridge.mean()

print(f"\n--- RIDGE Regression (L2 Regularization) ---")
print(f"Alpha: {alpha_ridge}")
print(f"Training MSE: {mse_ridge:.3f}")
print(f"CV Mean MSE: {mse_cv_ridge:.3f} (± {-cv_scores_ridge.std():.3f})")

# Visualization
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey', alpha=0.6,
           label='Data points')
ax.plot(x_range, pipe_ridge.predict(x_range), color=colors[2], linewidth=2,
        label='Ridge fit')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title(f'RIDGE Regression (L2 Regularization, λ={alpha_ridge})',
             fontsize=13, fontweight='bold')
ax.text(0.9, 0.9,
        f'Train MSE: {mse_ridge:.3f}\nCV MSE: {mse_cv_ridge:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props, ha='right')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
# plt.show()


# ============================================================================
# SUBSECTION 4.3: ELASTIC NET REGRESSION
# ============================================================================

"""
ELASTIC NET:

Cost Function:
    C(w) = (1/2n) * Σ_i [y(w, x_i) - t_i]² + (λ/2) * [γ * Σ_j |w_j| + (1-γ) * Σ_j w_j²]

where:
    - First term: Mean Squared Error
    - Second term: Combination of L1 and L2 penalties
    - λ: Overall regularization strength
    - γ ∈ [0, 1]: Balance between L1 and L2
        * γ = 0: Pure L2 (Ridge)
        * γ = 1: Pure L1 (LASSO)
        * 0 < γ < 1: Combination of both

Motivation:
Combines advantages of both LASSO and Ridge:
1. Can select features (like LASSO) through L1 term
2. Handles multicollinearity (like Ridge) through L2 term
3. More stable than LASSO in high-dimensional settings
4. Better for correlated features

When to use Elastic Net:
- You want feature selection but suspect correlated features
- You want to reduce model complexity without losing interpretability
- High-dimensional data with feature correlations
- More robust than LASSO for small sample sizes

Parameter Tuning:
Two hyperparameters to optimize:
1. λ (alpha): Overall regularization strength
2. γ (l1_ratio): Balance between L1 and L2
   - l1_ratio = 0: Pure L2 (Ridge)
   - l1_ratio = 1: Pure L1 (LASSO)
   - 0 < l1_ratio < 1: Elastic Net
"""

# Elastic Net with fixed hyperparameters
alpha_en = 0.5
gamma_en = 0.3

pipe_elasticnet = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', ElasticNet(alpha=alpha_en, l1_ratio=gamma_en, max_iter=10000))
])

pipe_elasticnet.fit(X, y)

# Cross-validation evaluation
cv_scores_en = cross_val_score(
    estimator=pipe_elasticnet, X=X, y=y,
    cv=10, scoring='neg_mean_squared_error'
)

mse_en = mean_squared_error(pipe_elasticnet.predict(X), y)
mse_cv_en = -cv_scores_en.mean()

print(f"\n--- ELASTIC NET Regression ---")
print(f"Alpha: {alpha_en}, Gamma (L1 ratio): {gamma_en}")
print(f"Training MSE: {mse_en:.3f}")
print(f"CV Mean MSE: {mse_cv_en:.3f} (± {-cv_scores_en.std():.3f})")

# Visualization
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey', alpha=0.6,
           label='Data points')
ax.plot(x_range, pipe_elasticnet.predict(x_range), color=colors[2],
        linewidth=2, label='Elastic Net fit')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title(
    f'Elastic Net Regression (λ={alpha_en}, γ={gamma_en})',
    fontsize=13, fontweight='bold')
ax.text(0.9, 0.9,
        f'Train MSE: {mse_en:.3f}\nCV MSE: {mse_cv_en:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props, ha='right')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
# plt.show()


# ============================================================================
# SECTION 5: POLYNOMIAL BASIS FUNCTIONS
# ============================================================================

"""
POLYNOMIAL BASIS FUNCTIONS:

Motivation:
Linear models assume a linear relationship between input and output.
Real-world data often exhibits non-linear relationships. Polynomial basis
functions allow linear models to fit non-linear patterns.

Polynomial Regression Model:
For a single input x, create polynomial features:
    φ(x) = [1, x, x², x³, ..., x^d]

Then apply linear regression to these engineered features:
    y = w₀ + w₁x + w₂x² + w₃x³ + ... + w_d x^d

This is still LINEAR in the parameters w, but NONLINEAR in the input x.

Mathematical Notation:
    y(w, φ(x)) = Σ_j w_j * φ_j(x)

where:
    - φ_j(x) = x^(j-1) for j = 1, 2, ..., d+1
    - d = polynomial degree

Advantages:
1. Captures non-linear relationships while keeping model linear in weights
2. Same linear regression algorithms apply
3. Easy to interpret polynomial coefficients

Challenges:
1. Choice of degree d is critical
2. High-degree polynomials can severely overfit
3. Polynomial features are highly correlated (multicollinearity)
4. Numerical instability for high degrees and large input values

Overfitting Risk:
- Degree too high: model learns training noise, poor generalization
- Degree too low: underfitting, high bias
- Optimal degree found via cross-validation

Implementation:
sklearn.preprocessing.PolynomialFeatures automatically generates all
polynomial terms up to specified degree.
"""

# Polynomial regression with degree 3
degree = 3

pipe_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('features', PolynomialFeatures(degree=degree, include_bias=True)),
    ('regression', LinearRegression())
])

pipe_poly.fit(X, y)

# Cross-validation evaluation
cv_scores_poly = cross_val_score(
    estimator=pipe_poly, X=X, y=y,
    cv=10, scoring='neg_mean_squared_error'
)

mse_poly = mean_squared_error(pipe_poly.predict(X), y)
mse_cv_poly = -cv_scores_poly.mean()

print(f"\n--- Polynomial Regression (Degree {degree}) ---")
print(f"Training MSE: {mse_poly:.3f}")
print(f"CV Mean MSE: {mse_cv_poly:.3f} (± {-cv_scores_poly.std():.3f})")

# Visualization
x_min = np.floor(X.min())
x_max = np.ceil(X.max())
x_range = np.linspace(x_min, x_max, 200).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(X, y, c=colors[8], edgecolor='xkcd:light grey', alpha=0.6,
           label='Data points')
ax.plot(x_range, pipe_poly.predict(x_range), color=colors[2], linewidth=2,
        label=f'Polynomial fit (degree {degree})')
ax.set_xlabel(feat, fontsize=11, fontweight='bold')
ax.set_ylabel('MEDV', fontsize=11, fontweight='bold')
ax.set_title(f'Polynomial Regression (Degree {degree})',
             fontsize=13, fontweight='bold')
ax.text(0.9, 0.9,
        f'Train MSE: {mse_poly:.3f}\nCV MSE: {mse_cv_poly:.3f}',
        fontsize=11, transform=ax.transAxes, bbox=bbox_props, ha='right')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
# plt.show()


"""
RESIDUAL ANALYSIS:

Residuals are the differences between predicted and actual values:
    residual_i = ŷ_i - y_i

A residual plot shows residuals (y-axis) vs. predicted values (x-axis).

Good residual plots should show:
1. Points randomly scattered around y=0 (no systematic pattern)
2. Similar spread across all predicted value ranges (homoscedasticity)
3. No obvious outliers

Problems indicated by residual patterns:
1. Curved pattern: non-linearity not captured (need higher degree)
2. Increasing spread: heteroscedasticity (variance depends on predicted value)
3. Systematic bias: model systematically over/under predicts
4. Outliers: unusual data points or measurement errors
"""

# Residual analysis for polynomial model
y_pred_poly = pipe_poly.predict(X)
residuals = y - y_pred_poly

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(y_pred_poly, residuals, c=colors[8], edgecolor='xkcd:light grey',
           alpha=0.6, label='Residuals')
ax.axhline(y=0, color=colors[2], linewidth=2, linestyle='--', label='Zero line')
ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax.set_ylabel('Residuals (y - ŷ)', fontsize=11, fontweight='bold')
ax.set_title(f'Residual Plot - Polynomial Regression (Degree {degree})',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
# plt.show()


"""
HYPERPARAMETER TUNING: POLYNOMIAL DEGREE

Study how model performance varies with polynomial degree.
Key questions:
1. At what degree does test error reach minimum? (optimal complexity)
2. How much does error increase for degrees beyond optimal? (overfitting severity)
3. What is the bias-variance trade-off?
"""

# Test multiple polynomial degrees
poly_degrees = range(1, 20)
results_poly = []

for deg in poly_degrees:
    pipe_poly_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('features', PolynomialFeatures(degree=deg, include_bias=True)),
        ('regression', LinearRegression())
    ])
    
    pipe_poly_temp.fit(X, y)
    
    # Training and CV errors
    mse_train_temp = mean_squared_error(pipe_poly_temp.predict(X), y)
    cv_scores_temp = cross_val_score(
        estimator=pipe_poly_temp, X=X, y=y,
        cv=10, scoring='neg_mean_squared_error'
    )
    mse_cv_temp = -cv_scores_temp.mean()
    
    results_poly.append({
        'degree': deg,
        'train_mse': mse_train_temp,
        'cv_mse': mse_cv_temp
    })

results_poly_df = pd.DataFrame(results_poly)
print("\n--- Polynomial Degree Tuning Results ---")
print(results_poly_df.head(10))

# Visualization: MSE vs Polynomial Degree
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(results_poly_df['degree'], results_poly_df['train_mse'],
        marker='o', label='Training MSE', color=colors[8], linewidth=2)
ax.plot(results_poly_df['degree'], results_poly_df['cv_mse'],
        marker='s', label='CV MSE', color=colors[2], linewidth=2)
ax.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
ax.set_title('Model Performance vs Polynomial Degree',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# Find optimal degree
optimal_idx = results_poly_df['cv_mse'].idxmin()
optimal_degree = results_poly_df.loc[optimal_idx, 'degree']
optimal_cv_mse = results_poly_df.loc[optimal_idx, 'cv_mse']

print(f"\nOptimal polynomial degree: {optimal_degree}")
print(f"CV MSE at optimal degree: {optimal_cv_mse:.3f}")


# ============================================================================
# SECTION 6: MULTI-FEATURE REGRESSION
# ============================================================================

"""
MULTI-FEATURE REGRESSION:

Problem Extension:
So far, we've used only a single feature for regression. Real problems
typically have multiple features. We now extend to:
    y = w₀ + w₁x₁ + w₂x₂ + ... + w_p x_p

where:
    - x₁, x₂, ..., x_p are p input features
    - w₀, w₁, ..., w_p are p+1 parameters to learn

Cost Function (MSE for multi-feature case):
    C(w) = (1/n) * Σ_i [Σ_j w_j x_ij - t_i]²

Matrix Notation:
    C(w) = (1/n) * ||Xw - t||²₂

where:
    - X is n × p design matrix (n samples, p features)
    - w is p-dimensional weight vector
    - t is n-dimensional target vector

Challenges with Multiple Features:
1. Curse of dimensionality: high-dimensional data, large p
2. Feature scaling: features with large scales dominate
3. Multicollinearity: correlated features lead to unstable solutions
4. Overfitting: more parameters to fit, higher risk of overfitting
5. Interpretability: harder to understand which features matter

Solutions:
1. Feature standardization/normalization
2. Feature selection (select subset of features)
3. Dimensionality reduction (PCA)
4. Regularization (L1, L2, Elastic Net)
5. Cross-validation for robust evaluation
"""

# Extract all features except target
X_all = df[df.columns[:-1]].values
y_all = df['MEDV'].values

print(f"\n--- Multi-Feature Regression ---")
print(f"Number of samples: {X_all.shape[0]}")
print(f"Number of features: {X_all.shape[1]}")
print(f"Feature names: {df.columns[:-1].tolist()}")

# Basic linear regression (no scaling)
model_lr_all = LinearRegression()
model_lr_all.fit(X_all, y_all)
mse_lr_all = mean_squared_error(model_lr_all.predict(X_all), y_all)
r2_lr_all = r2_score(y_all, model_lr_all.predict(X_all))

print(f"\nLinear Regression (no scaling):")
print(f"Training MSE: {mse_lr_all:.3f}, R²: {r2_lr_all:.3f}")

# With feature standardization
pipe_lr_all = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])
pipe_lr_all.fit(X_all, y_all)
mse_lr_scaled = mean_squared_error(pipe_lr_all.predict(X_all), y_all)
r2_lr_scaled = r2_score(y_all, pipe_lr_all.predict(X_all))

print(f"\nLinear Regression (with scaling):")
print(f"Training MSE: {mse_lr_scaled:.3f}, R²: {r2_lr_scaled:.3f}")

# Cross-validation evaluation
cv_scores_lr_all = cross_val_score(
    estimator=pipe_lr_all, X=X_all, y=y_all,
    cv=5, scoring='neg_mean_squared_error'
)

print(f"\n5-Fold Cross-Validation:")
print(f"MSE scores: {-cv_scores_lr_all}")
print(f"Mean CV MSE: {-cv_scores_lr_all.mean():.3f} (± {-cv_scores_lr_all.std():.3f})")


# ============================================================================
# SUBSECTION 6.1: REGULARIZED MULTI-FEATURE REGRESSION
# ============================================================================

"""
Applying regularization to multi-feature problem.
"""

# LASSO with all features
alpha_lasso_all = 0.5

pipe_lasso_all = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Lasso(alpha=alpha_lasso_all, max_iter=10000))
])

cv_scores_lasso_all = cross_val_score(
    estimator=pipe_lasso_all, X=X_all, y=y_all,
    cv=5, scoring='neg_mean_squared_error'
)

print(f"\n--- LASSO with all features ---")
print(f"Alpha: {alpha_lasso_all}")
print(f"Mean CV MSE: {-cv_scores_lasso_all.mean():.3f} (± {-cv_scores_lasso_all.std():.3f})")

# Ridge with all features
alpha_ridge_all = 10.0

pipe_ridge_all = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Ridge(alpha=alpha_ridge_all))
])

cv_scores_ridge_all = cross_val_score(
    estimator=pipe_ridge_all, X=X_all, y=y_all,
    cv=5, scoring='neg_mean_squared_error'
)

print(f"\n--- RIDGE with all features ---")
print(f"Alpha: {alpha_ridge_all}")
print(f"Mean CV MSE: {-cv_scores_ridge_all.mean():.3f} (± {-cv_scores_ridge_all.std():.3f})")

# Elastic Net with all features
alpha_en_all = 0.5
gamma_en_all = 0.3

pipe_en_all = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', ElasticNet(alpha=alpha_en_all, l1_ratio=gamma_en_all, max_iter=10000))
])

cv_scores_en_all = cross_val_score(
    estimator=pipe_en_all, X=X_all, y=y_all,
    cv=5, scoring='neg_mean_squared_error'
)

print(f"\n--- ELASTIC NET with all features ---")
print(f"Alpha: {alpha_en_all}, Gamma: {gamma_en_all}")
print(f"Mean CV MSE: {-cv_scores_en_all.mean():.3f} (± {-cv_scores_en_all.std():.3f})")


# ============================================================================
# SECTION 7: HYPERPARAMETER TUNING AND MODEL SELECTION
# ============================================================================

"""
HYPERPARAMETER TUNING:

Hyperparameters vs Parameters:
    - Parameters: learned from data (e.g., weights w)
    - Hyperparameters: set before learning (e.g., alpha in regularization)

Key Hyperparameters:
    1. Regularization strength (alpha, lambda)
    2. Regularization type (L1, L2, Elastic Net mix)
    3. Model complexity (polynomial degree)
    4. Learning rate, batch size, etc. (for gradient-based optimization)

Challenge:
How to find the best hyperparameter values?

Solutions:
1. Grid Search: Try a discrete grid of values
2. Random Search: Sample random values from distributions
3. Bayesian Optimization: Model hyperparameter performance surface
4. Cross-Validation: Use CV scores to compare different hyperparameters

GRID SEARCH METHODOLOGY:

1. Define a grid of hyperparameter values to test
2. For each combination of hyperparameters:
       a. Create model with those hyperparameters
       b. Evaluate using cross-validation
       c. Record CV performance
3. Select hyperparameters with best CV performance
4. Train final model with selected hyperparameters on entire dataset
5. Evaluate on held-out test set

Example Grid Search:
    For LASSO, search alpha ∈ {0.01, 0.1, 1, 10, 100}
    For Ridge, search alpha ∈ {0.001, 0.01, 0.1, 1, 10}
    For Elastic Net, search pairs (alpha, gamma) from 2D grid

Advantages of Grid Search:
    - Systematic exploration of hyperparameter space
    - Parallelizable (test multiple combinations in parallel)
    - Reproducible
    - Easy to implement

Disadvantages:
    - Computationally expensive for large grids
    - May miss optimal values between grid points
    - Curse of dimensionality (exponential growth with hyperparameter count)
"""


# ============================================================================
# SUBSECTION 7.1: GRID SEARCH FOR LASSO ALPHA
# ============================================================================

"""
LASSO: Search for optimal alpha using Grid Search

Manual implementation using K-Fold CV:
"""

# Define search domain
alpha_domain_lasso = np.linspace(0, 10, 100)
cv_folds = 10
cv_results_lasso = []

print(f"\n--- LASSO: Manual Grid Search for Alpha ---")
print(f"Domain: alpha ∈ [{alpha_domain_lasso.min()}, {alpha_domain_lasso.max()}]")
print(f"Grid points: {len(alpha_domain_lasso)}")
print(f"CV folds: {cv_folds}")

kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

for alpha in alpha_domain_lasso:
    pipe_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', Lasso(alpha=alpha, max_iter=10000))
    ])
    
    fold_errors = []
    for train_idx, test_idx in kf.split(X_all):
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_test_fold = X_all[test_idx]
        y_test_fold = y_all[test_idx]
        
        pipe_temp.fit(X_train_fold, y_train_fold)
        y_pred_fold = pipe_temp.predict(X_test_fold)
        fold_error = mean_squared_error(y_test_fold, y_pred_fold)
        fold_errors.append(fold_error)
    
    mean_cv_error = np.mean(fold_errors)
    cv_results_lasso.append((alpha, mean_cv_error))

cv_results_lasso = np.array(cv_results_lasso)

# Find optimal alpha
best_idx_lasso = np.argmin(cv_results_lasso[:, 1])
best_alpha_lasso = cv_results_lasso[best_idx_lasso, 0]
best_mse_lasso = cv_results_lasso[best_idx_lasso, 1]

print(f"\nBest alpha: {best_alpha_lasso:.5f}")
print(f"Best CV MSE: {best_mse_lasso:.3f}")

# Visualization: CV error vs Alpha
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(cv_results_lasso[:, 0], cv_results_lasso[:, 1], linewidth=2, color=colors[8])
ax.scatter([best_alpha_lasso], [best_mse_lasso], s=200, color=colors[2],
           marker='*', edgecolors='red', linewidth=2, zorder=5,
           label=f'Optimal: α={best_alpha_lasso:.3f}')
ax.set_xlabel('Alpha (Regularization Strength)', fontsize=11, fontweight='bold')
ax.set_ylabel('Cross-Validation MSE', fontsize=11, fontweight='bold')
ax.set_title('LASSO: Grid Search for Optimal Alpha',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()


"""
Using scikit-learn's GridSearchCV:
Simplified interface for grid search with automatic handling of CV.
"""

print(f"\n--- LASSO: GridSearchCV ---")

# Define parameter grid
param_grid_lasso = {
    'regression__alpha': np.linspace(0, 10, 100)
}

pipe_lasso_gs = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Lasso(max_iter=10000))
])

# GridSearchCV automatically performs CV
grid_search_lasso = GridSearchCV(
    estimator=pipe_lasso_gs,
    param_grid=param_grid_lasso,
    cv=10,
    scoring='neg_mean_squared_error',
    n_jobs=-1  # Use all CPU cores
)

grid_search_lasso.fit(X_all, y_all)

best_alpha_lasso_gs = grid_search_lasso.best_params_['regression__alpha']
best_mse_lasso_gs = -grid_search_lasso.best_score_

print(f"Best alpha: {best_alpha_lasso_gs:.5f}")
print(f"Best CV MSE: {best_mse_lasso_gs:.3f}")

# Visualization
cv_results_gs = grid_search_lasso.cv_results_['mean_test_score']

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(param_grid_lasso['regression__alpha'], -cv_results_gs,
        linewidth=2, color=colors[8])
ax.scatter([best_alpha_lasso_gs], [best_mse_lasso_gs], s=200, color=colors[2],
           marker='*', edgecolors='red', linewidth=2, zorder=5,
           label=f'Optimal: α={best_alpha_lasso_gs:.3f}')
ax.set_xlabel('Alpha', fontsize=11, fontweight='bold')
ax.set_ylabel('CV MSE', fontsize=11, fontweight='bold')
ax.set_title('LASSO: GridSearchCV Results',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()


"""
Using LassoCV:
Specialized class that efficiently searches for optimal alpha.
"""

print(f"\n--- LASSO: LassoCV ---")

alpha_domain_cv = np.linspace(0, 10, 100)

pipe_lasso_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LassoCV(cv=10, alphas=alpha_domain_cv, max_iter=10000))
])

pipe_lasso_cv.fit(X_all, y_all)

best_alpha_lasso_cv = pipe_lasso_cv.named_steps['regression'].alpha_

# Compute CV scores for each alpha
cv_scores_lasso_all_alphas = np.mean(
    pipe_lasso_cv.named_steps['regression'].mse_path_, axis=1
)

print(f"Best alpha: {best_alpha_lasso_cv:.5f}")

# Find MSE at best alpha
best_mse_lasso_cv = cv_scores_lasso_all_alphas[
    np.argmin(np.abs(alpha_domain_cv - best_alpha_lasso_cv))
]
print(f"CV MSE at best alpha: {best_mse_lasso_cv:.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(alpha_domain_cv, cv_scores_lasso_all_alphas, linewidth=2, color=colors[8])
ax.scatter([best_alpha_lasso_cv], [best_mse_lasso_cv], s=200, color=colors[2],
           marker='*', edgecolors='red', linewidth=2, zorder=5,
           label=f'Optimal: α={best_alpha_lasso_cv:.3f}')
ax.set_xlabel('Alpha', fontsize=11, fontweight='bold')
ax.set_ylabel('CV MSE', fontsize=11, fontweight='bold')
ax.set_title('LASSO: LassoCV Results',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# Final evaluation with optimal alpha
pipe_lasso_final = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Lasso(alpha=best_alpha_lasso_cv, max_iter=10000))
])

cv_scores_final_lasso = cross_val_score(
    estimator=pipe_lasso_final, X=X_all, y=y_all,
    cv=20, scoring='neg_mean_squared_error'
)

print(f"\nFinal evaluation with optimal alpha:")
print(f"20-Fold CV MSE: {-cv_scores_final_lasso.mean():.3f} (± {-cv_scores_final_lasso.std():.3f})")

# Inspect non-zero coefficients (feature selection effect of LASSO)
pipe_lasso_final.fit(X_all, y_all)
coef_lasso = pipe_lasso_final.named_steps['regression'].coef_

print(f"\nLASSO coefficients (non-zero features):")
feature_names = df.columns[:-1].tolist()
for i, (name, coef) in enumerate(zip(feature_names, coef_lasso)):
    if coef != 0:
        print(f"  {name}: {coef:.6f}")


# ============================================================================
# SUBSECTION 7.2: GRID SEARCH FOR RIDGE ALPHA
# ============================================================================

"""
RIDGE: Search for optimal alpha using similar methodology as LASSO.
Note: Ridge typically requires larger alpha values than LASSO.
"""

print(f"\n--- RIDGE: Grid Search for Alpha ---")

# Ridge usually needs larger alpha values
alpha_domain_ridge = np.linspace(0.1, 100, 100)

# Using RidgeCV (specialized class for Ridge hyperparameter search)
pipe_ridge_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', RidgeCV(alphas=alpha_domain_ridge, cv=20, store_cv_values=True))
])

pipe_ridge_cv.fit(X_all, y_all)

best_alpha_ridge = pipe_ridge_cv.named_steps['regression'].alpha_

# Get CV scores
cv_values_ridge = pipe_ridge_cv.named_steps['regression'].cv_values_
cv_scores_ridge_all = np.mean(cv_values_ridge, axis=0)

print(f"Best alpha: {best_alpha_ridge:.5f}")

# Find MSE at best alpha
best_mse_ridge = cv_scores_ridge_all[
    np.argmin(np.abs(alpha_domain_ridge - best_alpha_ridge))
]
print(f"CV MSE at best alpha: {best_mse_ridge:.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(alpha_domain_ridge, cv_scores_ridge_all, linewidth=2, color=colors[8])
ax.scatter([best_alpha_ridge], [best_mse_ridge], s=200, color=colors[2],
           marker='*', edgecolors='red', linewidth=2, zorder=5,
           label=f'Optimal: α={best_alpha_ridge:.3f}')
ax.set_xlabel('Alpha', fontsize=11, fontweight='bold')
ax.set_ylabel('CV MSE', fontsize=11, fontweight='bold')
ax.set_title('RIDGE: RidgeCV Results',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()

# Final evaluation
pipe_ridge_final = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Ridge(alpha=best_alpha_ridge))
])

cv_scores_final_ridge = cross_val_score(
    estimator=pipe_ridge_final, X=X_all, y=y_all,
    cv=20, scoring='neg_mean_squared_error'
)

print(f"\nFinal evaluation with optimal alpha:")
print(f"20-Fold CV MSE: {-cv_scores_final_ridge.mean():.3f} (± {-cv_scores_final_ridge.std():.3f})")

# Inspect coefficients
pipe_ridge_final.fit(X_all, y_all)
coef_ridge = pipe_ridge_final.named_steps['regression'].coef_

print(f"\nRIDGE coefficients (all features retained, some shrunk to near zero):")
for name, coef in zip(feature_names, coef_ridge):
    print(f"  {name}: {coef:.6f}")


# ============================================================================
# SUBSECTION 7.3: GRID SEARCH FOR ELASTIC NET
# ============================================================================

"""
ELASTIC NET: 2D Grid Search for optimal (alpha, gamma) pair
"""

print(f"\n--- ELASTIC NET: 2D Grid Search ---")

# Define 2D parameter grid
alpha_domain_en = np.linspace(0.01, 1, 10)
gamma_domain_en = np.linspace(0, 1, 10)

param_grid_en = {
    'regression__alpha': alpha_domain_en,
    'regression__l1_ratio': gamma_domain_en
}

pipe_en_gs = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', ElasticNet(max_iter=10000))
])

# GridSearchCV with 2D parameter grid
grid_search_en = GridSearchCV(
    estimator=pipe_en_gs,
    param_grid=param_grid_en,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search_en.fit(X_all, y_all)

best_alpha_en = grid_search_en.best_params_['regression__alpha']
best_gamma_en = grid_search_en.best_params_['regression__l1_ratio']
best_mse_en = -grid_search_en.best_score_

print(f"Best alpha: {best_alpha_en:.5f}")
print(f"Best gamma (L1 ratio): {best_gamma_en:.5f}")
print(f"Best CV MSE: {best_mse_en:.3f}")

# Final evaluation
pipe_en_final = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', ElasticNet(alpha=best_alpha_en, l1_ratio=best_gamma_en, max_iter=10000))
])

cv_scores_final_en = cross_val_score(
    estimator=pipe_en_final, X=X_all, y=y_all,
    cv=20, scoring='neg_mean_squared_error'
)

print(f"\nFinal evaluation with optimal parameters:")
print(f"20-Fold CV MSE: {-cv_scores_final_en.mean():.3f} (± {-cv_scores_final_en.std():.3f})")


# ============================================================================
# SECTION 8: MODEL COMPARISON AND SUMMARY
# ============================================================================

"""
FINAL MODEL COMPARISON:

Summary of all regression models tested, their hyperparameters, and
cross-validation performance.
"""

print("\n" + "="*80)
print("FINAL MODEL COMPARISON - MULTI-FEATURE REGRESSION")
print("="*80)

summary_results = [
    {
        'Model': 'Linear Regression',
        'Hyperparameters': 'None',
        'Train MSE': mse_lr_scaled,
        'CV MSE': -cv_scores_lr_all.mean(),
        'CV Std': -cv_scores_lr_all.std()
    },
    {
        'Model': 'LASSO',
        'Hyperparameters': f'α={best_alpha_lasso_cv:.5f}',
        'Train MSE': mean_squared_error(pipe_lasso_final.predict(X_all), y_all),
        'CV MSE': -cv_scores_final_lasso.mean(),
        'CV Std': -cv_scores_final_lasso.std()
    },
    {
        'Model': 'RIDGE',
        'Hyperparameters': f'α={best_alpha_ridge:.5f}',
        'Train MSE': mean_squared_error(pipe_ridge_final.predict(X_all), y_all),
        'CV MSE': -cv_scores_final_ridge.mean(),
        'CV Std': -cv_scores_final_ridge.std()
    },
    {
        'Model': 'ELASTIC NET',
        'Hyperparameters': f'α={best_alpha_en:.5f}, γ={best_gamma_en:.5f}',
        'Train MSE': mean_squared_error(pipe_en_final.predict(X_all), y_all),
        'CV MSE': -cv_scores_final_en.mean(),
        'CV Std': -cv_scores_final_en.std()
    }
]

summary_df = pd.DataFrame(summary_results)
print("\n", summary_df.to_string(index=False))

# Determine best model by CV MSE
best_model_idx = summary_df['CV MSE'].idxmin()
best_model_name = summary_df.loc[best_model_idx, 'Model']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"CV MSE: {summary_df.loc[best_model_idx, 'CV MSE']:.3f}")
print(f"{'='*80}")


print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# %%
