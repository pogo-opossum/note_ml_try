# %% [markdown]
# # Gaussian Process Regression
#
# A **Gaussian Process (GP)** is a probabilistic model that defines a distribution over functions.
# Rather than learning a single function, a GP maintains a probability distribution over all
# functions consistent with the observed data, providing both predictions and uncertainty estimates.
#
# Formally, a GP is a collection of random variables, any finite subset of which follows a
# multivariate Gaussian distribution. It is fully specified by:
#
# - a **mean function** $m(x) = \mathbb{E}[f(x)]$, typically assumed to be zero
# - a **covariance (kernel) function** $\kappa(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$
#
# We write: $f(x) \sim \mathcal{GP}(m(x),\, \kappa(x, x'))$
#
# In this notebook we implement GP regression from scratch, then compare it with scikit-learn's
# `GaussianProcessRegressor`.

# %%
# --- Standard library and numerical computing ---
import warnings
import itertools

import numpy as np

# --- Plotting ---
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors             # custom colourmap construction
import seaborn as sns

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Global Plot Configuration
#
# We set a consistent visual style for all figures in this notebook.

# %%
# Apply ggplot base style
plt.style.use("ggplot")                         # clean grid-based style
plt.rcParams.update({
    "font.family":         "sans-serif",        # base font family
    "font.serif":          "Ubuntu",            # serif fallback
    "font.monospace":      "Ubuntu Mono",       # monospace for code labels
    "font.size":           10,                  # default text size
    "axes.labelsize":      10,                  # axis label size
    "axes.labelweight":    "bold",              # bold axis labels
    "axes.titlesize":      10,                  # subplot title size
    "xtick.labelsize":     8,                   # x-tick text size
    "ytick.labelsize":     8,                   # y-tick text size
    "legend.fontsize":     10,                  # legend text size
    "figure.titlesize":    12,                  # suptitle size
    "image.cmap":          "jet",               # default image colormap
    "image.interpolation": "none",              # no interpolation for imshow
    "figure.figsize":      (16, 8),             # default figure size (inches)
    "lines.linewidth":     2,                   # default line width
    "lines.markersize":    8,                   # default marker size
})

# Human-readable colour names from the xkcd palette for clear labelling
COLORS = [
    "xkcd:pale orange", "xkcd:sea blue",    "xkcd:pale red",
    "xkcd:sage green",  "xkcd:terra cotta", "xkcd:dull purple",
    "xkcd:teal",        "xkcd:goldenrod",   "xkcd:cadet blue",
    "xkcd:scarlet",
]

cmap = mcolors.LinearSegmentedColormap.from_list("", ["#82cafc", "#069af3", "#0485d1", COLORS[0], COLORS[8]])

# %% [markdown]
# ## The Target Function
#
# We define a latent (hidden) function $f : \mathbb{R} \to \mathbb{R}$ that we wish to recover
# from a finite set of noisy observations. This function is deliberately chosen to be
# non-trivial, combining oscillations at two different frequencies:
#
# $$
# f(x) = \sin\!\left(\frac{3\pi x}{10}\right) + \cos\!\left(\frac{7\pi x}{10}\right)
# $$
#
# In a realistic scenario, $f$ is unknown; here we keep it visible to evaluate the quality
# of the GP predictions.

# %%
d = 1  # Input space dimension (univariate regression)

def f(x):
    """True latent function to be recovered by GP regression."""
    return np.sin((3 * np.pi) * x / 10) + np.cos((7 * np.pi) * x / 10)

# Dense grid of 1000 equally spaced points over [0, 10] — used for plotting the true curve
X = np.linspace(start=0, stop=10, num=1000)
Y = f(X)

# %% [markdown]
# ## Generating Training Data
#
# We draw $n$ input points uniformly at random from $[0, 10]$ and evaluate $f$ on them.
# Two scenarios are considered:
#
# 1. **Noise-free observations**: $t_i = f(x_i)$
# 2. **Noisy observations**: $t_i = f(x_i) + \varepsilon_i$, where
#    $\varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_n^2)$

# %%
# --- Noise-free training set ---
n = 10                              # Number of training samples
x = np.sort(np.random.rand(n) * 10) # Sorted random inputs in [0, 10]
t = f(x)                            # Noise-free targets

# --- Noisy training set ---
sigma_n = 0.2                                             # Noise standard deviation
epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)  # i.i.d. Gaussian noise
t_n     = t + epsilon                                      # Noisy targets

# %% [markdown]
# ### Visualization: Noise-Free Training Set
#
# We overlay the $n = 10$ noise-free observations on the true curve $f(x)$ to confirm that
# the sampled points lie exactly on it.

# %%
fig, ax = plt.subplots()
plt.plot(X, Y, color=COLORS[1], label='$f(x)$', zorder=2)
plt.scatter(x, t, color=COLORS[0], label='Training set (no noise)', zorder=3)
plt.legend(loc='upper left')
plt.title(r'$f(x) = \sin(3\pi x/10) + \cos(7\pi x/10)$ — noise-free training set');

# %% [markdown]
# ### Visualization: Noise Distribution
#
# The histogram below confirms that the additive errors $\varepsilon_i$ are approximately
# Gaussian with mean zero and standard deviation $\sigma_n$.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
sns.distplot(a=epsilon, color=COLORS[1], ax=ax)   # Kernel density + histogram of errors
ax.set(title=f'Error distribution  ($\\sigma_n = {sigma_n}$)');

# %% [markdown]
# ### Visualization: Noisy Training Set
#
# The scatter plot now shows observations perturbed by Gaussian noise, making the regression
# problem more realistic and challenging.

# %%
fig, ax = plt.subplots()
plt.plot(X, Y, color=COLORS[1], label='$f(x)$', zorder=2)
plt.scatter(x, t_n, color=COLORS[0], label=f'Training set (Gaussian noise, $\\sigma_n={sigma_n}$)', zorder=3)
plt.legend(loc='upper left')
plt.title(r'$f(x)$ with noisy training set ($\sigma_n = {}$)'.format(sigma_n));

# %% [markdown]
# ## Test Points
#
# We predict on the same dense grid $X$ used to plot $f$, so that we can compare
# the GP posterior directly with the true function over the full domain $[0, 10]$.

# %%
x_star = X  # 1000 test points coinciding with the plotting grid

# %% [markdown]
# ## Kernel Function (Squared Exponential / RBF)
#
# A Gaussian process is completely characterised by its kernel function. The most widely used
# choice is the **squared exponential** (also called *radial basis function* or *RBF*) kernel:
#
# $$
# \kappa_{\sigma_f, \ell}(x_p, x_q)
# = \sigma_f \exp\!\left(-\frac{\lVert x_p - x_q \rVert^2}{2\ell^2}\right)
# $$
#
# **Hyperparameters:**
#
# | Symbol | Name | Effect |
# |--------|------|--------|
# | $\sigma_f > 0$ | signal variance / amplitude | scales the overall output range |
# | $\ell > 0$ | length-scale | controls how quickly correlations decay with distance |
#
# A large $\ell$ means that distant points remain correlated (smooth, global behaviour);
# a small $\ell$ means correlations decay quickly (rough, local behaviour).

# %%
def kernel_function(x, y, sigma_f=1, l=1):
    """
    Squared exponential (RBF) kernel.

    Parameters
    ----------
    x, y     : scalar inputs
    sigma_f  : signal amplitude hyperparameter (default 1)
    l        : length-scale hyperparameter (default 1)

    Returns
    -------
    float : kernel value kappa(x, y)
    """
    return sigma_f * np.exp(-(np.linalg.norm(x - y) ** 2) / (2 * l ** 2))

# Baseline hyperparameter values used in the first experiments
l       = 0.8   # Length-scale
sigma_f = 1     # Amplitude

# %% [markdown]
# ## Computing the Covariance Matrices
#
# The joint prior distribution over training values $\mathbf{f}$ and test values $\mathbf{f}_*$
# is Gaussian, determined by the block covariance matrix:
#
# $$
# C =
# \begin{pmatrix}
# K(X, X) & K(X, X_*) \\
# K(X_*, X) & K(X_*, X_*)
# \end{pmatrix}
# $$
#
# where each block is defined entry-wise by the kernel:
#
# | Block | Notation | Size | Definition |
# |-------|----------|------|------------|
# | Training–training | $K(X, X)$ | $n \times n$ | $K_{ij} = \kappa(x_i, x_j)$ |
# | Test–test | $K(X_*, X_*)$ | $m \times m$ | $(K_*)_{ij} = \kappa(x^*_i, x^*_j)$ |
# | Test–training | $K(X_*, X)$ | $m \times n$ | $(k_*)_{ij} = \kappa(x^*_i, x_j)$ |
#
# Under a **Gaussian noise** model, the training-side block acquires a noise term:
# $K(X, X) \leftarrow K(X, X) + \sigma_n^2 I_n$.

# %%
def compute_cov_matrices(x, x_star, sigma_f=1, l=1, noise=True, sigma_n=0.1):
    """
    Build all covariance matrix blocks of the GP joint distribution.

    Parameters
    ----------
    x        : (n,)   array of training inputs
    x_star   : (m,)   array of test inputs
    sigma_f  : kernel amplitude hyperparameter
    l        : kernel length-scale hyperparameter
    noise    : if True, add sigma_n^2 * I to K (Gaussian noise model)
    sigma_n  : noise standard deviation (used only when noise=True)

    Returns
    -------
    C       : (n+m, n+m) full block covariance matrix
    K       : (n, n)     training covariance (+ noise term if noise=True)
    K_star  : (m, m)     test covariance
    k_star  : (m, n)     cross-covariance between test and training points
    """
    n      = x.shape[0]
    n_star = x_star.shape[0]

    # K(X, X) — kernel evaluated on all pairs of training points
    K = [kernel_function(i, j, sigma_f=sigma_f, l=l)
         for (i, j) in itertools.product(x, x)]
    K = np.array(K).reshape(n, n)

    # K(X*, X*) — kernel evaluated on all pairs of test points
    K_star = [kernel_function(i, j, sigma_f=sigma_f, l=l)
              for (i, j) in itertools.product(x_star, x_star)]
    K_star = np.array(K_star).reshape(n_star, n_star)

    # k(X*, X) — cross-covariance between test and training points
    k_star = [kernel_function(i, j, sigma_f=sigma_f, l=l)
              for (i, j) in itertools.product(x_star, x)]
    k_star = np.array(k_star).reshape(n_star, n)

    # Add noise to the diagonal of the training block if required
    if noise:
        K = K + (sigma_n ** 2) * np.eye(n)

    # Assemble the full block matrix C
    top    = np.concatenate((K,        k_star.T), axis=1)   # [K  | k_star^T]
    bottom = np.concatenate((k_star,   K_star),   axis=1)   # [k* | K*      ]
    C      = np.concatenate((top, bottom),         axis=0)

    return C, K, K_star, k_star

# %% [markdown]
# ## Case 1: Noise-Free Observations
#
# When observations are exact ($t_i = f(x_i)$), the joint distribution of training and test
# values is:
#
# $$
# \begin{pmatrix} \mathbf{f} \\ \mathbf{f}_* \end{pmatrix}
# \sim \mathcal{N}\!\left(\mathbf{0},\;
# \begin{pmatrix} K(X,X) & K(X,X_*) \\ K(X_*,X) & K(X_*,X_*) \end{pmatrix}
# \right)
# $$
#
# Conditioning on the training observations $\mathbf{f}$ yields the **posterior**:
#
# $$
# \mathbf{f}_* \mid \mathbf{f}
# \;\sim\; \mathcal{N}\!\left(\bar{\mathbf{f}}_*,\; \text{cov}(\mathbf{f}_*)\right)
# $$
#
# $$
# \bar{\mathbf{f}}_* = K(X_*, X)\, K(X, X)^{-1}\, \mathbf{f}
# $$
#
# $$
# \text{cov}(\mathbf{f}_*) = K(X_*, X_*) - K(X_*, X)\, K(X, X)^{-1}\, K(X, X_*)
# $$

# %%
# Compute covariance blocks — no noise added to K
C_n, K_n, K_star_n, k_star_n = compute_cov_matrices(
    x, x_star, sigma_f=sigma_f, l=l, noise=False
)

# %% [markdown]
# ### Visualization: Covariance Matrices (Noise-Free Case)
#
# The heatmaps below show the structure of each block:
#
# - **$K(X, X)$** ($n \times n$): high values near the diagonal indicate that nearby
#   training points are strongly correlated.
# - **$K(X_*, X_*)$** ($m \times m$): smooth decay of correlation as test-point distance
#   increases.
# - **$K(X_*, X)$** ($m \times n$): cross-correlations used to transfer information from
#   training to test locations.
# - **Full block matrix $C$**: confirms the block structure of the joint prior.

# %%
# --- K(X, X) ---
print(f'K(X, X)  shape: {K_n.shape}')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_n, cmap='Blues', ax=ax)
plt.title('$K(X, X)$ — training covariance (no noise)');

# %%
# --- K(X*, X*) ---
print(f'K(X*, X*)  shape: {K_star_n.shape}')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_star_n, cmap='Blues', ax=ax)
plt.title('$K(X_*, X_*)$ — test covariance (no noise)');

# %%
# --- k(X*, X) ---
print(f'k(X*, X)  shape: {k_star_n.shape}')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=k_star_n, cmap='Blues', ax=ax)
plt.title('$k(X_*, X)$ — cross-covariance (no noise)');

# %%
# --- Full block matrix C ---
print(f'C  shape: {C_n.shape}')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=C_n, cmap='Blues', ax=ax)
plt.title('Full covariance matrix $C$ — no noise');

# %% [markdown]
# ### Visualization: Samples from the Prior Distribution (Noise-Free)
#
# Before conditioning on any observations, the GP prior assigns equal probability to all
# smooth functions consistent with the kernel. Each coloured curve below is one random
# function drawn from $\mathcal{N}(\mathbf{0},\, K(X_*, X_*))$.
#
# The spread of the samples reflects the prior uncertainty: we have no preference for any
# particular function shape.

# %%
fig, ax = plt.subplots()
mean_prior = np.zeros(len(X))          # Zero mean function
for _ in range(20):
    # Sample a function from the GP prior
    z_star = np.random.multivariate_normal(mean=mean_prior, cov=K_star_n)
    plt.plot(X, z_star, color=COLORS[0], alpha=0.3)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
ax.set(title='Samples from the GP prior — no noise')
ax.legend(loc='lower right');

# %% [markdown]
# ### GP Posterior Parameters
#
# The function below computes the closed-form posterior mean and covariance.
# These follow directly from the standard conditioning formula for multivariate Gaussians.

# %%
def compute_gpr_parameters(K, K_star, k_star, y):
    """
    Compute the GP posterior mean and covariance given observations.

    Parameters
    ----------
    K       : (n, n)   training covariance (with or without noise)
    K_star  : (m, m)   test covariance
    k_star  : (m, n)   cross-covariance (test × training)
    y       : (n,)     observed targets

    Returns
    -------
    f_bar_star  : (m, 1)  posterior mean at test points
    cov_f_star  : (m, m)  posterior covariance at test points
    """
    n           = K.shape[0]
    K_inv       = np.linalg.inv(K)                                      # (n, n) inverse of training covariance

    # Posterior mean: mu_* = k(X*, X) K^{-1} y
    f_bar_star  = k_star @ K_inv @ y.reshape(n, d)                      # (m, 1)

    # Posterior covariance: Sigma_* = K(X*, X*) - k(X*, X) K^{-1} k(X, X*)
    cov_f_star  = K_star - k_star @ K_inv @ k_star.T                    # (m, m)

    return f_bar_star, cov_f_star

# %%
# Compute the posterior for the noise-free case (using noise-free targets t)
m_n, c_n = compute_gpr_parameters(K_n, K_star_n, k_star_n, t)

# %% [markdown]
# ### Visualization: Samples from the Posterior Distribution (Noise-Free)
#
# After conditioning on the training points, the posterior distribution is sharply
# concentrated near $f(x)$. Each curve is drawn from
# $\mathcal{N}(\bar{\mathbf{f}}_*,\, \text{cov}(\mathbf{f}_*))$.
#
# Notice that all posterior samples pass exactly through the training observations
# (shown in red), because there is no noise in this scenario.

# %%
fig, ax = plt.subplots()
for _ in range(20):
    # Sample a function from the GP posterior
    z_star = np.random.multivariate_normal(mean=m_n.ravel(), cov=c_n)
    plt.plot(X, z_star, color=COLORS[0], alpha=0.3)
    plt.scatter(x, t, color=COLORS[4], zorder=4)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
ax.set(title='Samples from the GP posterior — no noise')
ax.legend(loc='lower right');

# %% [markdown]
# ### Point-Wise Predictive Distribution
#
# For each test point $x^*$ we compute:
# - the **predictive mean** $\bar{f}(x^*)$
# - the **predictive standard deviation** $\sigma(x^*)$
#
# The shaded region $\bar{f}(x^*) \pm \sigma(x^*)$ visualises the model's epistemic
# uncertainty: it is narrow near observed data and wider in regions with no training points.

# %%
def predict(x_new, X_train, t_train, noise=False):
    """
    Compute the predictive mean and variance at a single test point.

    Parameters
    ----------
    x_new    : scalar test input
    X_train  : (n,) training inputs
    t_train  : (n,) training targets
    noise    : whether to include noise in the model

    Returns
    -------
    mean : float  predictive mean at x_new
    var  : float  predictive variance at x_new
    """
    _, K, K_s, k_s = compute_cov_matrices(
        X_train, np.array([x_new]), l=l, sigma_n=sigma_n, sigma_f=sigma_f, noise=noise
    )
    m, c = compute_gpr_parameters(K, K_s, k_s, t_train)
    return m.item(), c.item()   # Scalar mean and variance

# %%
# Compute predictive mean and std for every point on the plotting grid (no noise)
z_star_n = np.array([predict(z, x, t, noise=False) for z in X])  # (1000, 2)

# %% [markdown]
# ### Visualization: Predictive Distribution with Uncertainty Band (Noise-Free)
#
# The green line is the posterior mean $\bar{f}(x^*)$; the shaded band covers
# $\bar{f}(x^*) \pm \sigma(x^*)$.  
# Where training data are dense, uncertainty collapses; in the gaps between observations
# it expands.

# %%
fig, ax = plt.subplots()
plt.plot(X, z_star_n[:, 0], color=COLORS[3], label='Posterior mean')
plt.fill_between(
    X,
    z_star_n[:, 0] - z_star_n[:, 1],   # Lower bound: mean - 1 std
    z_star_n[:, 0] + z_star_n[:, 1],   # Upper bound: mean + 1 std
    color=COLORS[3], alpha=0.3, label='$\\pm 1\\sigma$ band'
)
plt.scatter(x, t, color=COLORS[4], zorder=4, label='Training points')
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
ax.set(title='Predictive distribution — no noise')
ax.legend(loc='lower right');

# %% [markdown]
# ## Case 2: Gaussian Noise Observations
#
# We now assume that each observation is corrupted by independent Gaussian noise:
# $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$.
#
# The joint distribution of noisy targets $\mathbf{y}$ and test values $\mathbf{f}_*$ becomes:
#
# $$
# \begin{pmatrix} \mathbf{y} \\ \mathbf{f}_* \end{pmatrix}
# \sim \mathcal{N}\!\left(\mathbf{0},\;
# \begin{pmatrix} K(X,X) + \sigma_n^2 I & K(X,X_*) \\ K(X_*,X) & K(X_*,X_*) \end{pmatrix}
# \right)
# $$
#
# Conditioning gives the same structural formulas as before, but with the regularised
# inverse $(K + \sigma_n^2 I)^{-1}$:
#
# $$
# \bar{\mathbf{f}}_* = K(X_*, X)\,(K + \sigma_n^2 I)^{-1}\,\mathbf{y}
# $$
#
# $$
# \text{cov}(\mathbf{f}_*) = K(X_*, X_*) - K(X_*, X)\,(K + \sigma_n^2 I)^{-1}\,K(X, X_*)
# $$
#
# The noise term $\sigma_n^2 I$ acts as a **Tikhonov regulariser**, preventing overfitting
# and ensuring that the posterior no longer interpolates the training points exactly.

# %%
# Compute covariance blocks — noise term added to K
C_g, K_g, K_star_g, k_star_g = compute_cov_matrices(
    x, x_star, sigma_f=sigma_f, sigma_n=sigma_n, l=l, noise=True
)

# %% [markdown]
# ### Visualization: Covariance Matrices (Gaussian Noise Case)
#
# The heatmaps mirror those in the noise-free case, but $K(X, X)$ now has a
# $\sigma_n^2$-inflated diagonal, clearly visible in the first plot.

# %%
# --- K(X, X) with noise ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_g, cmap='Blues', ax=ax)
plt.title('$K(X, X) + \\sigma_n^2 I$ — training covariance (Gaussian noise)');

# %%
# --- K(X*, X*) ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_star_g, cmap='Blues', ax=ax)
plt.title('$K(X_*, X_*)$ — test covariance (Gaussian noise)');

# %%
# --- Full block matrix C ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=C_g, cmap='Blues', ax=ax)
plt.title('Full covariance matrix $C$ — Gaussian noise');

# %% [markdown]
# ### Visualization: Samples from the Prior Distribution (Gaussian Noise)
#
# The prior is identical in structure to the noise-free case — it does not depend on the
# data. However, because the test covariance $K(X_*, X_*)$ is the same, the sample
# functions look similar. The noise will only manifest after conditioning.

# %%
fig, ax = plt.subplots()
mean_prior = np.zeros(len(X))
for _ in range(20):
    z_star = np.random.multivariate_normal(mean=mean_prior, cov=K_star_g)
    plt.plot(X, z_star, color=COLORS[0], alpha=0.3)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
ax.set(title='Samples from the GP prior — Gaussian noise')
ax.legend(loc='lower right');

# %% [markdown]
# ### Visualization: Samples from the Posterior Distribution (Gaussian Noise)
#
# After conditioning on the noisy training set, posterior samples no longer pass exactly
# through the observations. The model has been regularised by the noise assumption, so it
# produces smoother functions that trade off data fit against prior smoothness.

# %%
# Compute posterior parameters using noisy targets t_n
m_g, c_g = compute_gpr_parameters(K_g, K_star_g, k_star_g, t_n)

fig, ax = plt.subplots()
for _ in range(20):
    z_star = np.random.multivariate_normal(mean=m_g.ravel(), cov=c_g)
    plt.plot(X, z_star, color=COLORS[0], alpha=0.2)
    plt.scatter(x, t_n, color=COLORS[4], zorder=4)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
ax.set(title='Samples from the GP posterior — Gaussian noise')
ax.legend(loc='lower right');

# %% [markdown]
# ### Visualization: Predictive Distribution with Uncertainty Band (Gaussian Noise)
#
# Compared to the noise-free case, the uncertainty band is wider because the model
# accounts for the irreducible aleatoric uncertainty introduced by $\sigma_n^2$.  
# The posterior mean is a smoothed interpolation of the noisy targets.

# %%
# Compute point-wise predictive mean and std (noise=True)
z_star_g = np.array([predict(z, x, t_n, noise=True) for z in X])  # (1000, 2)

fig, ax = plt.subplots()
plt.plot(X, z_star_g[:, 0], color=COLORS[3], label='Posterior mean')
plt.fill_between(
    X,
    z_star_g[:, 0] - z_star_g[:, 1],
    z_star_g[:, 0] + z_star_g[:, 1],
    color=COLORS[3], alpha=0.3, label='$\\pm 1\\sigma$ band'
)
plt.scatter(x, t_n, color=COLORS[4], zorder=4, label='Training points (noisy)')
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
ax.set(title='Predictive distribution — Gaussian noise')
ax.legend(loc='lower right');

# %% [markdown]
# ## Effect of Kernel Hyperparameters
#
# The shape of the GP posterior is entirely determined by the kernel hyperparameters
# $\sigma_f$ and $\ell$. Here we study their influence systematically.
#
# | Hyperparameter | Interpretation | Effect on posterior |
# |----------------|---------------|---------------------|
# | $\sigma_f \uparrow$ | larger signal amplitude | wider uncertainty band, higher-amplitude functions |
# | $\ell \uparrow$ | longer correlation range | smoother, more global fit |
# | $\ell \downarrow$ | shorter correlation range | rougher, more local fit |
#
# The three configurations below illustrate these regimes.

# %% [markdown]
# ### Configuration 1: $\sigma_f = 2$, $\ell = 1$ (smooth, global)
#
# A moderate length-scale means that distant points still share significant correlation.
# The posterior covariance matrix will have off-diagonal values that are non-negligible,
# reflecting a relatively global fit.

# %%
l, sigma_f = 1, 2

# Compute matrices and posterior (noise-free for clarity)
_, K_h1, K_star_h1, k_star_h1 = compute_cov_matrices(
    x, x_star, sigma_f=sigma_f, l=l, noise=False
)
f_bar_h1, cov_h1 = compute_gpr_parameters(K_h1, K_star_h1, k_star_h1, t)

# %% [markdown]
# #### Posterior Covariance Heatmap — $\sigma_f = 2$, $\ell = 1$
#
# Off-diagonal entries are non-negligible, indicating long-range correlations between test points.

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=cov_h1, cmap='Blues', ax=ax)
plt.title(f'Posterior covariance — $\\sigma_f = {sigma_f}$, $\\ell = {l}$');

# %% [markdown]
# #### Posterior Samples — $\sigma_f = 2$, $\ell = 1$

# %%
_, K_h1n, K_star_h1n, k_star_h1n = compute_cov_matrices(
    x, x_star, sigma_f=sigma_f, l=l, noise=True
)
f_bar_h1n, cov_h1n = compute_gpr_parameters(K_h1n, K_star_h1n, k_star_h1n, t)

fig, ax = plt.subplots()
for _ in range(20):
    z = np.random.multivariate_normal(mean=f_bar_h1n.squeeze(), cov=cov_h1n)
    plt.plot(X, z, color=COLORS[0], alpha=0.2)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
plt.title(f'Posterior samples — $\\sigma_f = {sigma_f}$, $\\ell = {l}$ (smooth, global)')
plt.legend(loc='upper right');

# %% [markdown]
# ### Configuration 2: $\sigma_f = 2$, $\ell = 0.001$ (rough, local)
#
# A very small length-scale implies that only points extremely close to each other are
# correlated. The posterior covariance matrix is nearly diagonal, and the fitted functions
# become spiky and local — the model barely generalises beyond the training points.

# %%
l, sigma_f = 0.001, 2

_, K_h2, K_star_h2, k_star_h2 = compute_cov_matrices(
    x, x_star, sigma_f=sigma_f, l=l, noise=True
)
f_bar_h2, cov_h2 = compute_gpr_parameters(K_h2, K_star_h2, k_star_h2, t)

# %% [markdown]
# #### Posterior Covariance Heatmap — $\sigma_f = 2$, $\ell = 0.001$
#
# The near-diagonal structure confirms that test-point correlations vanish rapidly, producing
# a highly local model.

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=cov_h2, cmap='Blues', ax=ax)
plt.title(f'Posterior covariance — $\\sigma_f = {sigma_f}$, $\\ell = {l}$');

# %% [markdown]
# #### Posterior Samples — $\sigma_f = 2$, $\ell = 0.001$

# %%
fig, ax = plt.subplots()
for _ in range(20):
    z = np.random.multivariate_normal(mean=f_bar_h2.squeeze(), cov=cov_h2)
    plt.plot(X, z, color=COLORS[0], alpha=0.2)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
plt.title(f'Posterior samples — $\\sigma_f = {sigma_f}$, $\\ell = {l}$ (rough, local)')
plt.legend(loc='upper right');

# %% [markdown]
# ### Configuration 3: $\sigma_f = 50$, $\ell = 0.1$ (large amplitude, semi-local)
#
# A large $\sigma_f$ inflates the overall variance of the process: posterior samples
# can exhibit much higher amplitudes. Combined with a moderate $\ell$, the model is
# flexible but may overfit if $\sigma_f$ is not regularised.

# %%
l, sigma_f = 0.1, 50

_, K_h3, K_star_h3, k_star_h3 = compute_cov_matrices(
    x, x_star, sigma_f=sigma_f, l=l, noise=True
)
f_bar_h3, cov_h3 = compute_gpr_parameters(K_h3, K_star_h3, k_star_h3, t)

# %% [markdown]
# #### Posterior Covariance Heatmap — $\sigma_f = 50$, $\ell = 0.1$

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=cov_h3, cmap='Blues', ax=ax)
plt.title(f'Posterior covariance — $\\sigma_f = {sigma_f}$, $\\ell = {l}$');

# %% [markdown]
# #### Posterior Samples — $\sigma_f = 50$, $\ell = 0.1$
#
# The large amplitude hyperparameter $\sigma_f = 50$ causes posterior samples to have
# a wide spread. This illustrates that $\sigma_f$ controls the *scale* of the output,
# independently of how the correlations decay.

# %%
fig, ax = plt.subplots()
for _ in range(20):
    z = np.random.multivariate_normal(mean=f_bar_h3.squeeze(), cov=cov_h3)
    plt.plot(X, z, color=COLORS[0], alpha=0.2)
plt.plot(X, Y, color=COLORS[1], label='$f(x)$')
plt.title(f'Posterior samples — $\\sigma_f = {sigma_f}$, $\\ell = {l}$ (large amplitude)')
plt.legend(loc='upper right');

# %% [markdown]
# ### Summary of Hyperparameter Effects
#
# | Config | $\sigma_f$ | $\ell$ | Behaviour |
# |--------|-----------|--------|-----------|
# | 1 | 2 | 1.0 | Smooth, global — posterior samples vary slowly |
# | 2 | 2 | 0.001 | Rough, local — posterior samples vary rapidly |
# | 3 | 50 | 0.1 | High amplitude — posterior samples have large range |
#
# In practice, hyperparameters are **optimised** by maximising the **log marginal likelihood**
# (evidence), which balances data fit against model complexity:
#
# $$
# \log p(\mathbf{y} \mid X, \theta)
# = -\tfrac{1}{2}\,\mathbf{y}^\top (K_\theta + \sigma_n^2 I)^{-1} \mathbf{y}
#   -\tfrac{1}{2}\log|K_\theta + \sigma_n^2 I|
#   -\tfrac{n}{2}\log 2\pi
# $$
#
# The first term rewards data fit; the second penalises model complexity (log-determinant
# acts as a regulariser). This is exactly what scikit-learn's `GaussianProcessRegressor`
# does internally.

# %% [markdown]
# ## GP Regression with scikit-learn
#
# scikit-learn's `GaussianProcessRegressor` automates hyperparameter optimisation via
# repeated gradient-based maximisation of the log marginal likelihood (with random
# restarts to mitigate local optima).
#
# We demonstrate it on a more complex target function defined on a denser training set,
# to showcase its practical usability.

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# %% [markdown]
# ### New Target Function and Training Set
#
# We define a richer, higher-frequency target:
#
# $$
# f_2(x) = \sin(4\pi x) + \sin(7\pi x) + \sin(3\pi x)
# $$
#
# Training set: $n = 1000$ points uniformly spaced on $[0, 2]$, observed with noise
# $\sigma_n = 0.4$.

# %%
d       = 1                                               # Input dimension
n       = 1000                                            # Training set size
L       = 2                                               # Domain length
sigma_n = 0.4                                             # Noise standard deviation

x = np.linspace(start=0, stop=L, num=n)                  # Training inputs
X = x.reshape(n, d)                                       # Reshaped for sklearn (n × d)

def f(x):
    """Complex multi-frequency target function for the sklearn demo."""
    return np.sin(4 * np.pi * x) + np.sin(7 * np.pi * x) + np.sin(3 * np.pi * x)

f_x     = f(x)
epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)  # Gaussian noise
y       = f_x + epsilon                                    # Noisy observations

# %% [markdown]
# ### Test Set
#
# The test domain extends $0.5$ units beyond the training range to evaluate
# **extrapolation** behaviour of the GP.

# %%
n_star  = n + 300                                                    # More test points than training
x_star  = np.linspace(start=0, stop=(L + 0.5), num=n_star)          # Test inputs (extend domain)
X_star  = x_star.reshape(n_star, d)                                  # Reshaped for sklearn

# %% [markdown]
# ### Kernel and Model Definition
#
# We use a **Constant × RBF** kernel, which is equivalent to the squared exponential with
# a learnable amplitude:
#
# $$
# \kappa(x, x') = \sigma_f \cdot \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
# $$
#
# Both $\sigma_f$ and $\ell$ are initialised with our prior guesses but will be optimised
# by the `GaussianProcessRegressor` via marginal likelihood maximisation.

# %%
l, sigma_f = 0.1, 2   # Initial hyperparameter guesses

# ConstantKernel models sigma_f; RBF models the length-scale
kernel = (
    ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2))
    * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))
)

# GaussianProcessRegressor with noise regularisation alpha = sigma_n^2
# n_restarts_optimizer: number of random restarts to find the global optimum
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=sigma_n ** 2,       # Noise variance added to diagonal (equivalent to sigma_n^2 I)
    n_restarts_optimizer=10   # Multiple restarts for robust hyperparameter optimisation
)

# %% [markdown]
# ### Model Fitting and Prediction
#
# `gp.fit` maximises the log marginal likelihood to learn the optimal $\sigma_f$ and $\ell$.
# `gp.predict` returns the posterior mean at test points.
# `gp.sample_y` draws full function samples from the posterior.

# %%
# Fit the GP to training data (hyperparameter optimisation happens here)
gp.fit(X, y)

# Posterior mean at test points
y_pred = gp.predict(X_star)                                              # (n_star,)

# Draw n_star posterior samples to estimate mean and uncertainty empirically
y_hat_samples = gp.sample_y(X_star, n_samples=n_star)                   # (n_star, n_star)
y_hat         = np.mean(y_hat_samples, axis=1)                           # Sample mean
y_hat_sd      = np.std(y_hat_samples, axis=1)                            # Sample std

# %% [markdown]
# ### Visualization: sklearn GP Predictions and Credible Interval
#
# The plot below shows:
#
# - **Scatter**: noisy training observations
# - **Red line**: true function $f_2(x)$
# - **Green line**: GP posterior mean (sklearn prediction)
# - **Green shaded band**: $\pm 2\hat{\sigma}$ empirical credible interval
#
# Note how the uncertainty band **widens** in the extrapolation region $x > 2$,
# where there are no training observations.

# %%
fig, ax = plt.subplots(figsize=(15, 8))

# Noisy training data
sns.scatterplot(x=x, y=y, label='Training data', color=COLORS[4],ax=ax)

# True latent function
sns.lineplot(x=x_star, y=f(x_star), color=COLORS[1], label='$f_2(x)$', ax=ax)

# Credible interval (±2 std from posterior samples)
ax.fill_between(
    x_star,
    y_hat - 2 * y_hat_sd,   # Lower bound: mean - 2 std
    y_hat + 2 * y_hat_sd,   # Upper bound: mean + 2 std
    color=COLORS[3], alpha=0.3, label='$\\pm 2\\sigma$ credible interval'
)

# Posterior mean
sns.lineplot(x=x_star, y=y_pred, color=COLORS[3], label='Posterior mean (sklearn)', ax=ax)

ax.set(title='sklearn GP regression — predictions and credible interval')
ax.legend(loc='lower left');

# %%
