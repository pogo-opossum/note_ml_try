
# %% [markdown]
# ---
# # Gaussian Discriminant Analysis for Binary Classification
#
# ## 1 — Generative vs. Discriminative Classifiers
#
# Machine learning classifiers can be divided into two broad families depending
# on *what* they model:
#
# **Discriminative models** learn the conditional distribution $p(C_k \mid \mathbf{x})$
# directly from the data, without modelling how the data were generated.
# Examples: logistic regression, SVM, neural networks.
#
# **Generative models** instead learn the *joint* distribution
# $p(\mathbf{x}, C_k) = p(\mathbf{x} \mid C_k)\, p(C_k)$ and then apply
# Bayes' theorem to obtain the posterior:
#
# $$
# p(C_k \mid \mathbf{x})
# = \frac{p(\mathbf{x} \mid C_k)\; p(C_k)}{p(\mathbf{x})}
# = \frac{p(\mathbf{x} \mid C_k)\; p(C_k)}
#        {\displaystyle\sum_{j} p(\mathbf{x} \mid C_j)\, p(C_j)}.
# $$
#
# Because generative models explicitly represent the data-generating process,
# they can also *sample* new data, compute likelihoods, handle missing features,
# and naturally incorporate prior knowledge.  The price is that the assumed
# distributional form may be misspecified.
#
# ## 2 — Gaussian Discriminant Analysis (GDA)
#
# **GDA** is a generative classifier that models the class-conditional density
# of each class $C_k$ as a multivariate Gaussian:
#
# $$
# p(\mathbf{x} \mid C_k) = \mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k)
# = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_k|^{1/2}}
#   \exp\!\left(-\tfrac{1}{2}
#   (\mathbf{x}-\boldsymbol{\mu}_k)^\top
#   \boldsymbol{\Sigma}_k^{-1}
#   (\mathbf{x}-\boldsymbol{\mu}_k)\right).
# $$
#
# The parameters $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$ are estimated
# by **maximum likelihood** from the training data belonging to class $C_k$:
#
# $$
# \hat{\boldsymbol{\mu}}_k = \frac{1}{N_k}\sum_{i:\, t_i=k} \mathbf{x}_i,
# \qquad
# \hat{\boldsymbol{\Sigma}}_k = \frac{1}{N_k}\sum_{i:\, t_i=k}
#   (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)^\top.
# $$
#
# The class prior is estimated as the empirical class frequency:
# $\hat{\pi}_k = N_k / N$.
#
# ## 3 — Linear vs. Quadratic Decision Boundary
#
# The decision boundary is the set of points where the posterior probabilities
# of two classes are equal.  For the binary case $\{C_0, C_1\}$ this is
# $p(C_0 \mid \mathbf{x}) = p(C_1 \mid \mathbf{x})$, or equivalently the
# set where the **log-posterior ratio** is zero:
#
# $$
# \ln \frac{p(C_0 \mid \mathbf{x})}{p(C_1 \mid \mathbf{x})}
# = \ln \frac{p(\mathbf{x} \mid C_0)}{p(\mathbf{x} \mid C_1)}
#   + \ln \frac{\pi_0}{\pi_1} = 0.
# $$
#
# ### Case A — Shared covariance: $\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}$
#
# When both classes share the same covariance matrix the quadratic terms in the
# Gaussian exponents cancel out and the log-ratio simplifies to a *linear*
# function of $\mathbf{x}$:
#
# $$
# \boldsymbol{\theta}^\top \tilde{\mathbf{x}} = 0,
# \qquad
# \tilde{\mathbf{x}} = \begin{pmatrix}1 \\ \mathbf{x}\end{pmatrix},
# $$
#
# with coefficients:
#
# $$
# \boldsymbol{\theta}_{1:d} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1),
# \qquad
# \theta_0 = -\tfrac{1}{2}\boldsymbol{\mu}_0^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0
#             +\tfrac{1}{2}\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1
#             +\ln\frac{\pi_0}{\pi_1}.
# $$
#
# This is **Linear Discriminant Analysis (LDA)**: the decision boundary is a
# hyperplane (a line in 2-D).
#
# ### Case B — Class-specific covariances: $\boldsymbol{\Sigma}_0 \neq \boldsymbol{\Sigma}_1$
#
# When the covariance matrices differ, the quadratic terms do not cancel and
# the boundary is a *quadratic* surface — this is **Quadratic Discriminant
# Analysis (QDA)**.
#
# This notebook implements and compares both cases on a real 2-D dataset.
# 

# %%
# ── Suppress non-critical warnings ───────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

# %%
# ── Core libraries ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.stats as st

# %%
# ── Plotting configuration ────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("ggplot")
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.size":           10,
    "axes.labelsize":      10,
    "axes.labelweight":    "bold",
    "axes.titlesize":      11,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     10,
    "figure.titlesize":    12,
    "image.cmap":          "jet",
    "image.interpolation": "none",
    "figure.figsize":      (10, 8),
    "lines.linewidth":     2,
    "lines.markersize":    8,
})

# Accessible xkcd colour palette
COLORS = [
    "xkcd:pale orange", "xkcd:sea blue",     "xkcd:pale red",
    "xkcd:sage green",  "xkcd:terra cotta",  "xkcd:dull purple",
    "xkcd:teal",        "xkcd:goldenrod",    "xkcd:cadet blue",
    "xkcd:scarlet",     "xkcd:steel grey",   "xkcd:mint green",
    "xkcd:burnt orange","xkcd:royal purple", "xkcd:bright pink",
    "xkcd:olive green",
]

# Custom colormap: upper slice of Spectral (warm-to-cool, avoids extremes)
_cmap_full = plt.colormaps["Spectral"].resampled(512)
CMAP = mcolors.ListedColormap(_cmap_full(np.linspace(0.7, 0.95, 256)))

# %%
# ── File path helper ──────────────────────────────────────────────────────────
FILEPATH = "./"

def get_file(filename: str) -> str:
    return FILEPATH + filename

# %% [markdown]
# ---
# ## 4 — Dataset
#
# We load a two-feature dataset from a CSV file.  Each row contains two real-
# valued features $(x_1, x_2)$ and a binary label $t \in \{0, 1\}$ indicating
# class membership.  The dataset is split into class-specific subsets
# $\mathcal{D}_0$ and $\mathcal{D}_1$ for estimation and visualisation.
# 

# %%
# ── Load dataset ──────────────────────────────────────────────────────────────
data = pd.read_csv(
    get_file("ex2data1.txt"),
    header=0, delimiter=',',
    names=['x1', 'x2', 't']
)

# Dataset dimensions
n        = len(data)                 # total number of samples
n0       = (data.t == 0).sum()      # samples in class C0
n1       = n - n0                   # samples in class C1
nfeatures = len(data.columns) - 1   # number of features (2)

# Feature matrix and target vector
X = data[['x1', 'x2']].to_numpy()  # shape (N, 2)
t = data['t'].to_numpy()            # shape (N,)

# Class-specific subsets
X0 = X[t == 0]    # feature rows for C0
X1 = X[t == 1]    # feature rows for C1

print(f"N = {n}  (C0: {n0}, C1: {n1}),  features: {nfeatures}")

# %%
# ── Figure 1 – Raw dataset scatter plot ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X0[:, 0], X0[:, 1], s=40, color=COLORS[4], alpha=0.9, label='$C_0$')
ax.scatter(X1[:, 0], X1[:, 1], s=40, color=COLORS[1], alpha=0.9, label='$C_1$')
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Dataset', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5 — Parameter Estimation: LDA (Shared Covariance)
#
# ### Class means
#
# The maximum-likelihood estimate of the class mean is the sample mean
# restricted to each class:
#
# $$
# \hat{\boldsymbol{\mu}}_k = \frac{1}{N_k}\sum_{i:\,t_i=k}\mathbf{x}_i.
# $$
#
# ### Shared covariance matrix
#
# Under the LDA assumption $\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1$,
# the shared matrix is estimated as the *pooled* within-class scatter.
# A common and convenient approximation is to use the sample covariance of
# the **entire** dataset:
#
# $$
# \hat{\boldsymbol{\Sigma}}
# \approx \frac{1}{N-1}\sum_{i=1}^{N}
#   (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top,
# $$
#
# which coincides with the pooled estimate when the classes are balanced.
#
# ### Class prior
#
# The prior probability of class $C_0$ is estimated by the relative frequency:
#
# $$
# \hat{\pi}_0 = \frac{N_0}{N}, \qquad \hat{\pi}_1 = 1 - \hat{\pi}_0.
# $$
# 

# %%
# ── Class means (MLE) ─────────────────────────────────────────────────────────
mu0 = X0.mean(axis=0)    # shape (2,)
mu1 = X1.mean(axis=0)    # shape (2,)

print(f"μ₀ = [{mu0[0]:.3f}, {mu0[1]:.3f}]")
print(f"μ₁ = [{mu1[0]:.3f}, {mu1[1]:.3f}]")

# %%
# ── Shared covariance matrix (whole-dataset approximation) ────────────────────
Sigma = np.cov(X.T)      # shape (2, 2)

print("\nShared covariance matrix Σ:")
print(Sigma)

# %%
# ── Class prior ───────────────────────────────────────────────────────────────
pi0 = n0 / n             # P(C0)
pi1 = 1.0 - pi0          # P(C1)

print(f"\nPrior  π₀ = P(C₀) = {pi0:.4f}")
print(f"Prior  π₁ = P(C₁) = {pi1:.4f}")

# %% [markdown]
# ---
# ## 6 — Analytical Decision Boundary (LDA)
#
# For the shared-covariance case, the log-posterior ratio
#
# $$
# \ln \frac{p(C_0 \mid \mathbf{x})}{p(C_1 \mid \mathbf{x})}
# = \boldsymbol{\theta}^\top \tilde{\mathbf{x}}
# $$
#
# is linear in $\mathbf{x}$.  Setting it to zero gives the separating hyperplane.
# The closed-form coefficients are:
#
# $$
# \boldsymbol{\theta}_{1:d}
# = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1),
# $$
#
# $$
# \theta_0
# = -\tfrac{1}{2}\boldsymbol{\mu}_0^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0
#   +\tfrac{1}{2}\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1
#   +\ln\pi_0 - \ln\pi_1.
# $$
#
# The decision rule assigns a new point $\mathbf{x}$ to $C_0$ if
# $\boldsymbol{\theta}^\top \tilde{\mathbf{x}} > 0$, and to $C_1$ otherwise.
#

# %%
# ── Compute LDA decision-boundary coefficients ────────────────────────────────
Sigma_inv = np.linalg.inv(Sigma)          # Σ⁻¹,  shape (2, 2)

# Feature coefficients:  θ_{1:d} = Σ⁻¹ (μ₀ - μ₁)
theta_feat = Sigma_inv @ (mu0 - mu1)      # shape (2,)

# Bias term:  θ₀ = -½ μ₀ᵀ Σ⁻¹ μ₀ + ½ μ₁ᵀ Σ⁻¹ μ₁ + ln(π₀/π₁)
theta_bias = (
    -0.5 * mu0 @ Sigma_inv @ mu0
    +0.5 * mu1 @ Sigma_inv @ mu1
    + np.log(pi0) - np.log(pi1)
)

# Full parameter vector [θ₀, θ₁, θ₂]
theta = np.concatenate([[theta_bias], theta_feat])

print("LDA parameter vector θ = [θ₀, θ₁, θ₂]:")
print(f"  [{theta[0]:.4f},  {theta[1]:.4f},  {theta[2]:.4f}]")

# %% [markdown]
# ---
# ## 7 — Evaluation Grid and Class-Conditional Densities
#
# To visualise the distributions and decision boundary we build a dense
# $100 \times 100$ grid covering the range of the data, and evaluate
# $p(\mathbf{x} \mid C_k)$ at every grid point using the fitted Gaussian
# parameters.
# 

# %%
# ── 100×100 evaluation grid ───────────────────────────────────────────────────
u = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
v = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
U, V = np.meshgrid(u, v)           # U, V each of shape (100, 100)

# %%
# ── Class-conditional densities on the grid (shared Σ) ───────────────────────
# scipy.stats.multivariate_normal.pdf is already vectorised over a (M,d) array,
# so we stack the grid points into an (M, 2) matrix for efficiency.
grid_points = np.column_stack([U.ravel(), V.ravel()])   # shape (10000, 2)

p0_lda = st.multivariate_normal.pdf(grid_points, mu0, Sigma).reshape(U.shape)
p1_lda = st.multivariate_normal.pdf(grid_points, mu1, Sigma).reshape(U.shape)

# %% [markdown]
# ---
# ## 8 — Class-Conditional Densities $p(\mathbf{x} \mid C_k)$
#
# The heatmaps below show $p(\mathbf{x} \mid C_0)$ and $p(\mathbf{x} \mid C_1)$
# over the input space, together with their iso-probability contours.
# Under the shared-covariance assumption both distributions have identical
# contour *shapes* (ellipses with the same orientation and eccentricity),
# differing only in their centres $\boldsymbol{\mu}_k$.  This is the defining
# visual signature of LDA.
# 

# %%
# ── Helper: shared keyword arguments for heatmap figures ─────────────────────
_imshow_kw = dict(
    origin='lower',
    extent=(u.min(), u.max(), v.min(), v.max()),
    alpha=0.7,
)
_scatter_kw0 = dict(s=40, c=COLORS[0], edgecolors=COLORS[8], alpha=0.9)
_scatter_kw1 = dict(s=40, c=COLORS[1], edgecolors=COLORS[7], alpha=0.9)

def plot_density(density, mu_star, title):
    """Plot a class-conditional density heatmap with contours and data."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(density, **_imshow_kw)
    ax.contour(U, V, density, linewidths=0.7, colors=[COLORS[6]])
    ax.scatter(X0[:, 0], X0[:, 1], label='$C_0$', **_scatter_kw0)
    ax.scatter(X1[:, 0], X1[:, 1], label='$C_1$', **_scatter_kw1)
    ax.scatter(*mu_star, s=180, c=COLORS[11], marker='*',
               zorder=5, label='mean')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(u.min(), u.max())
    ax.set_ylim(v.min(), v.max())
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# %%
# ── Figure 2a – p(x | C0), shared Σ ──────────────────────────────────────────
plot_density(p0_lda, mu0, r'$p(\mathbf{x}\mid C_0)$ — shared $\Sigma$ (LDA)')

# %%
# ── Figure 2b – p(x | C1), shared Σ ──────────────────────────────────────────
plot_density(p1_lda, mu1, r'$p(\mathbf{x}\mid C_1)$ — shared $\Sigma$ (LDA)')

# %% [markdown]
# ---
# ## 9 — Posterior Probabilities and Decision Boundary (LDA)
#
# ### Bayes' theorem
#
# Given the class-conditional densities and the priors, the posterior
# probability of class $C_k$ at any point $\mathbf{x}$ follows from Bayes:
#
# $$
# p(C_k \mid \mathbf{x})
# = \frac{p(\mathbf{x} \mid C_k)\,\pi_k}
#        {p(\mathbf{x} \mid C_0)\,\pi_0 + p(\mathbf{x} \mid C_1)\,\pi_1}.
# $$
#
# ### Two decision boundaries
#
# We visualise two related boundaries:
#
# **Likelihood-ratio boundary** (dashed): ignores priors, assigns to the class
# whose likelihood is higher:
# $$
# \frac{p(\mathbf{x} \mid C_0)}{p(\mathbf{x} \mid C_1)} = 1.
# $$
#
# **Posterior-ratio boundary** (solid): the full Bayesian boundary that
# accounts for class frequencies:
# $$
# \frac{p(\mathbf{x} \mid C_0)\,\pi_0}{p(\mathbf{x} \mid C_1)\,\pi_1} = 1.
# $$
#
# When the classes are balanced ($\pi_0 = \pi_1$) the two boundaries coincide.
# When one class is more frequent its boundary shifts *towards* the rarer class,
# reflecting the cost of missing it.
# 

# %%
# ── Posterior and ratio quantities on the grid ────────────────────────────────
evidence    = p0_lda * pi0 + p1_lda * pi1     # p(x), shape (100, 100)

pp0_lda = p0_lda * pi0 / evidence             # p(C0 | x)
pp1_lda = p1_lda * pi1 / evidence             # p(C1 | x)

ratio_likelihood = p0_lda / p1_lda            # p(x|C0) / p(x|C1)  — no priors
ratio_posterior  = p0_lda * pi0 / (p1_lda * pi1)  # full posterior ratio

# %%
# ── Helper: posterior heatmap with both decision boundaries ───────────────────
def plot_posterior(posterior, title, boundary_color):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(posterior, **_imshow_kw)
    ax.scatter(X0[:, 0], X0[:, 1], label='$C_0$', **_scatter_kw0)
    ax.scatter(X1[:, 0], X1[:, 1], label='$C_1$', **_scatter_kw1)
    # Bayesian boundary (solid): posterior ratio = 1
    cs1 = ax.contour(U, V, ratio_posterior,  levels=[1.0],
                     linewidths=1.5, colors=[boundary_color])
    # Likelihood-only boundary (dashed): likelihood ratio = 1
    cs2 = ax.contour(U, V, ratio_likelihood, levels=[1.0],
                     linewidths=1.5, linestyles='dashed', colors=[boundary_color])
    # Manual legend entries for the two boundaries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=boundary_color, lw=1.5,
               label='Bayesian boundary'),
        Line2D([0], [0], color=boundary_color, lw=1.5, linestyle='dashed',
               label='Likelihood-only boundary'),
    ]
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(u.min(), u.max())
    ax.set_ylim(v.min(), v.max())
    ax.set_title(title, fontsize=12)
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

# %%
# ── Figure 3a – p(C0 | x), LDA ───────────────────────────────────────────────
plot_posterior(pp0_lda, r'$p(C_0\mid\mathbf{x})$ — LDA', COLORS[6])

# %%
# ── Figure 3b – p(C1 | x), LDA ───────────────────────────────────────────────
plot_posterior(pp1_lda, r'$p(C_1\mid\mathbf{x})$ — LDA', COLORS[7])

# %% [markdown]
# ---
# ## 10 — Classification and Accuracy (LDA)
#
# The LDA decision rule classifies $\mathbf{x}$ as $C_0$ when the posterior
# ratio exceeds 1:
#
# $$
# \hat{t}(\mathbf{x}) =
# \begin{cases}
#   0 & \text{if } p(C_0 \mid \mathbf{x}) > p(C_1 \mid \mathbf{x}), \\
#   1 & \text{otherwise.}
# \end{cases}
# $$
#
# We evaluate this rule on the **training set itself** to obtain a lower-bound
# estimate of the model's discriminative power (no train/test split is applied
# here; for a fair evaluation one should use cross-validation).
# ---

# %%
# ── Evaluate LDA on the training set ─────────────────────────────────────────
# Compute p(x|Ck) at each training point using the shared-Σ Gaussians
p0_train = st.multivariate_normal.pdf(X, mu0, Sigma)
p1_train = st.multivariate_normal.pdf(X, mu1, Sigma)

# Posterior ratio at each training point
ratio_train = p0_train * pi0 / (p1_train * pi1)

# Predict C0 where ratio > 1, else C1
pred_lda = np.where(ratio_train >= 1.0, 0, 1)

# Accuracy
acc_lda = np.mean(pred_lda == t)
n_errors_lda = (pred_lda != t).sum()

print(f"LDA  |  Training accuracy : {acc_lda:.4f}  ({n_errors_lda} misclassified / {n})")

# %% [markdown]
# ---
# ## 11 — Quadratic Discriminant Analysis (QDA): Class-Specific Covariances
#
# ### Motivation
#
# The shared-covariance assumption is convenient but may be too restrictive when
# the two classes genuinely have different spreads or orientations.  Relaxing it
# — i.e. estimating $\boldsymbol{\Sigma}_0$ and $\boldsymbol{\Sigma}_1$
# separately from the class-specific data — yields **Quadratic Discriminant
# Analysis (QDA)**.
#
# ### Decision boundary
#
# Because the quadratic terms in the Gaussian log-likelihood no longer cancel,
# the log-posterior ratio becomes a **quadratic form** in $\mathbf{x}$:
#
# $$
# \ln \frac{p(C_0 \mid \mathbf{x})}{p(C_1 \mid \mathbf{x})}
# = -\tfrac{1}{2}\ln\frac{|\boldsymbol{\Sigma}_0|}{|\boldsymbol{\Sigma}_1|}
#   -\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_0^{-1}(\mathbf{x}-\boldsymbol{\mu}_0)
#   +\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_1^{-1}(\mathbf{x}-\boldsymbol{\mu}_1)
#   +\ln\frac{\pi_0}{\pi_1}.
# $$
#
# Setting this to zero gives a *conic section* (ellipse, hyperbola, or
# parabola) as the decision boundary.  The additional flexibility allows QDA
# to capture differences in class spread and orientation that LDA cannot
# represent, at the cost of estimating $\mathcal{O}(d^2)$ additional
# parameters and higher variance when training data are scarce.
# 

# %%
# ── Class-specific covariance matrices (QDA) ─────────────────────────────────
Sigma0 = np.cov(X0.T)    # shape (2, 2), estimated from C0 data only
Sigma1 = np.cov(X1.T)    # shape (2, 2), estimated from C1 data only

print("Σ₀ (class C0):")
print(Sigma0)
print("\nΣ₁ (class C1):")
print(Sigma1)

# %%
# ── Class-conditional densities on the grid (QDA) ────────────────────────────
p0_qda = st.multivariate_normal.pdf(grid_points, mu0, Sigma0).reshape(U.shape)
p1_qda = st.multivariate_normal.pdf(grid_points, mu1, Sigma1).reshape(U.shape)

# %% [markdown]
# ---
# ## 12 — Class-Conditional Densities: QDA vs. LDA
#
# With class-specific covariances the two Gaussian density functions now have
# **different ellipse shapes**: the orientation, eccentricity, and size of the
# contours can differ between $C_0$ and $C_1$.  Comparing Figures 4a/4b with
# the earlier Figures 2a/2b shows how much additional flexibility QDA provides.
# 

# %%
# ── Figure 4a – p(x | C0), QDA ───────────────────────────────────────────────
plot_density(p0_qda, mu0, r'$p(\mathbf{x}\mid C_0)$ — class-specific $\Sigma_0$ (QDA)')

# %%
# ── Figure 4b – p(x | C1), QDA ───────────────────────────────────────────────
plot_density(p1_qda, mu1, r'$p(\mathbf{x}\mid C_1)$ — class-specific $\Sigma_1$ (QDA)')

# %% [markdown]
# ---
# ## 13 — Posterior and Decision Boundary: QDA
#
# The QDA posterior and its decision boundary (the quadratic iso-contour at
# posterior ratio = 1) are computed and displayed below.  Observe how the
# boundary is now a *curved* line rather than the straight separator of LDA.
# 

# %%
# ── Posterior quantities on the grid (QDA) ────────────────────────────────────
evidence_qda = p0_qda * pi0 + p1_qda * pi1

pp0_qda = p0_qda * pi0 / evidence_qda
pp1_qda = p1_qda * pi1 / evidence_qda

ratio_likelihood_qda = p0_qda / p1_qda
ratio_posterior_qda  = p0_qda * pi0 / (p1_qda * pi1)

# %%
# ── Helper: QDA posterior plot (reuses the same function, different ratios) ───
def plot_posterior_qda(posterior, title, boundary_color):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(posterior, **_imshow_kw)
    ax.scatter(X0[:, 0], X0[:, 1], label='$C_0$', **_scatter_kw0)
    ax.scatter(X1[:, 0], X1[:, 1], label='$C_1$', **_scatter_kw1)
    ax.contour(U, V, ratio_posterior_qda,  levels=[1.0],
               linewidths=1.5, colors=[boundary_color])
    ax.contour(U, V, ratio_likelihood_qda, levels=[1.0],
               linewidths=1.5, linestyles='dashed', colors=[boundary_color])
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=boundary_color, lw=1.5,
               label='Bayesian boundary (QDA)'),
        Line2D([0], [0], color=boundary_color, lw=1.5, linestyle='dashed',
               label='Likelihood-only boundary (QDA)'),
    ]
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_xlim(u.min(), u.max())
    ax.set_ylim(v.min(), v.max())
    ax.set_title(title, fontsize=12)
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

# %%
# ── Figure 5a – p(C0 | x), QDA ───────────────────────────────────────────────
plot_posterior_qda(pp0_qda, r'$p(C_0\mid\mathbf{x})$ — QDA', COLORS[6])

# %%
# ── Figure 5b – p(C1 | x), QDA ───────────────────────────────────────────────
plot_posterior_qda(pp1_qda, r'$p(C_1\mid\mathbf{x})$ — QDA', COLORS[7])

# %% [markdown]
# ---
# ## 14 — Summary: LDA vs. QDA
#
# | | **LDA** | **QDA** |
# |---|---|---|
# | Covariance assumption | $\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}$ | $\boldsymbol{\Sigma}_0 \neq \boldsymbol{\Sigma}_1$ |
# | Decision boundary | Linear (hyperplane) | Quadratic (conic section) |
# | Parameters to estimate | $d + d(d+1)/2$ | $2d + d(d+1)$ |
# | Bias–variance trade-off | Lower variance, higher bias | Higher variance, lower bias |
# | Preferred when | Classes have similar spread; small $N$ | Classes have different spread; large $N$ |
#
# **Key takeaways:**
#
# 1. **Generative models give closed-form boundaries.** Unlike logistic
#    regression (which requires iterative optimisation), GDA derives the
#    decision boundary analytically from the estimated Gaussian parameters.
#
# 2. **Priors shift the boundary.** The dashed (likelihood-only) and solid
#    (Bayesian) boundaries differ whenever $\pi_0 \neq \pi_1$.  Using the
#    correct prior is especially important for imbalanced datasets.
#
# 3. **Shared $\boldsymbol{\Sigma}$ implies a linear boundary.** This is a
#    direct algebraic consequence of the quadratic terms cancelling in the
#    log-ratio — not an additional modelling assumption.
#
# 4. **LDA ≡ linear logistic regression** (asymptotically). When the Gaussian
#    assumption holds, LDA achieves the same decision boundary as logistic
#    regression but uses the data more efficiently by also modelling the
#    marginal $p(\mathbf{x})$.
#
# 5. **QDA is more flexible but needs more data.** Each class covariance has
#    $d(d+1)/2$ free parameters; with $d=2$ that is 3 extra parameters per
#    class — manageable here, but costly in high dimensions.
# 
# %%
