# 
# %% [markdown]
# ---
# # Support Vector Machines Applied to a XOR-Structured Dataset
#
# ## Introduction
#
# A **Support Vector Machine (SVM)** is a supervised learning algorithm that seeks
# the optimal separating hyperplane between two classes by maximising the *margin*,
# i.e. the distance between the hyperplane and the nearest training points
# (the **support vectors**).

#%% [markdown]
# ## The XOR Problem and Non-Linear Separability
#
# The **XOR function** is the canonical example of a problem that is *not linearly
# separable*: no straight line (or hyperplane in higher dimensions) can correctly
# separate the two classes, because the class label depends on the *interaction*
# between the two features:
#
# $$
# y = x_1 \oplus x_2 =
# \begin{cases}
#   1 & \text{if } \operatorname{sign}(x_1) \neq \operatorname{sign}(x_2) \\
#   0 & \text{otherwise}
# \end{cases}
# $$
#
# In this notebook we generate a continuous, noisy version of XOR by drawing
# $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_2)$ and assigning labels
# via $y = \mathbf{1}[x_1 > 0] \oplus \mathbf{1}[x_2 > 0]$.

#%% [markdown]
# ## The Kernel Trick
#
# To handle non-linear boundaries, SVMs use the **kernel trick**: instead of
# working in the original feature space $\mathcal{X}$, the data are implicitly
# mapped to a (possibly infinite-dimensional) Hilbert space $\mathcal{H}$ via
# $\phi: \mathcal{X} \to \mathcal{H}$. The kernel function evaluates inner
# products in $\mathcal{H}$ without computing $\phi$ explicitly:
#
# $$
# k(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_{\mathcal{H}}.
# $$
#
# Common choices include:
#
# | Kernel | Expression |
# |--------|-----------|
# | Linear | $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$ |
# | Polynomial | $k(\mathbf{x}, \mathbf{x}') = (\gamma\,\mathbf{x}^\top \mathbf{x}' + r)^d$ |
# | RBF (Gaussian) | $k(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2\right)$ |
# | Sigmoid | $k(\mathbf{x}, \mathbf{x}') = \tanh(\gamma\,\mathbf{x}^\top \mathbf{x}' + r)$ |
#
# For the XOR problem the **RBF kernel** is particularly effective: its Gaussian
# shape naturally captures the quadrant-based structure of XOR.  A large $\gamma$
# value makes the kernel very localised (each training point influences only a
# small neighbourhood), enabling the model to carve out the complex, non-convex
# decision boundary that XOR requires.

# %%
# ── Suppress non-critical warnings (e.g. matplotlib deprecations) ─────────────
import warnings
warnings.filterwarnings('ignore')

# %%
# ── Core numerical and modelling libraries ────────────────────────────────────
import numpy as np
from sklearn import svm                                   # kernel SVM (exact solver)
from sklearn.kernel_approximation import RBFSampler       # Nyström-like RBF feature map
from sklearn.linear_model import SGDClassifier            # linear SGD with hinge loss

# %%
# ── Plotting configuration ────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

plt.style.use("ggplot")           # clean, grid-based aesthetic
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.serif":          "Ubuntu",
    "font.monospace":      "Ubuntu Mono",
    "font.size":           10,
    "axes.labelsize":      10,
    "axes.labelweight":    "bold",
    "axes.titlesize":      10,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     10,
    "figure.titlesize":    12,
    "image.cmap":          "jet",
    "image.interpolation": "none",
    "figure.figsize":      (16, 8),
    "lines.linewidth":     2,
    "lines.markersize":    8,
})

# Distinct, accessible colours for class labels (xkcd palette)
COLORS = [
    "xkcd:pale orange", "xkcd:sea blue",    "xkcd:pale red",
    "xkcd:sage green",  "xkcd:terra cotta", "xkcd:dull purple",
    "xkcd:teal",        "xkcd:goldenrod",   "xkcd:cadet blue",
    "xkcd:scarlet",
]

# Custom colormap: upper half of Spectral avoids the harsh red-blue extremes
cmap_big = cm.get_cmap('Spectral', 512)
cmap = mcolors.ListedColormap(cmap_big(np.linspace(0.5, 1, 128)))

# %%
# ── Dataset generation ────────────────────────────────────────────────────────
# Dense evaluation grid for decision-surface visualisation
xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))

np.random.seed(0)

# 300 points drawn from a standard 2-D Gaussian
X = np.random.randn(300, 2)

# XOR label: class 1 when x1 and x2 have *opposite* signs, class 0 otherwise.
# Equivalent to the logical XOR of the two sign-bit indicators.
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# %%
# ── Figure 1 – Raw XOR dataset ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('white')

for cls in range(2):
    idx = np.where(Y == cls)
    ax.scatter(X[idx, 0], X[idx, 1],
               c=COLORS[cls], s=40, edgecolors='k',
               alpha=0.9, label=f'Class {cls}', cmap=cmap)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.legend()
ax.set_title('XOR dataset – raw scatter plot')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Part 1 – Kernel SVM (RBF) via Exact QP Solver
#
# ### Model
#
# We fit a kernel SVM using the **Radial Basis Function (RBF) kernel**:
#
# $$
# k(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right),
# \quad \gamma = 20.
# $$
#
# A large $\gamma$ produces a very *tight* kernel: each support vector influences
# only a small region around itself.  This is appropriate for XOR, where the
# decision boundary is highly non-convex and locally defined by the quadrant
# boundaries $x_1 = 0$ and $x_2 = 0$.
#
# ### Decision Function
#
# After fitting, the signed distance of a new point $\mathbf{x}$ from the
# separating hyperplane is:
#
# $$
# f(\mathbf{x}) = \sum_{i \in \mathcal{SV}} \alpha_i\, y_i\, k(\mathbf{x}_i, \mathbf{x}) + b,
# $$
#
# where $\alpha_i \ge 0$ are the Lagrange multipliers (non-zero only for support
# vectors) and $b$ is the bias term.  The predicted class is
# $\hat{y} = \operatorname{sign}(f(\mathbf{x}))$.
#
# The colour map in Figure 2 encodes the *magnitude* of $f$: deeper colours
# indicate higher confidence; the zero-level contour is the decision boundary.
# 

# %%
# ── Fit kernel SVM with RBF kernel ───────────────────────────────────────────
# gamma=20  → very localised kernel, captures the fine XOR quadrant structure.
# Alternative kernels (commented out) are left for comparison:
#   svm.SVC(kernel='linear')                   – cannot separate XOR
#   svm.SVC(kernel='poly', degree=2, coef0=1)  – degree-2 poly may partially work
#   svm.SVC(kernel='sigmoid', gamma=15)        – sigmoid is not a Mercer kernel
clf_rbf = svm.SVC(gamma=20)
clf_rbf.fit(X, Y)

# %%
# ── Figure 2 – RBF-SVM decision surface ──────────────────────────────────────
# Evaluate the signed distance f(x) on every point of the dense grid
Z = clf_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('white')

# Background colour encodes decision-function value (confidence)
ax.imshow(Z,
          interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          aspect='auto', origin='lower', alpha=0.5, cmap=cmap)

# Zero-level contour = decision boundary
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors=[COLORS[6]])

# Training points coloured by true label
for cls in range(2):
    idx = np.where(Y == cls)
    ax.scatter(X[idx, 0], X[idx, 1],
               c=COLORS[cls], edgecolors='k', s=40,
               label=f'Class {cls}', cmap=cmap)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.legend()
ax.set_title(r'RBF-SVM decision surface ($\gamma = 20$)')
plt.tight_layout()
plt.show()

# %%
# ── Training accuracy – kernel SVM ───────────────────────────────────────────
acc_rbf = np.mean(Y == clf_rbf.predict(X)) * 100
print(f'RBF-SVM  |  Training accuracy: {acc_rbf:.5f}%')

#%% [markdown]
# ---
# ## Part 2 – Approximate Kernel SVM via Random Features + Hinge Loss SGD
#
# ### Motivation
#
# The exact QP solver used in Part 1 has time complexity $\mathcal{O}(N^2)$–
# $\mathcal{O}(N^3)$ in the number of training points and stores the full kernel
# matrix.  For large datasets this is infeasible.
#
# **Bochner's theorem** guarantees that any shift-invariant, positive-definite
# kernel can be approximated as:
#
# $$
# k(\mathbf{x}, \mathbf{x}') \approx \phi(\mathbf{x})^\top \phi(\mathbf{x}'),
# $$
#
# where $\phi: \mathbb{R}^d \to \mathbb{R}^D$ is an explicit, *finite-dimensional*
# feature map constructed via **random Fourier features** (Rahimi & Recht, 2007).
#
# For the RBF kernel the random feature map is:
#
# $$
# \phi(\mathbf{x}) = \sqrt{\frac{2}{D}}
# \begin{bmatrix}
#   \cos(\boldsymbol{\omega}_1^\top \mathbf{x} + b_1) \\
#   \vdots \\
#   \cos(\boldsymbol{\omega}_D^\top \mathbf{x} + b_D)
# \end{bmatrix},
# \quad
# \boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0},\, 2\gamma\,\mathbf{I}),
# \quad
# b_j \sim \mathcal{U}[0, 2\pi].
# $$
#
# Scikit-learn's `RBFSampler` implements this approximation.
#
# ### Hinge Loss and SGD
#
# Once we have explicit features $\phi(\mathbf{x})$ we can train a *linear*
# classifier using **Stochastic Gradient Descent (SGD)** minimising the
# regularised **hinge loss**:
#
# $$
# \mathcal{L}(\mathbf{w}) =
# \frac{\lambda}{2}\|\mathbf{w}\|^2
# + \frac{1}{N}\sum_{i=1}^{N} \max\!\bigl(0,\; 1 - y_i\,\mathbf{w}^\top \phi(\mathbf{x}_i)\bigr).
# $$
# The first term is the $\ell_2$ regulariser (weight decay, controlled by
# `alpha`); the second is the average hinge loss.  SGD processes one (or a
# mini-batch of) sample(s) per step, making the overall pipeline scalable to
# very large datasets at the cost of a slight approximation in the kernel.
# 

# %%
# ── Random Fourier Features (RBF approximation) ───────────────────────────────
# gamma=10 in the RBFSampler should match the kernel bandwidth used earlier.
# n_components (default 100) controls approximation quality vs. speed.
rbf_feature = RBFSampler(gamma=10, random_state=1)
X_features = rbf_feature.fit_transform(X)   # shape: (300, n_components)

# %%
# ── Linear SGD classifier with hinge loss ─────────────────────────────────────
# penalty='l2', alpha=0.001  →  soft-margin SVM in the lifted feature space.
# max_iter=1000 ensures convergence of the SGD loop.
clf_sgd = SGDClassifier(max_iter=1000, penalty='l2', alpha=0.001)
clf_sgd.fit(X_features, Y)

# %%
# ── Training accuracy – SGD + random features ─────────────────────────────────
acc_sgd = np.mean(Y == clf_sgd.predict(X_features)) * 100
print(f'SGD+RFF  |  Training accuracy: {acc_sgd:.5f}%')

# %%
# ── Figure 3 – Approximate-kernel decision surface ────────────────────────────
# Project the dense grid into the random feature space and predict class labels
Z_sgd = clf_sgd.predict(
    rbf_feature.transform(np.c_[xx.ravel(), yy.ravel()])
).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('white')

# Binary predicted-label map (0 / 1) used as background image
ax.imshow(Z_sgd,
          interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          aspect='auto', origin='lower', alpha=0.5, cmap=cmap)

# Decision boundary (transition between label 0 and label 1 on the grid)
ax.contour(xx, yy, Z_sgd, levels=[0.5], linewidths=2, colors=[COLORS[6]])

# Training points coloured by true label
for cls in range(2):
    idx = np.where(Y == cls)
    ax.scatter(X[idx, 0], X[idx, 1],
               c=COLORS[cls], edgecolors='k', s=40,
               label=f'Class {cls}', cmap=cmap)

ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.legend()
ax.set_title('Approximate RBF-SVM – SGD + Random Fourier Features')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary and Takeaways
#
# | Method | Kernel | Complexity | Training Acc. |
# |--------|--------|-----------|--------------|
# | Exact SVM (`svm.SVC`) | RBF, $\gamma=20$ | $\mathcal{O}(N^2)$–$\mathcal{O}(N^3)$ | ~100% |
# | SGD + RFF (`SGDClassifier`) | RBF approx., $\gamma=10$ | $\mathcal{O}(N \cdot D)$ | high |
#
# **Key observations:**
#
# 1. **Linear classifiers cannot solve XOR.** A plain linear SVM would achieve
#    ~50% accuracy — no better than random guessing — because the XOR
#    boundary is not a hyperplane.
#
# 2. **The RBF kernel solves XOR exactly** (on the training set) by implicitly
#    lifting the data to an infinite-dimensional space where the four XOR
#    quadrants become linearly separable.  The high $\gamma$ value ($=20$)
#    makes the classifier memorise local structure.
#
# 3. **Random Fourier Features offer a scalable approximation.** By
#    constructing an explicit $D$-dimensional feature map $\phi(\mathbf{x})$,
#    the problem reduces to ordinary linear SGD — linear in both $N$ and $D$.
#    The approximation quality improves with $D$ and converges to the exact
#    kernel in the limit $D \to \infty$.
#
# 4. **The hinge loss** $\ell_{\text{hinge}}(t) = \max(0, 1-t)$ is a convex
#    upper bound on the 0-1 loss.  Combined with $\ell_2$ regularisation it
#    recovers the soft-margin SVM objective in the primal form, enabling
#    efficient gradient-based optimisation.
# 

ß
