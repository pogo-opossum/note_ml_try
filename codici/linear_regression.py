# %% [markdown]
# # Linear Regression
#
# This notebook illustrates the main approaches to **regression on scalar
# noisy data**, progressing from the simplest deterministic formulation
# (Empirical Risk Minimisation) through probabilistic methods (MLE and
# Bayesian inference) to non-parametric kernel-based techniques.
#
# The unifying theme is the **linear-in-parameters** model:
# predictions are always a linear combination of (possibly non-linear)
# basis functions applied to the inputs. This single framework accommodates
# a surprisingly rich variety of function classes while retaining the
# elegance and tractability of linear algebra.
#
# ## Topics covered
#
# | # | Topic | Key idea |
# |---|-------|----------|
# | 1 | **Basis functions** | Transform inputs to capture non-linearities |
# | 2 | **Reference dataset** | Two synthetic 1-D regression problems |
# | 3 | **ERM** | Least-squares closed-form solution |
# | 4 | **MLE** | Probabilistic re-reading of ERM; noise estimation |
# | 5 | **Bayesian approach** | Posterior over parameters; predictive distribution |
# | 6 | **Marginal likelihood** | Model selection via evidence; Empirical Bayes |
# | 7 | **Equivalent kernel** | Bridge between parametric and non-parametric views |
# | 8 | **Kernel regression** | Nadaraya-Watson estimator; bandwidth selection |
# | 9 | **LOESS** | Locally weighted linear regression |

# %% [markdown]
# ## Imports and plot configuration

# %%
import random                                   # for random index sampling
from functools import partial                   # for partial function application

import matplotlib.colors as mcolors             # custom colourmap construction
import matplotlib.pyplot as plt                 # plotting
import numpy as np                              # numerical computing
from scipy import stats                         # multivariate Gaussian pdf
from scipy.spatial.distance import cdist        # pairwise distance matrices
from sklearn.linear_model import BayesianRidge  # reference Bayesian ridge implementation

# ── Plot style ────────────────────────────────────────────────────────────────
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

# Custom colormap: a restricted slice of Spectral (avoids extreme hues)
_cmap_big  = plt.colormaps["Spectral"].resampled(512)           # full Spectral, 512 colours
CMAP       = mcolors.ListedColormap(_cmap_big(np.linspace(0.7, 0.95, 256)))  # warm sub-range
BBOX_PROPS = dict(boxstyle="round,pad=0.3", fc=COLORS[0], alpha=0.5)         # text-box style


# %% [markdown]
# ## Plotting utilities
#
# All recurring visualisation patterns are encapsulated in three helpers so
# that the scientific sections below remain focused on modelling logic.
#
# | Function | Purpose |
# |----------|---------|
# | `plot_regression` | All-purpose regression plot (scatter, curves, uncertainty band) |
# | `plot_bivariate_gaussian` | 2-D Gaussian density as a colour image |
# | `plot_basis_functions` | Family of basis functions over a shared domain |

# %%
def plot_regression(
    x, y_true,
    X_train=None, t_train=None,
    y_pred=None,
    y_std=None,
    X_highlight=None, t_highlight=None,
    w_samples=None, expand_fn=None,
    title="",
    ax=None,
):
    """
    Generic regression plot.

    Draws any combination of:
      - scatter plot of training observations
      - ground-truth curve
      - predicted mean curve
      - uncertainty band  (±1 std around the predicted mean)
      - posterior parameter samples (requires `expand_fn`)
      - a highlighted subset of training points

    Parameters
    ----------
    x            : array (n_plot, 1), dense grid for smooth curves.
    y_true       : array (n_plot, 1), noiseless ground-truth values on `x`.
    X_train      : array (n, 1) or None, training inputs.
    t_train      : array (n, 1) or None, training targets.
    y_pred       : array (n_plot, 1) or None, predicted mean.
    y_std        : float or array (n_plot, 1) or None, std of predictions.
    X_highlight  : array or None, a subset of training points to emphasise.
    t_highlight  : array or None, targets for the highlighted subset.
    w_samples    : array (S, m) or None, parameter samples to plot as curves.
    expand_fn    : callable or None, design-matrix builder used with w_samples.
    title        : str, axis title.
    ax           : matplotlib Axes or None; if None, uses the current axes.
    """
    if ax is None:
        ax = plt.gca()                          # fall back to current active axes

    # Scatter-plot the full training set if provided
    if X_train is not None and t_train is not None:
        ax.scatter(X_train, t_train, s=20, color=COLORS[1], label="Observations")

    # Overlay a highlighted subset (e.g. the examples seen so far in sequential updates)
    if X_highlight is not None and t_highlight is not None:
        ax.scatter(X_highlight, t_highlight, s=50, color=COLORS[4],
                   label="Selected examples")

    # Noiseless ground-truth curve (dashed for visual distinction)
    ax.plot(x, y_true, lw=2, color=COLORS[2], linestyle="dashed",
            label="True function")

    # Draw one curve per sampled parameter vector (visualises posterior uncertainty)
    if w_samples is not None and expand_fn is not None:
        for w in w_samples:
            ax.plot(x, expand_fn(x) @ w.reshape(-1, 1), lw=1, alpha=0.7)

    # Predicted mean curve
    if y_pred is not None:
        ax.plot(x, y_pred, lw=2, color=COLORS[3], label="Predicted mean")

    # ±1σ uncertainty band around the predicted mean
    if y_pred is not None and y_std is not None:
        y_pred_r = y_pred.ravel()
        y_std_r  = np.asarray(y_std).ravel() if not np.isscalar(y_std) else y_std
        ax.fill_between(
            x.ravel(),
            y_pred_r - y_std_r,           # lower bound: mean − 1σ
            y_pred_r + y_std_r,           # upper bound: mean + 1σ
            color=COLORS[8], alpha=0.2, label="Uncertainty ±1σ"
        )

    ax.legend(fontsize=8)
    ax.set_title(title)


def plot_bivariate_gaussian(mean, cov, resolution=100, ax=None):
    """
    Visualise the density of a bivariate Gaussian as a colour image.

    Evaluates the pdf on a regular grid over [−1, 1]² and renders it with
    `imshow`. Useful for inspecting the prior and posterior over a 2-D
    parameter vector (w₀, w₁).

    Parameters
    ----------
    mean       : array (2,), distribution mean.
    cov        : array (2, 2), covariance matrix.
    resolution : int, number of grid points per axis.
    ax         : matplotlib Axes or None.
    """
    if ax is None:
        ax = plt.gca()

    grid      = np.linspace(-1, 1, resolution)                # 1-D axis values
    grid_flat = np.dstack(np.meshgrid(grid, grid)).reshape(-1, 2)  # all (w0, w1) pairs
    # Evaluate the 2-D Gaussian pdf at every grid point and reshape to image
    densities = stats.multivariate_normal.pdf(
        grid_flat, mean=mean.ravel(), cov=cov
    ).reshape(resolution, resolution)

    ax.imshow(densities, origin="lower", extent=(-1, 1, -1, 1))   # colour image of the pdf
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("$w_0$")
    ax.set_ylabel("$w_1$")
    ax.grid(False)                                                   # remove grid lines from image


def plot_basis_functions(x, bf, params_list, labels=None, title=""):
    """
    Plot a family of basis functions over a common domain.

    Parameters
    ----------
    x           : array (n,), evaluation domain.
    bf          : callable, basis-function factory (e.g. gaussian_basis_function).
    params_list : list of dict, keyword arguments for each basis function instance.
    labels      : list of str or None, legend labels.
    title       : str, figure title.
    """
    plt.figure(figsize=(12, 6))
    for i, kw in enumerate(params_list):
        y   = partial(bf, **kw)(x)                              # evaluate the i-th basis function
        lbl = labels[i] if labels else str(kw)                  # label from list or repr of params
        plt.plot(x, y, lw=1, color=COLORS[4 + i % len(COLORS)], label=lbl)
    plt.legend()
    plt.title(title)
    plt.show()


# %% [markdown]
# ---
# ## 1. Linear Regression and Basis Functions
#
# ### The linear model
#
# Given $n$ observations $(\mathbf{x}_1, t_1), \ldots, (\mathbf{x}_n, t_n)$
# with $\mathbf{x} \in \mathbb{R}^d$ and $t \in \mathbb{R}$, we want to learn
# a predictive function that, given a new input $\mathbf{x}$, returns an
# estimate of the unknown target $t$.
#
# The most general **linear-in-parameters** model is:
#
# $$
# h(\mathbf{x}, \mathbf{w})
#   = \sum_{j=0}^{m-1} w_j \, \phi_j(\mathbf{x})
#   = \boldsymbol{\phi}(\mathbf{x}) \, \mathbf{w}
# $$
#
# where $\boldsymbol{\phi}(\mathbf{x}) = (\phi_0(\mathbf{x}), \ldots,
# \phi_{m-1}(\mathbf{x}))$ is the **basis-function vector** (also called
# the *feature map*) and $\mathbf{w} \in \mathbb{R}^m$ is the
# **parameter vector**.
# By convention $\phi_0(\mathbf{x}) = 1$, so that $w_0$ acts as a bias
# (intercept) term.
#
# The model is **linear in the parameters** $\mathbf{w}$ but can be
# **non-linear in the features** $\mathbf{x}$ through the choice of $\phi_j$.
# This allows approximation of arbitrarily complex relationships while
# retaining all the mathematical convenience of linear algebra (closed-form
# solutions, convexity guarantees, conjugate priors).
#
# ### Basis-function families
#
# | Type | Formula | Locality |
# |------|---------|---------|
# | Identity | $\phi_j(x) = x_j$ | global |
# | Polynomial | $\phi_j(x) = x^j$ | global |
# | Gaussian | $\phi_j(x) = \exp\!\left(-\dfrac{(x-\mu_j)^2}{2s^2}\right)$ | local |
# | Sigmoidal | $\phi_j(x) = \sigma\!\left(\dfrac{x-\mu_j}{s}\right) = \dfrac{1}{1+e^{-(x-\mu_j)/s}}$ | local |
# | Hyperbolic tangent | $\phi_j(x) = \tanh\!\left(\dfrac{x-\mu_j}{s}\right) = 2\sigma\!\left(\dfrac{x-\mu_j}{s}\right)-1$ | local |
#
# **Local** basis functions (Gaussian, sigmoidal, tanh) are nearly constant
# far from their centre $\mu_j$, so each one "responds" only to a portion of
# the input axis. **Global** functions (polynomial) influence the model over
# the entire domain; a change in a single coefficient affects every prediction.
#
# ### Design matrix
#
# Given a training set $\mathbf{X} \in \mathbb{R}^{n \times d}$, the
# **design matrix** $\boldsymbol{\Phi} \in \mathbb{R}^{n \times m}$ stacks
# all feature-space representations:
#
# $$
# \boldsymbol{\Phi} =
# \begin{pmatrix}
# \phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \cdots & \phi_{m-1}(\mathbf{x}_1) \\
# \vdots               & \vdots               & \ddots & \vdots \\
# \phi_0(\mathbf{x}_n) & \phi_1(\mathbf{x}_n) & \cdots & \phi_{m-1}(\mathbf{x}_n)
# \end{pmatrix}
# $$
#
# With this notation the model writes compactly as
# $\hat{\mathbf{t}} = \boldsymbol{\Phi}\,\mathbf{w}$.
# Choosing different basis functions amounts to choosing different
# representations of the input, i.e. different geometries of the feature space
# in which the regression is linear.

# %%
def identity_basis_function(x):
    """Identity basis function: φ(x) = x."""
    return x


def polynomial_basis_function(x, power=2):
    """Polynomial basis function: φ(x) = x^power."""
    return x ** power


def gaussian_basis_function(x, mu=1.0, sigma=0.1):
    """
    Gaussian (local) basis function: φ(x) = exp(−(x−μ)² / (2σ²)).

    Each function is localised around its centre μ with width σ.
    """
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def sigmoid_basis_function(x, mu=1.0, sigma=0.1):
    """
    Sigmoidal (local) basis function: φ(x) = 1 / (1 + exp(−(x−μ)/σ)).

    Produces an S-shaped response centred at μ; σ controls the transition width.
    """
    return 1.0 / (1.0 + np.exp(-((x - mu) / sigma)))


def tanh_basis_function(x, mu=1.0, sigma=0.1):
    """
    Hyperbolic-tangent basis function: φ(x) = 2·sigmoid((x−μ)/σ) − 1.

    Equivalent to a rescaled sigmoid with output range (−1, 1).
    """
    return 2.0 / (1.0 + np.exp(-((x - mu) / sigma))) - 1.0


def expand(x, bf=None, bf_args_list=None):
    """
    Build the design matrix Φ by applying basis functions to the inputs.

    If `bf` is None, the augmented identity basis [1, x] is used, which
    corresponds to simple affine (bias + slope) regression.
    Otherwise `bf` is applied once per entry in `bf_args_list` and the
    results are stacked as columns to form Φ.

    Parameters
    ----------
    x            : array (n, 1) or scalar.
    bf           : callable or None, basis function to apply.
    bf_args_list : list of dict or None, keyword arguments for each column.

    Returns
    -------
    Phi : array (n, m), design matrix.
    """
    if np.isscalar(x):
        x = np.full((1, 1), x)                 # promote scalar to (1,1) array

    if bf is not None:
        if bf_args_list:
            # Apply bf with each set of parameters and stack results as columns
            cols = [partial(bf, **kw)(x) for kw in bf_args_list]
            ll   = np.array(cols).squeeze().T
            if ll.ndim == 1:
                ll = ll.reshape(-1, 1)          # keep 2-D shape for single basis function
        else:
            ll = np.array(bf(x))
    else:
        # Default: augmented identity [1, x] — supports bias + linear term
        ll = np.c_[np.ones(x.shape[0]), x].squeeze()

    if x.shape[0] == 1 or ll.ndim == 1:
        return ll.reshape(1, -1)                # ensure output is always 2-D
    return ll


# %% [markdown]
# ### Visual comparison of basis-function families
#
# The plot below shows one representative from each family evaluated over
# $[-1, 1]$. Observe:
#
# - The **Gaussian** concentrates its response near $\mu = 0$ and decays
#   rapidly — useful as a local "bump detector".
# - The **polynomial** (degree 3) is entirely global: it grows without bound
#   outside the training domain, which can cause extrapolation issues.
# - The **sigmoid** transitions smoothly from 0 to 1 around its centre,
#   encoding a soft indicator of whether $x > \mu$.
# - The **hyperbolic tangent** is a centred, zero-mean version of the sigmoid,
#   transitioning from $-1$ to $+1$.

# %%
x_plot = np.linspace(-1, 1, 1000)      # dense domain for smooth curves

plt.figure(figsize=(12, 6))
# Gaussian centred at 0 with σ = 0.2
plt.plot(x_plot, partial(gaussian_basis_function,  mu=0, sigma=0.2)(x_plot),
         lw=1, label=r"Gaussian ($\mu=0, \sigma=0.2$)")
# Cubic polynomial
plt.plot(x_plot, partial(polynomial_basis_function, power=3)(x_plot),
         lw=1, label=r"Polynomial (degree 3)")
# Sigmoid centred at 0 with σ = 0.1
plt.plot(x_plot, partial(sigmoid_basis_function,    mu=0, sigma=0.1)(x_plot),
         lw=1, label=r"Sigmoidal ($\mu=0, \sigma=0.1$)")
# Tanh centred at 0 with σ = 0.1
plt.plot(x_plot, partial(tanh_basis_function,       mu=0, sigma=0.1)(x_plot),
         lw=1, label=r"Hyperbolic tangent ($\mu=0, \sigma=0.1$)")
plt.legend()
plt.title("Examples of basis functions")
plt.show()


# %% [markdown]
# ---
# ## 2. Reference Dataset
#
# All experiments in this notebook use scalar inputs $x_i \in [-1, 1]$
# collected in $\mathbf{X} \in \mathbb{R}^{n \times 1}$.
# Targets are generated from two functions with qualitatively different
# structure:
#
# - **Linear function** $f$:
#   $$f(x) = w_0 + w_1 x + \varepsilon, \qquad
#     w_0 = -0.3,\; w_1 = 0.5$$
#   A simple affine relationship — perfectly captured by an identity basis.
#
# - **Sinusoidal function** $g$:
#   $$g(x) = 0.5 + \sin(2\pi x) + \varepsilon$$
#   A periodic, non-linear relationship that requires basis functions beyond
#   the identity to be well approximated.
#
# In both cases the noise is additive Gaussian:
# $\varepsilon \sim \mathcal{N}(0, \beta^{-1})$,
# where $\beta$ is the **precision** (inverse variance). A large $\beta$
# implies a high signal-to-noise ratio. Here we set $\beta = 250$, so the
# noise standard deviation is $\sigma_\varepsilon = 1/\sqrt{250} \approx 0.063$.
#
# Having two datasets with very different structures lets us compare learning
# strategies in both an easy case (linear $f$) and a harder case (non-linear $g$).

# %%
F_W0, F_W1 = -0.3, 0.5     # true parameters of the linear function f


def _noise(size, variance):
    """Sample Gaussian noise N(0, variance) with the given shape."""
    return np.random.normal(scale=np.sqrt(variance), size=size)


def f(X, noise_variance):
    """Linear function with noise: f(x) = F_W0 + F_W1·x + ε."""
    return F_W0 + F_W1 * X + _noise(X.shape, noise_variance)


def g(X, noise_variance):
    """Sinusoidal function with noise: g(x) = 0.5 + sin(2πx) + ε."""
    return 0.5 + np.sin(2 * np.pi * X) + _noise(X.shape, noise_variance)


# %%
# ── Global dataset parameters ─────────────────────────────────────────────────
N    = 50          # number of training samples drawn uniformly from [−1, 1]
BETA = 250.0       # noise precision (β = 1/σ²); high value → low noise

# Dense grid for evaluating noiseless (ground-truth) curves
x  = np.linspace(-1, 1, 1000).reshape(-1, 1)   # (1000, 1) column vector
y1 = f(x, noise_variance=0)                     # noiseless linear curve
y2 = g(x, noise_variance=0)                     # noiseless sinusoidal curve

# Training samples with Gaussian noise (noise_variance = 1/β)
X  = np.random.rand(N, 1) * 2 - 1              # N points uniform in [−1, 1]
t1 = f(X, noise_variance=1 / BETA)             # noisy targets for linear f
t2 = g(X, noise_variance=1 / BETA)             # noisy targets for sinusoidal g


# %% [markdown]
# ### Overview of the two reference datasets
#
# The scatter plots below show the $n = 50$ noisy observations alongside the
# noiseless ground-truth curves. The low noise level ($\beta = 250$)
# makes both curves clearly visible through the scatter.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
plt.scatter(X, t1, s=20, color=COLORS[1], label="Observations")    # noisy training points
plt.plot(x, y1, lw=1, color=COLORS[3], label="True function f")    # ground truth
plt.legend()
plt.title("Dataset – Linear function $f$")

plt.sca(axes[1])
plt.scatter(X, t2, s=20, color=COLORS[1], label="Observations")    # noisy training points
plt.plot(x, y2, lw=1, color=COLORS[3], label="True function g")    # ground truth
plt.legend()
plt.title("Dataset – Sinusoidal function $g$")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Basis functions applied to the dataset
#
# For the sinusoidal dataset $g$ we will use non-linear basis functions
# placed at $k$ equally-spaced centres $\mu_j \in [-1, 1]$.
#
# The three panels below show how each basis family transforms the training
# points (small dots on the curve of each basis function at the $x$-values
# of the training set). Key observations:
#
# - **Gaussian** functions with $k = 5$ give a smooth, localised encoding of
#   the input; points far from a centre contribute very little to that column
#   of the design matrix.
# - **Polynomial** functions are global and grow rapidly outside $[-1, 1]$,
#   which can cause numerical issues and poor extrapolation with high degrees.
# - **Sigmoidal** functions provide a cumulative encoding: the $j$-th column
#   equals approximately 1 for inputs above $\mu_j$ and 0 below.

# %%
k   = 5                              # number of basis functions
mus = np.linspace(-1, 1, k)          # equally-spaced centres

for bf_cls, bf_params, title in [
    (gaussian_basis_function,   [{"mu": mu, "sigma": 0.1} for mu in mus],
     "Gaussian basis functions ($k=5$)"),
    (polynomial_basis_function, [{"power": p} for p in range(k)],
     "Polynomial basis functions (degrees 0–4)"),
    (sigmoid_basis_function,    [{"mu": mu, "sigma": 0.1} for mu in mus],
     "Sigmoidal basis functions ($k=5$)"),
]:
    plt.figure(figsize=(12, 6))
    # Mark training inputs on the x-axis (targets not shown; focus is on the features)
    plt.scatter(X, np.zeros_like(X), s=20, color=COLORS[1])
    for i, kw in enumerate(bf_params):
        bf_i = partial(bf_cls, **kw)                            # instantiate i-th basis function
        lbl  = str(kw)
        plt.plot(x, bf_i(x), lw=1, color=COLORS[4 + i % len(COLORS)], label=lbl)
        plt.scatter(X, bf_i(X), s=20, color=COLORS[4 + i % len(COLORS)])  # projected training points
    plt.title(title)
    plt.show()


# %% [markdown]
# ---
# ## 3. Empirical Risk Minimisation (ERM)
#
# ### Quadratic empirical risk
#
# Given a hypothesis class of linear models
# $h(\mathbf{x}, \mathbf{w}) = \boldsymbol{\phi}(\mathbf{x})\,\mathbf{w}$,
# the **quadratic empirical risk** measures average squared deviation from
# the observed targets:
#
# $$
# \overline{\mathcal{R}}(\mathbf{w};\,\boldsymbol{\Phi},\mathbf{t})
#   = \frac{1}{n} \|\boldsymbol{\Phi}\,\mathbf{w} - \mathbf{t}\|^2
# $$
#
# This objective is **strictly convex** in $\mathbf{w}$ (the Hessian
# $\frac{2}{n}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$ is positive
# semi-definite and positive definite when $\boldsymbol{\Phi}$ has full
# column rank). Setting the gradient to zero yields a **unique global
# minimum**:
#
# $$
# \hat{\mathbf{w}} = \underset{\mathbf{w}}{\arg\min}\;
#   \overline{\mathcal{R}}(\mathbf{w})
#   = \underbrace{\left(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\right)^{-1}
#     \boldsymbol{\Phi}^\top}_{\boldsymbol{\Phi}^+} \mathbf{t}
# $$
#
# This is the **Ordinary Least-Squares (OLS)** solution. The matrix
# $\boldsymbol{\Phi}^+$ is the **Moore–Penrose pseudo-inverse** of
# $\boldsymbol{\Phi}$; the product $\boldsymbol{\Phi}\,\boldsymbol{\Phi}^+$
# is the orthogonal projector onto the column space of $\boldsymbol{\Phi}$.
#
# > **Existence and uniqueness**: The solution is unique if and only if
# > $\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$ is invertible, i.e. the columns
# > of $\boldsymbol{\Phi}$ are linearly independent. When the number of
# > parameters exceeds the number of observations ($m > n$), or when the
# > features are multicollinear, the system is underdetermined and
# > regularisation is required.

# %%
def erm(X, t, args):
    """
    Find the parameters that minimise the quadratic empirical risk.

    Implements the ordinary least-squares solution:
        ŵ = (ΦᵀΦ)⁻¹ Φᵀ t

    Parameters
    ----------
    X    : array (n, 1), training inputs.
    t    : array (n, 1), training targets.
    args : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    w_hat : array (m, 1), optimal parameter vector.
    """
    Phi = expand(X, **args)                                 # build design matrix
    return np.linalg.inv(Phi.T @ Phi) @ Phi.T @ t          # normal equations


def predict(x, w, args):
    """
    Compute predictions of the linear model: ŷ = Φ(x) w.

    Parameters
    ----------
    x    : array or scalar, test points.
    w    : array (m, 1), model parameters.
    args : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    y : array or scalar, predicted values.
    """
    y = expand(x, **args) @ w                               # design matrix times parameters
    return y.item() if np.isscalar(x) else y               # scalar output for scalar input


# %% [markdown]
# ### ERM fit on both datasets
#
# For the linear dataset $f$ we use the default identity (augmented) basis
# $\boldsymbol{\phi}(x) = [1, x]$, which perfectly matches the true function
# class. For the sinusoidal dataset $g$ we switch to $K = 10$ sigmoidal
# basis functions equally spaced over $[-1, 1]$, giving the model sufficient
# expressiveness to approximate the oscillations.

# %%
# ── Linear dataset: identity basis ───────────────────────────────────────────
args_linear = {}                            # empty dict → augmented identity basis
w_erm       = erm(X, t1, args_linear)       # solve normal equations

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
plot_regression(x, y1, X, t1,
                y_pred=predict(x, w_erm, args_linear),
                title="ERM – Linear function $f$")

# ── Sinusoidal dataset: sigmoidal basis ───────────────────────────────────────
K   = 10                                    # number of sigmoidal basis functions
mus = np.linspace(-1, 1, K)                 # centre positions
args_sigmoid = {
    "bf":           sigmoid_basis_function,
    "bf_args_list": [{"mu": mu} for mu in mus],
}
w_erm_sin = erm(X, t2, args_sigmoid)        # solve normal equations with sigmoidal basis

plt.sca(axes[1])
plot_regression(x, y2, X, t2,
                y_pred=predict(x, w_erm_sin, args_sigmoid),
                title="ERM – Sinusoidal function $g$ (sigmoidal basis)")

plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 4. Maximum Likelihood Estimation (MLE)
#
# ### Probabilistic model
#
# MLE provides a probabilistic reinterpretation of ERM. We model each
# observation $t_i$ as the deterministic prediction $h(\mathbf{x}_i, \mathbf{w})$
# corrupted by independent Gaussian noise:
#
# $$
# t_i = h(\mathbf{x}_i, \mathbf{w}) + \varepsilon_i,
# \qquad \varepsilon_i \sim \mathcal{N}(0, \beta^{-1})
# $$
#
# This is equivalent to saying that the conditional distribution of $t$ given
# $\mathbf{x}$ is:
#
# $$
# p(t \mid \mathbf{x}, \mathbf{w}, \beta)
#   = \mathcal{N}\!\left(t \mid h(\mathbf{x}, \mathbf{w}),\, \beta^{-1}\right)
# $$
#
# ### Likelihood and log-likelihood
#
# Assuming i.i.d. observations, the **likelihood** of the full training set is:
#
# $$
# p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \beta)
#   = \prod_{i=1}^{n} \mathcal{N}\!\left(t_i \mid
#     \boldsymbol{\phi}(\mathbf{x}_i)\mathbf{w},\, \beta^{-1}\right)
# $$
#
# Taking the logarithm (which is monotone) turns the product into a sum and
# yields the **log-likelihood**:
#
# $$
# \log p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \beta)
#   = \frac{n}{2}\log\beta - \frac{n}{2}\log(2\pi)
#     - \beta\, E_D(\mathbf{w}),
# \quad
# E_D(\mathbf{w}) = \frac{1}{2}\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}\|^2
# $$
#
# Maximising over $\mathbf{w}$ (the only term that depends on $\mathbf{w}$ is
# $-\beta E_D$) is equivalent to **minimising $E_D$** — i.e. solving the same
# normal equations as ERM. The MLE parameter estimate is therefore:
#
# $$
# \hat{\mathbf{w}}_{\mathrm{ML}}
#   = \left(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\right)^{-1}
#     \boldsymbol{\Phi}^\top \mathbf{t}
# $$
#
# ### Noise variance estimation
#
# Unlike ERM, the probabilistic framework also allows estimating the noise
# precision $\beta$ (or equivalently the noise variance $\sigma^2 = \beta^{-1}$)
# by maximising the log-likelihood w.r.t. $\beta$:
#
# $$
# \hat{\sigma}^2 = \hat{\beta}^{-1}
#   = \frac{1}{n}\|\mathbf{t} - \boldsymbol{\Phi}\hat{\mathbf{w}}\|^2
# $$
#
# This is the **mean squared residual** and provides the ±σ̂ uncertainty band
# shown in the plots below.
#
# > **Limitation of MLE**: in expressive models (e.g. many basis functions
# > relative to the number of observations) the likelihood can be maximised
# > by fitting the noise, resulting in severe **overfitting**. The Bayesian
# > approach in the next section addresses this via regularisation.

# %%
def mle(X, t, args):
    """
    Maximum Likelihood Estimation of the parameter vector w.

    Algebraically identical to the ERM solution; included here for
    conceptual clarity and to emphasise the probabilistic interpretation.

        ŵ_ML = (ΦᵀΦ)⁻¹ Φᵀ t
    """
    Phi = expand(X, **args)                                 # build design matrix
    return np.linalg.inv(Phi.T @ Phi) @ Phi.T @ t          # normal equations (same as ERM)


def std_mle(X, t, w, args):
    """
    MLE estimate of the noise standard deviation.

        σ̂ = sqrt( (1/n) ‖t − Φw‖² )

    This is the root mean squared residual evaluated at the MLE parameters.

    Returns
    -------
    sigma_hat : float, estimated standard deviation.
    """
    Phi = expand(X, **args)                                 # build design matrix
    return np.sqrt(np.mean((t - Phi @ w) ** 2))            # RMS residual


# %% [markdown]
# ### MLE fit and estimated uncertainty bands
#
# The ±σ̂ band represents the estimated noise level around the predictive
# mean. Note that MLE provides a **single global** uncertainty estimate (a
# constant band), unlike the Bayesian approach which produces
# position-dependent uncertainty.

# %%
# ── Linear dataset: identity basis ───────────────────────────────────────────
args_linear = {}                            # augmented identity basis
w_ML        = mle(X, t1, args_linear)       # MLE parameters
std_ML      = std_mle(X, t1, w_ML, args_linear)  # estimated noise std
pred        = predict(x, w_ML, args_linear) # predictive mean on dense grid

# ── Sinusoidal dataset: Gaussian basis ───────────────────────────────────────
K   = 10                                    # 10 Gaussian basis functions
mus = np.linspace(-1, 1, K)                 # equally-spaced centres
args_gaussian = {
    "bf":           gaussian_basis_function,
    "bf_args_list": [{"mu": mu} for mu in mus],
}
w_ML_sin   = mle(X, t2, args_gaussian)      # MLE parameters
std_ML_sin = std_mle(X, t2, w_ML_sin, args_gaussian)  # estimated noise std
pred_sin   = predict(x, w_ML_sin, args_gaussian)       # predictive mean

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
plot_regression(x, y1, X, t1,
                y_pred=pred, y_std=std_ML,
                title="MLE – Linear function $f$")

plt.sca(axes[1])
plot_regression(x, y2, X, t2,
                y_pred=pred_sin, y_std=std_ML_sin,
                title="MLE – Sinusoidal function $g$ (Gaussian basis)")

plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 5. Bayesian Approach
#
# ### Prior distribution over parameters
#
# In the Bayesian approach the parameters $\mathbf{w}$ are treated as random
# variables. Before observing any data, our beliefs are encoded in a **prior
# distribution**. We adopt a zero-mean isotropic Gaussian:
#
# $$
# p(\mathbf{w} \mid \alpha)
#   = \mathcal{N}(\mathbf{w} \mid \mathbf{0},\, \alpha^{-1}\mathbf{I})
# $$
#
# The hyperparameter $\alpha > 0$ is the **prior precision** (inverse
# variance): a large $\alpha$ encodes the belief that parameters should be
# close to zero, acting as an implicit regulariser. A small $\alpha$
# corresponds to a vague, diffuse prior that lets the data speak more freely.
#
# ### Posterior distribution
#
# After observing the training set $(\mathbf{X}, \mathbf{t})$, Bayes' rule
# updates the prior to the **posterior**. Because the Gaussian prior is
# *conjugate* to the Gaussian likelihood, the posterior is also Gaussian:
#
# $$
# p(\mathbf{w} \mid \mathbf{X}, \mathbf{t}, \alpha, \beta)
#   = \mathcal{N}(\mathbf{w} \mid \mathbf{m}_p,\, \boldsymbol{\Sigma}_p)
# $$
#
# with:
#
# $$
# \boldsymbol{\Sigma}_p
#   = \left(\alpha\mathbf{I} + \beta\,\boldsymbol{\Phi}^\top\boldsymbol{\Phi}
#     \right)^{-1}, \qquad
# \mathbf{m}_p = \beta\,\boldsymbol{\Sigma}_p\,\boldsymbol{\Phi}^\top\mathbf{t}
# $$
#
# **Interpretation**: the posterior precision $\boldsymbol{\Sigma}_p^{-1}$
# is the sum of the prior precision $\alpha\mathbf{I}$ and the data-driven
# precision $\beta\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$. As $n \to \infty$
# the data term dominates and the posterior concentrates around the MLE
# solution; when $n = 0$ the posterior reduces to the prior.

# %%
def posterior_params(X, t, alpha, beta, args):
    """
    Compute the mean and covariance of the Gaussian posterior over w.

    Closed-form solution (Bishop, PRML, eq. 3.50–3.51):
        Σ_p = (αI + β ΦᵀΦ)⁻¹
        m_p = β Σ_p Φᵀ t

    When X is empty (n=0), returns the prior: Σ_p = α⁻¹ I, m_p = 0.

    Parameters
    ----------
    X, t          : training inputs and targets.
    alpha, beta   : prior precision and noise precision hyperparameters.
    args          : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    m_p     : array (m, 1), posterior mean (= MAP estimate).
    Sigma_p : array (m, m), posterior covariance.
    """
    if X.shape[0] == 0:
        # No data: return the prior parameters
        m   = expand(np.zeros((1, X.shape[1])), **args).shape[1]
        Sigma_p = (1 / alpha) * np.eye(m)      # prior covariance
        m_p     = np.zeros((m, 1))             # prior mean is zero
        return m_p, Sigma_p
    Phi         = expand(X, **args)            # design matrix (n × m)
    m           = Phi.shape[1]                 # number of basis functions
    # Posterior precision matrix (inverse covariance)
    Sigma_p_inv = alpha * np.eye(m) + beta * Phi.T @ Phi
    Sigma_p     = np.linalg.inv(Sigma_p_inv)  # posterior covariance
    m_p         = beta * Sigma_p @ Phi.T @ t  # posterior mean
    return m_p, Sigma_p


# %% [markdown]
# ### Prior distribution over parameters
#
# With $\alpha = 10$ the prior assigns roughly 95% of its mass within
# $\pm 2/\sqrt{10} \approx \pm 0.63$ of the origin — a moderate shrinkage
# towards zero. The colour image below shows the joint density
# $p(w_0, w_1 \mid \alpha)$; the symmetric, circular shape reflects the
# isotropic (equal-variance, uncorrelated) nature of the prior.

# %%
ALPHA = 10.0                                    # prior precision hyperparameter

cov_prior  = (1 / ALPHA) * np.eye(2, dtype=int)    # 2×2 prior covariance matrix
mean_prior = np.zeros(2)                            # prior mean is zero

plt.figure(figsize=(6, 6))
plot_bivariate_gaussian(mean_prior, cov_prior, resolution=200)
plt.title(r"Prior distribution $p(\mathbf{w}\mid\alpha)$")
plt.show()


# %% [markdown]
# ### Sequential Bayesian update — posterior over $(w_0, w_1)$
#
# Each panel below corresponds to the posterior after incorporating $n$
# training examples. The posterior sharpens and shifts towards the true
# parameter values $(-0.3, 0.5)$ as more data are included. The sequence
# illustrates the **Bayesian learning** process: the prior is gradually
# over-ridden by the likelihood, and the uncertainty (spread of the
# distribution) decreases monotonically with $n$.

# %%
N_MAX_EXAMPLES = 20                             # total examples used in the sequence

idx          = random.sample(range(X.shape[0]), N_MAX_EXAMPLES)  # random indices
X_sub, t_sub = X[idx], t1[idx]                 # subset for sequential update

n_examples   = (0, 1, 2, 3, 10, 20)            # observation counts to visualise
means, covs  = [], []                           # store posterior parameters
for n_ex in n_examples:
    # Compute posterior using only the first n_ex examples
    m_p, Sigma_p = posterior_params(X_sub[:n_ex], t_sub[:n_ex], ALPHA, BETA, args_linear)
    means.append(m_p.ravel())
    covs.append(Sigma_p)

fig, axes = plt.subplots(2, len(n_examples) // 2, figsize=(18, 6*len(n_examples) // 2))
for i, n_ex in enumerate(n_examples):
    plt.sca(axes[i // 3, i % 3])
    plot_bivariate_gaussian(means[i], covs[i], resolution=200)
    plt.title(r"Posterior over $(w_0, w_1)$, $n = %d$" % n_ex)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Parameter samples from the posterior — linear dataset
#
# Sampling from the posterior $p(\mathbf{w} \mid \mathbf{X}, \mathbf{t})$
# generates an ensemble of plausible parameter vectors. Each sample defines
# a regression line; plotting multiple samples visualises the **functional
# uncertainty** induced by parameter uncertainty. As more data arrive, the
# sampled lines concentrate around the true function.

# %%
n_examples  = (0, 1, 2, 3, 10, 20)     # observation counts to visualise
N_SAMPLES   = 7                         # number of posterior samples per panel

idx = random.sample(range(X.shape[0]), n_examples[-1])  # 20 random training indices
X_sub, t1_sub = X[idx], t1[idx]        # corresponding inputs and targets

m_p_list       = []     # list of posterior means
w_samples_list = []     # list of posterior sample arrays
Sigma_p_list   = []     # list of posterior covariances

for n_ex in n_examples:
    # Posterior given the first n_ex examples
    m_p, Sigma_p = posterior_params(X_sub[:n_ex], t1_sub[:n_ex], ALPHA, BETA, args_linear)
    # Draw N_SAMPLES vectors from the posterior
    w_samples    = np.random.multivariate_normal(m_p.ravel(), Sigma_p, N_SAMPLES)
    m_p_list.append(m_p)
    w_samples_list.append(w_samples)
    Sigma_p_list.append(Sigma_p)

fig, axes = plt.subplots(len(n_examples), 2, figsize=(18, 6 * len(n_examples)))
for i, n_ex in enumerate(n_examples):
    # Left column: 2-D posterior density
    plt.sca(axes[i, 0])
    axes[i, 0].grid(False)
    plot_bivariate_gaussian(m_p_list[i], Sigma_p_list[i], resolution=200)
    plt.title(r"Posterior over $(w_0, w_1)$, $n = %d$" % n_ex)

    # Right column: sampled regression lines
    plt.sca(axes[i, 1])
    plot_regression(x, y1, X, t1,
        X_highlight=X_sub[:n_ex], t_highlight=t1_sub[:n_ex],
        w_samples=w_samples_list[i],
        expand_fn=lambda z: expand(z, **args_linear),
        title="Parameter samples from the posterior ($n = %d$)" % n_ex)

plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 5.1 MAP Estimate and Regularisation
#
# The **Maximum A Posteriori (MAP)** estimate is the mode of the posterior.
# For a Gaussian posterior the mode coincides with its mean $\mathbf{m}_p$,
# so $\hat{\mathbf{w}}_{\mathrm{MAP}} = \mathbf{m}_p$.
#
# Maximising the log posterior is equivalent to minimising:
#
# $$
# \beta\, E_D(\mathbf{w}) + \alpha\, E_W(\mathbf{w}),
# \quad \text{with}\quad
# E_W(\mathbf{w}) = \frac{1}{2}\|\mathbf{w}\|^2
# $$
#
# The term $E_W$ is an **$\ell_2$ regulariser** (*weight decay*) that
# emerges naturally from the Gaussian prior. It prevents overfitting by
# penalising large parameter magnitudes. The effective regularisation
# coefficient is $\lambda = \alpha / \beta$: stronger prior ($\alpha \uparrow$)
# or weaker noise ($\beta \downarrow$) both increase regularisation.
#
# This shows that **Ridge regression** (OLS with $\ell_2$ penalty) is
# mathematically equivalent to MAP estimation under a Gaussian prior.

# %%
def map_estimate(X, t, args):
    """
    Maximum A Posteriori (MAP) estimate of the parameter vector w.

    Since the posterior is Gaussian, the MAP estimate equals the posterior
    mean m_p. Uses the global ALPHA and BETA hyperparameters.
    """
    return posterior_params(X, t, alpha=ALPHA, beta=BETA, args=args)[0]


def std_map(X, t, w, args):
    """
    MAP estimate of the noise standard deviation.

    Identical to the MLE noise estimate because the MAP solution is
    the posterior mean (the Gaussian likelihood is symmetric around its mode).

        σ̂ = sqrt( (1/n) ‖t − Φw‖² )

    Returns
    -------
    sigma_hat : float, estimated standard deviation.
    """
    return std_mle(X, t, w, args)     # delegates to the MLE noise estimator


# %% [markdown]
# ### MAP sequential update — linear dataset
#
# The left column shows posterior parameter samples; the right column shows
# the MAP predictive mean with ±σ̂ uncertainty band. As the number of
# observations grows, the MAP solution converges towards the MLE solution
# and the band narrows.

# %%
n_examples  = (0, 1, 2, 3, 10, 20)     # observation counts to visualise

idx = random.sample(range(X.shape[0]), n_examples[-1])
X_sub, t1_sub = X[idx], t1[idx]        # 20 random training examples

m_p_list, w_samples_list = [], []
pred_map_list, std_map_list = [], []

for n_ex in n_examples:
    # Compute posterior for the first n_ex examples
    m_p, Sigma_p = posterior_params(X_sub[:n_ex], t1_sub[:n_ex], ALPHA, BETA, args_linear)
    w_samples    = np.random.multivariate_normal(m_p.ravel(), Sigma_p, N_SAMPLES)
    pred_map     = predict(x, m_p, args_linear)             # MAP predictive mean
    std_MAP      = std_map(X_sub[:n_ex], t1_sub[:n_ex], m_p, args_linear)

    m_p_list.append(m_p)
    w_samples_list.append(w_samples)
    pred_map_list.append(pred_map)
    std_map_list.append(std_MAP)

fig, axes = plt.subplots(len(n_examples), 2, figsize=(18, 6 * len(n_examples)))
for i, n_ex in enumerate(n_examples):
    # Left: posterior parameter samples as regression lines
    plt.sca(axes[i, 0])
    plot_regression(x, y1, X, t1,
        X_highlight=X_sub[:n_ex], t_highlight=t_sub[:n_ex],
        w_samples=w_samples_list[i],
        expand_fn=lambda z: expand(z, **args_linear),
        title="Parameter samples – $n = %d$" % n_ex)

    # Right: MAP mean with uncertainty band
    plt.sca(axes[i, 1])
    plot_regression(x, y1, X, t1,
        X_highlight=X_sub[:n_ex], t_highlight=t_sub[:n_ex],
        y_pred=pred_map_list[i], y_std=std_map_list[i],
        title="MAP predictive distribution – $n = %d$" % n_ex)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### MAP sequential update — sinusoidal dataset
#
# The same analysis repeated on the sinusoidal dataset $g$ using $K = 10$
# Gaussian basis functions. With a non-linear basis the MAP solution
# can represent oscillatory patterns; the uncertainty is larger wherever
# training examples are sparse.

# %%
K   = 10                                    # number of Gaussian basis functions
mus = np.linspace(-1, 1, K)                 # equally-spaced centres
args_gaussian = {
    "bf":           gaussian_basis_function,
    "bf_args_list": [{"mu": mu} for mu in mus],
}
n_examples = (0, 1, 2, 3, 10, 20)

idx = random.sample(range(X.shape[0]), n_examples[-1])
X_sub, t2_sub = X[idx], t2[idx]            # subset from the sinusoidal dataset

m_p_list, w_samples_list = [], []
pred_map_list, std_map_list = [], []

for n_ex in n_examples:
    m_p, Sigma_p = posterior_params(X_sub[:n_ex], t2_sub[:n_ex], ALPHA, BETA, args_gaussian)
    w_samples    = np.random.multivariate_normal(m_p.ravel(), Sigma_p, N_SAMPLES)
    pred_map     = predict(x, m_p, args_gaussian)
    std_MAP      = std_map(X_sub[:n_ex], t2_sub[:n_ex], m_p, args_gaussian)

    m_p_list.append(m_p)
    w_samples_list.append(w_samples)
    pred_map_list.append(pred_map)
    std_map_list.append(std_MAP)

fig, axes = plt.subplots(len(n_examples), 2, figsize=(18, 6 * len(n_examples)))
for i, n_ex in enumerate(n_examples):
    # Left: sampled regression curves
    plt.sca(axes[i, 0])
    plot_regression(x, y2, X, t2,
        X_highlight=X_sub[:n_ex], t_highlight=t_sub[:n_ex],
        w_samples=w_samples_list[i],
        expand_fn=lambda z: expand(z, **args_gaussian),
        title="Parameter samples – $n = %d$" % n_ex)

    # Right: MAP mean ± noise std
    plt.sca(axes[i, 1])
    plot_regression(x, y2, X, t2,
        X_highlight=X_sub[:n_ex], t_highlight=t2_sub[:n_ex],
        y_pred=pred_map_list[i], y_std=std_map_list[i],
        title="MAP predictive distribution – $n = %d$" % n_ex)

plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 5.2 Fully Bayesian Prediction
#
# Rather than committing to a single parameter vector (as in MAP), **fully
# Bayesian inference** marginalises $\mathbf{w}$ out of the predictive
# distribution:
#
# $$
# p(t \mid \mathbf{x}, \mathbf{X}, \mathbf{t}, \alpha, \beta)
#   = \int p(t \mid \mathbf{x}, \mathbf{w}, \beta)\,
#     p(\mathbf{w} \mid \mathbf{X}, \mathbf{t}, \alpha, \beta)\,d\mathbf{w}
#   = \mathcal{N}\!\left(t \mid \boldsymbol{\phi}(\mathbf{x})\mathbf{m}_p,\;
#     \sigma_n^2(\mathbf{x})\right)
# $$
#
# The **predictive variance** decomposes into two terms:
#
# $$
# \sigma_n^2(\mathbf{x})
#   = \underbrace{\frac{1}{\beta}}_{\text{aleatoric: irreducible noise}}
#   + \underbrace{\boldsymbol{\phi}(\mathbf{x})\,\boldsymbol{\Sigma}_p\,
#     \boldsymbol{\phi}(\mathbf{x})^\top}_{\text{epistemic: parameter uncertainty}}
# $$
#
# - The **aleatoric** term $1/\beta$ is irreducible — it reflects the
#   inherent noise in the data-generating process. Collecting more data
#   does not reduce it.
# - The **epistemic** term $\boldsymbol{\phi}(\mathbf{x})\boldsymbol{\Sigma}_p
#   \boldsymbol{\phi}(\mathbf{x})^\top$ captures our uncertainty about the
#   parameter values. It is large in regions of the input space far from
#   training data and shrinks towards zero as $n \to \infty$.
#
# This decomposition is one of the key advantages of the Bayesian approach
# over MLE: the predictive uncertainty is **input-dependent** and correctly
# inflates in regions where the model lacks training examples.

# %%
def posterior_predictive(X_test, m_p, Sigma_p, beta, args):
    """
    Compute the mean and variance of the Bayesian predictive distribution.

    For each test point x:
        E[t|x]   = φ(x) m_p                      (predictive mean)
        Var[t|x] = 1/β + φ(x) Σ_p φ(x)ᵀ          (predictive variance)

    Parameters
    ----------
    X_test      : array (n_test, d), test inputs.
    m_p         : array (m, 1), posterior mean.
    Sigma_p     : array (m, m), posterior covariance.
    beta        : float, noise precision.
    args        : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    y     : array (n_test, 1), predictive mean.
    y_var : array (n_test, 1), predictive variance (aleatoric + epistemic).
    """
    Phi   = expand(X_test, **args)                         # (n_test, m) design matrix
    y     = Phi @ m_p                                      # predictive mean
    y_var = (1.0 / beta + np.diag(Phi @ Sigma_p @ Phi.T)).reshape(-1, 1)  # predictive variance
    return y, y_var


# %% [markdown]
# ### Bayesian predictive distribution — linear dataset
#
# With only $N = 3$ training examples the posterior is still diffuse and
# the predictive band (±1 predictive std) is wide. The left panel shows
# sampled regression lines from the posterior; the right panel shows the
# predictive mean with the input-dependent uncertainty band — note that
# uncertainty grows near the edges of the domain where data are sparser.

# %%
N_EXAMPLES = 3                              # number of training examples to condition on

idx           = random.sample(range(X.shape[0]), N_EXAMPLES)
X_sub, t1_sub = X[idx], t1[idx]            # small subset from the linear dataset

# Compute posterior and predictive distribution
m_p, Sigma_p         = posterior_params(X_sub, t1_sub, ALPHA, BETA, args_linear)
w_samples            = np.random.multivariate_normal(m_p.ravel(), Sigma_p, N_SAMPLES)
pred_bay, sigma2_bay = posterior_predictive(x, m_p, Sigma_p, BETA, args_linear)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(axes[0])
plot_regression(x, y1, X, t1,
                X_highlight=X_sub, t_highlight=t1_sub,
                w_samples=w_samples,
                expand_fn=lambda z: expand(z, **args_linear),
                title="Parameter samples from the posterior (linear $f$)")

plt.sca(axes[1])
plot_regression(x, y1, X, t1,
                X_highlight=X_sub, t_highlight=t1_sub,
                y_pred=pred_bay, y_std=np.sqrt(sigma2_bay),  # std = sqrt(var)
                title="Bayesian predictive distribution (linear $f$)")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Bayesian predictive distribution — sinusoidal dataset
#
# On the sinusoidal dataset $g$ with $K = 10$ Gaussian basis functions,
# the predictive uncertainty is especially pronounced in regions between
# the training points, where the model has little support from the data.
# As more data are collected, the band narrows and the mean converges to
# the true sinusoidal curve.

# %%
N_EXAMPLES = 3                              # number of training examples to condition on

idx          = random.sample(range(X.shape[0]), N_EXAMPLES)
X_sub, t_sub = X[idx], t2[idx]             # small subset from the sinusoidal dataset

K   = 10
mus = np.linspace(-1, 1, K)
args_gaussian = {
    "bf":           gaussian_basis_function,
    "bf_args_list": [{"mu": mu} for mu in mus],
}

# Compute posterior and predictive distribution
m_p, Sigma_p         = posterior_params(X_sub, t_sub, ALPHA, BETA, args_gaussian)
w_samples            = np.random.multivariate_normal(m_p.ravel(), Sigma_p, N_SAMPLES)
pred_bay, sigma2_bay = posterior_predictive(x, m_p, Sigma_p, BETA, args_gaussian)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(axes[0])
plot_regression(x, y2, X, t2,
                X_highlight=X_sub, t_highlight=t_sub,
                w_samples=w_samples,
                expand_fn=lambda z: expand(z, **args_gaussian),
                title="Parameter samples from the posterior (sinusoidal $g$)")

plt.sca(axes[1])
plot_regression(x, y2, X, t2,
                X_highlight=X_sub, t_highlight=t_sub,
                y_pred=pred_bay, y_std=np.sqrt(sigma2_bay),
                title="Bayesian predictive distribution (sinusoidal $g$)")

plt.tight_layout()
plt.show()


# %% [markdown]
# As expected, the predictive variance is **larger in regions with few
# observations** (far from the highlighted training points) and decreases
# wherever training data are available. This adaptive behaviour is the
# hallmark of the full Bayesian treatment: unlike MLE, the uncertainty
# estimate is not constant but reflects the actual information content of
# the data at each point of the domain.


# %% [markdown]
# ---
# ## 6. Marginal Likelihood and Evidence
#
# ### Model selection
#
# So far the hyperparameters $\alpha$, $\beta$ and the number $m$ of basis
# functions have been fixed by hand. A principled data-driven approach is
# provided by the **marginal likelihood** (also called the *evidence*).
#
# ### Marginal likelihood (evidence)
#
# The marginal likelihood is obtained by integrating the parameters
# $\mathbf{w}$ out of the joint distribution:
#
# $$
# p(\mathbf{t} \mid \mathbf{X}, \alpha, \beta)
#   = \int p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \beta)\,
#     p(\mathbf{w} \mid \alpha)\, d\mathbf{w}
# $$
#
# Maximising this quantity w.r.t. $\alpha$ and $\beta$ yields hyperparameter
# estimates **directly from the training set**, without a separate validation
# set. Comparing the evidence of two models (different values of $m$) is
# equivalent to computing a **Bayes factor**.
#
# ### Log-evidence formula
#
# In the Gaussian setting the integral is analytically tractable:
#
# $$
# \log p(\mathbf{t} \mid \mathbf{X}, \alpha, \beta)
#   = \frac{m}{2}\log\alpha + \frac{n}{2}\log\beta
#     - E(\mathbf{m}_p)
#     - \frac{1}{2}\log\left|\boldsymbol{\Sigma}_p^{-1}\right|
#     - \frac{n}{2}\log(2\pi)
# $$
#
# where $E(\mathbf{m}_p)$ combines the data-fit and regularisation terms
# evaluated at the posterior mean:
#
# $$
# E(\mathbf{m}_p)
#   = \frac{\beta}{2}\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{m}_p\|^2
#   + \frac{\alpha}{2}\|\mathbf{m}_p\|^2
# $$
#
# The evidence automatically penalises overly complex models through the
# log-determinant term — this is the **Bayesian Occam's razor**: models
# that are unnecessarily complex spread their probability mass over too
# many functions and are penalised relative to simpler alternatives that
# fit the data equally well.

# %%
def log_marginal_likelihood(X, t, alpha, beta, args):
    """
    Compute the log marginal likelihood (log evidence).

    Formula (Bishop, PRML, eq. 3.86):
        log p(t|X,α,β) = (m/2)log α + (n/2)log β
                         − E(m_p) − (1/2)log|Σ_p⁻¹| − (n/2)log(2π)
    with:
        E(m_p) = (β/2)‖t − Φ m_p‖² + (α/2)‖m_p‖²

    Parameters
    ----------
    X, t        : training inputs and targets.
    alpha, beta : hyperparameters.
    args        : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    log_ml : float, log marginal likelihood.
    """
    Phi          = expand(X, **args)                            # (n, m) design matrix
    n, m         = Phi.shape                                    # number of samples and basis functions
    m_p, Sigma_p = posterior_params(X, t, alpha, beta, args)   # posterior parameters
    Sigma_p_inv  = np.linalg.inv(Sigma_p)                      # posterior precision matrix

    # Data-fit term (weighted sum of squared residuals)
    E_D = beta  * np.sum((t - Phi @ m_p) ** 2)
    # Regularisation term (weighted squared parameter norm)
    E_W = alpha * np.sum(m_p ** 2)

    score = (
        m * np.log(alpha)                           # prior precision contribution
        + n * np.log(beta)                          # noise precision contribution
        - E_D - E_W                                 # fit + regularisation penalty
        - np.log(np.linalg.det(Sigma_p_inv))        # model complexity penalty
        - n * np.log(2 * np.pi)                     # normalisation constant
    )
    return 0.5 * score


# %% [markdown]
# ### Model comparison via evidence
#
# The panels below show Bayesian regression fits for models with an
# increasing number $m$ of equally-spaced Gaussian basis functions
# (from 1 to 10). Each panel displays the posterior predictive mean and
# ±1σ band. The log-evidence curve (separate plot) reveals the model
# complexity preferred by the data — too few basis functions underfit,
# too many overfit, and the evidence peaks at the right trade-off.

# %%
BETA_EV  = 1 / (0.3 ** 2)          # noise precision for the evidence experiment
ALPHA_EV = 0.005                    # prior precision for the evidence experiment
N_EV     = 20                       # number of training examples
M_MAX    = 10                       # maximum number of basis functions to try

# Generate a dedicated dataset on [0, 1] for this experiment
x0   = np.linspace(0, 1, 1000).reshape(-1, 1)  # dense evaluation grid
y0   = g(x0, noise_variance=0).T               # noiseless sinusoidal curve
X0   = np.random.rand(50, 1)[:N_EV]            # N_EV random training inputs
t0   = g(X0, noise_variance=1 / BETA_EV)       # noisy training targets

rows = M_MAX // 3 + 1               # number of subplot rows

plt.figure(figsize=(18, 18))
plt.subplots_adjust(hspace=0.4)

mlls = []                           # store log-evidence for each model
for d in range(1, M_MAX + 1):
    # Build d equally-spaced Gaussian basis functions on [0, 1]
    mus_ev  = np.linspace(x0[0], x0[-1], d + 1)
    args_ev = {
        "bf": gaussian_basis_function,
        "bf_args_list": [
            # Centre at the midpoint of each interval
            {"mu": mus_ev[k] + 0.5 * (mus_ev[k + 1] - mus_ev[k])}
            for k in range(len(mus_ev) - 1)
        ],
    }
    m_p_ev, Sigma_p_ev = posterior_params(X0, t0, ALPHA_EV, BETA_EV, args_ev)
    y_ev, y_var_ev     = posterior_predictive(x0, m_p_ev, Sigma_p_ev, BETA_EV, args_ev)
    mlls.append(log_marginal_likelihood(X0, t0, ALPHA_EV, BETA_EV, args_ev))

    ax = plt.subplot(rows, 3, d)
    plt.scatter(X0, t0, s=15, color=COLORS[1])                         # training points
    plt.plot(x0, y0.ravel(), color=COLORS[2], linestyle="dashed")      # true curve
    plt.plot(x0, y_ev, color=COLORS[3])                                 # predictive mean
    plt.fill_between(
        x0.ravel(),
        y_ev.ravel() - y_var_ev.ravel(),    # lower ±1σ bound
        y_ev.ravel() + y_var_ev.ravel(),    # upper ±1σ bound
        color=COLORS[8], alpha=0.2
    )
    plt.title(f"$m = {d}$ basis function(s)")
    plt.ylim(-1.0, 2.0)
plt.show()


# %% [markdown]
# ### Log-evidence as a function of model complexity
#
# The plot below shows $\log p(\mathbf{t} \mid \mathbf{X}, \alpha, \beta)$
# as a function of the number of basis functions $m$. The optimal model
# (vertical dashed line) maximises the evidence and corresponds to the best
# compromise between data fit and model complexity.

# %%
degree_max = int(np.argmax(mlls)) + 1       # 1-indexed optimal number of basis functions

plt.figure(figsize=(12, 6))
plt.plot(range(1, M_MAX + 1), mlls, color=COLORS[6], lw=1, marker="o", markersize=5)
plt.axvline(x=degree_max, linestyle="dashed", color=COLORS[2], lw=1,
            label=f"Optimal: {degree_max} basis function(s)")
plt.xticks(range(1, M_MAX + 1))
plt.xlabel("Number of equally-spaced Gaussian basis functions")
plt.ylabel("Log marginal likelihood")
plt.legend()
plt.show()


# %% [markdown]
# ### Empirical Bayes: iterative optimisation of $\alpha$ and $\beta$
#
# Rather than scanning a grid of hyperparameter values, we can
# **analytically maximise** the log-evidence w.r.t. $\alpha$ and $\beta$.
# The stationarity conditions yield a pair of implicit equations that are
# solved iteratively (Expectation-Maximisation style):
#
# $$
# \alpha^{\mathrm{new}} = \frac{\gamma}{\|\mathbf{m}_p\|^2}, \qquad
# \left(\beta^{\mathrm{new}}\right)^{-1}
#   = \frac{1}{n - \gamma}
#     \sum_{i=1}^{n}\bigl(t_i - \boldsymbol{\phi}(\mathbf{x}_i)\mathbf{m}_p\bigr)^2
# $$
#
# where $\gamma$ is the **effective number of parameters** (also known as
# the *hat matrix trace* in frequentist statistics):
#
# $$
# \gamma = \sum_{i=0}^{m-1} \frac{\lambda_i}{\alpha + \lambda_i}, \qquad
# \lambda_i = \text{eigenvalues of } \beta\,\boldsymbol{\Phi}^\top\boldsymbol{\Phi}
# $$
#
# When $\lambda_i \gg \alpha$, eigenvalue $i$ contributes ~1 to $\gamma$
# (the corresponding direction in parameter space is well-determined by the
# data); when $\lambda_i \ll \alpha$, it contributes ~0 (the direction is
# dominated by the prior). This selective "switching on" of parameters is
# called **Automatic Relevance Determination (ARD)**.

# %%
def marginal_likelihood_maximization(
    X, t, args, alpha=1e-5, beta=1e-5, max_iter=200, rtol=1e-5, verbose=False
):
    """
    Iterative hyperparameter estimation via Empirical Bayes (type-II MLE).

    Algorithm (Bishop, PRML, Sec. 3.5):
        1. Compute eigenvalues λ_i of ΦᵀΦ  (done once, before the loop).
        2. Compute γ = Σ λ_i·β / (α + λ_i·β)  (effective parameter count).
        3. Update α ← γ / ‖m_p‖²
        4. Update β ← (n − γ) / ‖t − Φ m_p‖²
        5. Repeat until convergence (relative change < rtol in both α and β).

    Parameters
    ----------
    X, t         : training inputs and targets.
    args         : dict, keyword arguments forwarded to `expand`.
    alpha, beta  : initial hyperparameter values (small positive numbers).
    max_iter     : maximum number of iterations before stopping.
    rtol         : relative tolerance for convergence check.
    verbose      : if True, print iteration count upon termination.

    Returns
    -------
    alpha, beta : optimised hyperparameters.
    """
    Phi              = expand(X, **args)                        # (n, m) design matrix
    # Eigenvalues of ΦᵀΦ — computed once; scaled by β inside the loop
    eigenvalues_base = np.linalg.eigvalsh(Phi.T @ Phi)

    for i in range(max_iter):
        alpha_prev, beta_prev = alpha, beta     # store previous values for convergence check

        eigenvalues = eigenvalues_base * beta   # scaled eigenvalues λ_i = β · eigval_i
        m_p, _      = posterior_params(X, t, alpha, beta, args)    # posterior mean
        # Effective number of parameters (sum of "data contribution" per eigendirection)
        gamma       = np.sum(eigenvalues / (eigenvalues + alpha))

        # Update α: ratio of effective parameters to squared norm of posterior mean
        alpha    = gamma / np.sum(m_p ** 2)
        # Update β: inverse of mean squared residual scaled by (n − γ)
        beta_inv = np.sum((t - Phi @ m_p) ** 2) / (Phi.shape[0] - gamma)
        beta     = 1.0 / beta_inv

        # Check convergence: both hyperparameters must stabilise
        if (np.isclose(alpha_prev, alpha, rtol=rtol)
                and np.isclose(beta_prev, beta, rtol=rtol)):
            if verbose:
                print(f"  Converged after {i + 1} iterations.")
            return alpha, beta

    if verbose:
        print(f"  Stopped after {max_iter} iterations without convergence.")
    return alpha, beta


# %%
N_EB   = 30     # number of training examples for the Empirical Bayes experiment
DEGREE = 4      # polynomial degree

# Regularly-spaced training set on [0, 1] — deterministic grid for reproducibility
Xc = np.linspace(0, 1, N_EB).reshape(-1, 1)
tc = g(Xc, noise_variance=0.3 ** 2)            # sinusoidal targets with σ² = 0.09

# Polynomial basis: x, x², x³, x⁴ (no bias — the sinusoidal dataset has a DC component
# already captured by the 0.5 offset in g)
args_poly = {
    "bf":           polynomial_basis_function,
    "bf_args_list": [{"power": k} for k in range(1, DEGREE + 1)],
}

print("=== Empirical Bayes (custom implementation) ===")
alpha_opt, beta_opt = marginal_likelihood_maximization(
    Xc, tc, args_poly, rtol=1e-5, verbose=True
)
print(f"  alpha* = {alpha_opt:.6f}")
print(f"  beta*  = {beta_opt:.6f}")


# %% [markdown]
# ### Comparison with scikit-learn's `BayesianRidge`
#
# `BayesianRidge` implements the same Empirical Bayes algorithm with a
# minor variation: it places **Gamma priors** over $\alpha$ and $\beta$
# rather than using flat (improper) priors. This introduces a small
# regularisation effect on the hyperparameters themselves, slightly
# favouring smaller values. In practice, however, the estimates converge
# to values very close to our implementation.
#
# The mapping between scikit-learn attributes and our notation is:
# `br.lambda_` $= \alpha$ (weight precision), `br.alpha_` $= \beta$
# (noise precision).

# %%
Phi_c = expand(Xc, **args_poly)                         # design matrix for polynomial basis
br    = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=True)
br.fit(Phi_c, tc.ravel())                               # fit the model (estimates α and β)

print("\n=== BayesianRidge (scikit-learn) ===")
print(f"  alpha* = {br.lambda_:.6f}")     # weight precision (our alpha)
print(f"  beta*  = {br.alpha_:.6f}")      # noise precision (our beta)


# %% [markdown]
# ---
# ## 7. Equivalent Kernel
#
# ### Prediction as a linear combination of training targets
#
# The Bayesian predictive mean can be rewritten as a **weighted sum of
# training targets** by substituting the posterior mean $\mathbf{m}_p =
# \beta\boldsymbol{\Sigma}_p\boldsymbol{\Phi}^\top\mathbf{t}$:
#
# $$
# \boldsymbol{\phi}(\mathbf{x})\mathbf{m}_p
#   = \beta\,\boldsymbol{\phi}(\mathbf{x})\,\boldsymbol{\Sigma}_p\,
#     \boldsymbol{\Phi}^\top\mathbf{t}
#   = \sum_{i=1}^{n}
#     \underbrace{\beta\,\boldsymbol{\phi}(\mathbf{x})\,\boldsymbol{\Sigma}_p\,
#     \boldsymbol{\phi}(\mathbf{x}_i)^\top}_{\kappa(\mathbf{x},\,\mathbf{x}_i)}
#     \, t_i
# $$
#
# The function $\kappa(\mathbf{x}, \mathbf{x}')$ is the **equivalent kernel**:
#
# $$
# \kappa(\mathbf{x}, \mathbf{x}')
#   = \beta\,\boldsymbol{\phi}(\mathbf{x})\,\boldsymbol{\Sigma}_p\,
#     \boldsymbol{\phi}(\mathbf{x}')^\top
# $$
#
# It assigns a weight to each training target that depends on the "distance"
# between $\mathbf{x}$ and $\mathbf{x}'$ in the feature space induced by the
# basis functions, measured with the posterior covariance $\boldsymbol{\Sigma}_p$
# as the metric.
#
# This representation reveals that Bayesian linear regression is, in effect,
# a **non-parametric model**: predictions are not mediated by a fixed set of
# parameters but by the entire training set. The equivalent kernel implements
# a **locality principle**: training points close to the test point receive
# higher weights, as illustrated by the concentration along the diagonal in
# the colour maps below.

# %%
def equiv_kernel(x1, x2, X, t, alpha, beta, args):
    """
    Compute the equivalent kernel κ(x₁, x₂).

        κ(x, x') = β · φ(x) · Σ_p · φ(x')ᵀ

    Parameters
    ----------
    x1, x2      : float or array, evaluation points.
    X, t        : training set (needed to compute Σ_p).
    alpha, beta : hyperparameters.
    args        : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    k : float or array, equivalent kernel value(s).
    """
    phi_1      = expand(x1, **args)                         # feature vector at x1
    phi_2      = expand(x2, **args)                         # feature vector at x2
    _, Sigma_p = posterior_params(X, t, alpha, beta, args)  # posterior covariance
    return beta * phi_1 @ Sigma_p @ phi_2.T                 # κ = β φ(x1) Σ_p φ(x2)ᵀ


def predict_equiv_kernel(X_test, X, t, alpha, beta, args):
    """
    Make predictions via the equivalent kernel:
        y(x) = β · φ(x) · Σ_p · Φᵀ · t = Σ_i κ(x, x_i) · t_i

    Parameters
    ----------
    X_test      : array, test inputs.
    X, t        : training set.
    alpha, beta : hyperparameters.
    args        : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    y : array, predicted values (should match `posterior_predictive` mean).
    """
    phi_test   = expand(X_test, **args)                     # test design matrix
    phi_train  = expand(X, **args)                          # training design matrix
    _, Sigma_p = posterior_params(X, t, alpha, beta, args)  # posterior covariance
    return beta * phi_test @ Sigma_p @ phi_train.T @ t      # weighted sum of targets


# %% [markdown]
# ### Equivalent kernel visualisation
#
# The colour maps below show $\kappa(x_1, x_2)$ over the grid
# $[-1,1] \times [-1,1]$, for both the identity and Gaussian bases.
# High values (warm colours) near the diagonal confirm the locality
# principle: the prediction at $x$ borrows most of its strength from
# training targets with $x_i \approx x$. The Gaussian basis produces a
# narrower kernel (stronger locality) than the identity basis.

# %%
n_values = 200                                  # grid resolution per axis
xx = np.linspace(-1, 1, n_values)              # x₁ axis
yy = np.linspace(-1, 1, n_values)              # x₂ axis
XX, YY = np.meshgrid(xx, yy)                   # 2-D grid

# Kernel arguments for each dataset/basis combination
args_lin_ker = {}                               # identity (augmented) basis
args_gau_ker = {
    "bf":           gaussian_basis_function,
    "bf_args_list": [{"mu": mu} for mu in np.linspace(-1, 1, K)],
}

for label, t_ref, args_ref in [
    ("Identity basis ($f$)",           t1, args_lin_ker),
    ("Gaussian basis functions ($g$)", t2, args_gau_ker),
]:
    # Vectorise over the grid (equiv_kernel is defined for scalars)
    func = np.vectorize(
        lambda xi, yi: equiv_kernel(xi, yi, X, t_ref, ALPHA, BETA, args_ref)
    )
    Z = func(XX, YY)                            # (n_values, n_values) kernel matrix

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(Z, origin="lower",
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              aspect="auto", alpha=0.8)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.grid(color="0.35")
    plt.title(f"Equivalent kernel – {label}")
    plt.show()


# %% [markdown]
# ---
# ## 8. Kernel Regression (Nadaraya-Watson)
#
# ### From the equivalent kernel to non-parametric regression
#
# The equivalent kernel's ability to weight training targets by input-space
# similarity is the founding principle of **kernel regression** methods.
# Rather than deriving the kernel from a parametric model, we can directly
# specify it as a function of the input distance.
#
# The **Nadaraya-Watson** estimator approximates the conditional expectation
# $f(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]$ by placing a kernel
# density estimator on the joint distribution $p(\mathbf{x}, t)$:
#
# $$
# p(\mathbf{x}, t) \approx \frac{1}{n} \sum_{i=1}^{n}
#   \kappa_h(\mathbf{x} - \mathbf{x}_i)\,\kappa_h(t - t_i)
# $$
#
# Substituting into the formula for the conditional expectation yields:
#
# $$
# f(\mathbf{x}) \approx
#   \frac{\sum_{i=1}^{n}\kappa_h(\mathbf{x}-\mathbf{x}_i)\,t_i}
#        {\sum_{i=1}^{n}\kappa_h(\mathbf{x}-\mathbf{x}_i)}
# $$
#
# This is a **locally constant** (kernel-weighted average) estimator. The
# most common kernel choice is the **RBF (Gaussian) kernel**:
#
# $$
# \kappa_h(\mathbf{x}) = \exp\!\left(-\frac{\|\mathbf{x}\|^2}{2h^2}\right)
# $$
#
# The **bandwidth** $h$ is the key smoothing parameter:
# - **Small $h$**: highly local predictions — the estimator nearly
#   interpolates the training data; prone to overfitting.
# - **Large $h$**: smooth, global predictions — approaches the sample mean;
#   prone to underfitting.
#
# ### Bandwidth selection via LOO cross-validation
#
# The optimal $h^*$ is found by minimising the Leave-One-Out (LOO) MSE.
# A key computational shortcut: zeroing the diagonal of the kernel matrix
# $K_{ii} = 0$ excludes each point's self-contribution without refitting
# for each fold, giving an efficient $O(n^2)$ approximation.

# %%
def rbf_kernel(x, X, sigma):
    """
    Radial Basis Function (Gaussian) kernel matrix.

        K[i, j] = exp(−‖X[i] − x[j]‖² / (2σ²))

    Parameters
    ----------
    x, X  : arrays of points already mapped through the design matrix (n×d).
    sigma : float, bandwidth h.

    Returns
    -------
    K : array (|X|, |x|), kernel weight matrix. Entry K[i,j] is the
        weight assigned to training point i when predicting at test point j.
    """
    dist_mat = cdist(X, x, "minkowski", p=2.0)              # pairwise Euclidean distances
    return np.exp(-0.5 / sigma ** 2 * dist_mat ** 2)        # Gaussian kernel


def predict_kernel_regression(x, X, t, h, args):
    """
    Nadaraya-Watson prediction with RBF kernel.

    For each test point x_j:
        y(x_j) = Σ_i K[i,j] · t_i / Σ_i K[i,j]

    Parameters
    ----------
    x, X : test and training points (scalar or array).
    t    : array (n, 1), training targets.
    h    : float, bandwidth.
    args : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    y : array or scalar, Nadaraya-Watson predictions.
    """
    weights = rbf_kernel(expand(x, **args), expand(X, **args), h)  # (n, n_test) weight matrix
    y       = (weights * t).sum(axis=0) / weights.sum(axis=0)      # weighted average per test point
    return y.item() if np.isscalar(x) else y


def select_bandwidth_kernel(X, t, hs, args):
    """
    Select the optimal bandwidth via approximate LOO cross-validation.

    For each candidate bandwidth h, zeroing the diagonal of the kernel
    matrix K simulates leave-one-out prediction: point i never contributes
    to its own prediction, avoiding trivial overfitting.

    Parameters
    ----------
    X, t : training set.
    hs   : array, candidate bandwidth values.
    args : dict, keyword arguments forwarded to `expand`.

    Returns
    -------
    h_star : float, bandwidth that minimises the LOO MSE.
    """
    XX       = expand(X, **args)    # map training inputs to feature space once
    mse_list = []
    for h in hs:
        K   = rbf_kernel(XX, XX, h)             # (n, n) kernel matrix
        K   = K - np.diag(np.diag(K))           # zero diagonal → LOO exclusion
        y   = (K * t).sum(axis=0) / K.sum(axis=0)  # LOO predictions
        mse_list.append(((y[:, np.newaxis] - t) ** 2).mean())  # LOO MSE
    return hs[np.argmin(mse_list)]              # return bandwidth with minimum LOO MSE


# %% [markdown]
# ### Kernel regression with optimal bandwidth
#
# The plots below show the Nadaraya-Watson estimator fitted with the
# bandwidth $h^*$ chosen by LOO cross-validation. On the linear dataset
# the estimator recovers the linear trend. On the sinusoidal dataset,
# sigmoid basis functions are applied before computing the kernel distances,
# enriching the feature space and allowing the kernel to capture the
# oscillatory pattern.

# %%
# Candidate bandwidth values to evaluate
hs = np.linspace(0.1, 1, 10)

# ── Linear dataset: identity features ────────────────────────────────────────
args_kr = {}
h_star  = select_bandwidth_kernel(X, t1, hs, args_kr)              # LOO-optimal bandwidth

# ── Sinusoidal dataset: sigmoidal features ────────────────────────────────────
K   = 10
mus = np.linspace(-1, 1, K)
args_kr_sin = {
    "bf":           sigmoid_basis_function,
    "bf_args_list": [{"mu": mu} for mu in mus],
}
h_star_sin = select_bandwidth_kernel(X, t2, hs, args_kr_sin)       # LOO-optimal bandwidth

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
plot_regression(x, y1, X, t1,
                y_pred=predict_kernel_regression(x, X, t1, h_star, args_kr),
                title=f"Kernel Regression – Linear $f$  ($h^*={h_star:.2f}$)")

plt.sca(axes[1])
plot_regression(x, y2, X, t2,
                y_pred=predict_kernel_regression(x, X, t2, h_star_sin, args_kr_sin),
                title=f"Kernel Regression – Sinusoidal $g$  ($h^*={h_star_sin:.2f}$)")

plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## 9. Locally Weighted Regression (LOESS)
#
# ### Motivation
#
# The Nadaraya-Watson estimator is effectively a **locally constant** model:
# it estimates $f(\mathbf{x})$ as a weighted mean of targets. When the
# target function has locally varying slope or curvature, this introduces
# systematic **bias** even as $h \to 0$.
#
# **LOESS** (Locally Weighted Scatterplot Smoothing, also LOWESS) addresses
# this by fitting a **local linear model** around each test point, with
# training data weighted by proximity. The key insight is that locally, any
# smooth function can be approximated by its first-order Taylor expansion;
# using a linear model rather than a constant therefore reduces bias.
#
# ### Formulation
#
# To predict the target at a test point $\mathbf{x}$, we minimise the
# **locally weighted least-squares loss**:
#
# $$
# L(\mathbf{x})
#   = \sum_{i=1}^{n} \kappa_h(\mathbf{x} - \mathbf{x}_i)\,
#     \bigl(\mathbf{w}^\top \bar{\mathbf{x}}_i - t_i\bigr)^2
# $$
#
# where $\bar{\mathbf{x}}_i = [1, \mathbf{x}_i]$ is the bias-augmented input.
# The closed-form solution uses the diagonal weight matrix
# $\boldsymbol{\Psi}(\mathbf{x})$ with $\Psi_{ii} = \kappa_h(\mathbf{x}-\mathbf{x}_i)$:
#
# $$
# \hat{\mathbf{w}}(\mathbf{x})
#   = \left(\bar{\mathbf{X}}^\top \boldsymbol{\Psi}(\mathbf{x}) \bar{\mathbf{X}}\right)^{-1}
#     \bar{\mathbf{X}}^\top \boldsymbol{\Psi}(\mathbf{x}) \mathbf{t}
# $$
#
# and the prediction at $\mathbf{x}$ is $y(\mathbf{x}) = \bar{\mathbf{x}}^\top
# \hat{\mathbf{w}}(\mathbf{x})$.
#
# Note that $\hat{\mathbf{w}}(\mathbf{x})$ **depends on $\mathbf{x}$**: the
# local model is re-fitted for every test point. This makes LOESS
# computationally more expensive than Nadaraya-Watson ($O(n^2)$ per
# prediction), but generally more accurate when the function has non-constant
# local trends.

# %%
def _loess_weights(x, X, h):
    """
    Gaussian kernel weights for LOESS: κ_h(x − X_i), vectorised over training points.

    Parameters
    ----------
    x : float or array (1,), query point.
    X : array (n, d), training inputs.
    h : float, bandwidth.

    Returns
    -------
    w : array (n,), kernel weight for each training point.
    """
    return np.exp(np.sum((X - x) ** 2, axis=1) / (-2 * h * h))


def _loess_weight_matrix(x, X, h):
    """
    Diagonal weight matrix Ψ(x) for LOESS.

    Returns a diagonal (n×n) matrix with κ_h(x − x_i) on the diagonal.
    """
    return np.diag(_loess_weights(x, X, h))


def local_regression_coeffs(x, X, t, h):
    """
    Compute the locally weighted linear regression coefficients at x.

        ŵ(x) = (X_ext^T Ψ(x) X_ext)⁻¹ X_ext^T Ψ(x) t

    where X_ext = [1 | X] is the bias-augmented training matrix.

    Parameters
    ----------
    x    : float, query point.
    X, t : training set.
    h    : float, bandwidth.

    Returns
    -------
    w : array (2,), local linear coefficients [w₀ (bias), w₁ (slope)].
    """
    Psi   = _loess_weight_matrix(x, X, h)              # (n, n) diagonal weight matrix
    X_ext = np.c_[np.ones(len(X)), X]                  # (n, 2) bias-augmented inputs
    # Weighted normal equations
    return np.linalg.pinv(X_ext.T @ Psi @ X_ext) @ X_ext.T @ Psi @ t


def local_regression(x, X, t, h):
    """
    LOESS prediction at scalar test point x.

        y = x_ext^T ŵ(x),  with x_ext = [1, x]

    Parameters
    ----------
    x    : float, test point.
    X, t : training set.
    h    : float, bandwidth.

    Returns
    -------
    y : float, LOESS predicted value.
    """
    x_ext = np.r_[1, x]                                # bias-augmented test point [1, x]
    w     = local_regression_coeffs(x, X, t, h)        # locally fitted coefficients
    return x_ext @ w                                    # linear prediction


# %% [markdown]
# ### Gaussian kernel centred at a query point
#
# The plot below shows the Gaussian kernel $\kappa_h(x - x_0)$ for a fixed
# query point $x_0 = 0.1$ and bandwidth $h = 0.05$. Training points close
# to $x_0$ receive weights near 1; points far away are downweighted towards 0.
# This weight profile governs which training examples are "active" when
# fitting the local linear model at $x_0$.

# %%
h      = 0.05           # bandwidth for LOESS
x_     = 0.1            # query point to visualise
domain = np.linspace(-1, 1, 500)    # evaluation domain

plt.figure(figsize=(16, 4))
plt.title(f"LOESS kernel centred at $x_0 = {x_}$ (bandwidth $h = {h}$)")
plt.plot(domain, _loess_weights(x_, domain.reshape(-1, 1), h), color=COLORS[1])
plt.xlabel("$x$")
plt.ylabel(r"Weight $\kappa_h(x - x_0)$")
plt.show()


# %% [markdown]
# ### Local linear fit at a single query point
#
# At $x = 0.1$, the local regression is a weighted least-squares fit of
# a straight line to the training data, with weights given by the kernel
# above. The green dot marks the LOESS prediction $y(x_0)$ (the line
# evaluated at the query point), and the dashed line extends the local
# model across the domain for illustration.

# %%
y_local = local_regression(x_, X, t1, h)               # LOESS prediction at x_
w_local = local_regression_coeffs(x_, X, t1, h)        # locally fitted coefficients

plt.figure(figsize=(16, 8))
plt.title(f"Local regression at $x = {x_}$ (bandwidth $h = {h}$)")
plt.scatter(X, t1, c=COLORS[1], alpha=0.5, label="Observations")
plt.plot([x_], [y_local], marker="o", color=COLORS[4],
         markersize=10, label="Local prediction")
plt.plot(domain, np.c_[np.ones(len(domain)), domain] @ w_local,
         color=COLORS[1], lw=1, label="Local fitted line")    # global extent of local model
plt.plot(x, y1, lw=2, color=COLORS[2], linestyle="dashed", label="True function")
plt.legend()
plt.show()


# %% [markdown]
# ### Full LOESS curve — fixed bandwidth
#
# By applying local regression at every point of the dense domain grid,
# we obtain the full LOESS curve. The result is a smooth non-parametric
# estimate that adapts locally to the data. With a small bandwidth $h = 0.05$
# the fit is very local and can closely track the training data, but may
# exhibit slight oscillations in sparse regions.

# %%
# Apply local regression at each point of the evaluation grid
prediction_loess = [local_regression(xi, X, t1, h) for xi in domain]

plt.figure(figsize=(16, 8))
plt.title(f"LOESS – Linear function $f$ (bandwidth $h = {h}$)")
plt.scatter(X, t1, c=COLORS[1], alpha=0.5, label="Observations")
plt.plot(domain, prediction_loess, lw=2, color=COLORS[1], label="LOESS")
plt.plot(x, y1, lw=2, color=COLORS[2], linestyle="dashed", label="True function")
plt.legend()
plt.show()


# %% [markdown]
# ### LOESS with LOO-optimal bandwidth
#
# Using the same LOO cross-validation strategy as for kernel regression,
# we select $h^*$ that minimises the leave-one-out MSE. A larger $h^*$
# produces a smoother LOESS estimate that is less sensitive to individual
# observations but may miss fine-scale structure.

# %%
args_loess = {}                             # identity (raw) features for LOESS
hs         = np.linspace(0.1, 1, 10)       # candidate bandwidths
h_star_lw  = select_bandwidth_kernel(X, t1, hs, args_loess)    # LOO-optimal bandwidth

# Compute LOESS predictions at every grid point using the optimal bandwidth
prediction_star = [local_regression(xi, X, t1, h_star_lw) for xi in domain]

plt.figure(figsize=(16, 8))
plt.scatter(X, t1, s=20, color=COLORS[1], label="Observations")
plt.plot(domain, prediction_star, lw=2, color=COLORS[6],
         label=f"LOESS ($h^*={h_star_lw:.2f}$)")
plt.plot(x, y1, lw=2, color=COLORS[2], linestyle="dashed", label="True function")
plt.legend()
plt.title("LOESS with optimal bandwidth – Linear function $f$")
plt.show()
