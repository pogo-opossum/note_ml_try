# %% [markdown]
# ---
# # Face Recognition with PCA (Eigenfaces) and SVM
#
# ## Overview
#
# This notebook tackles a **multi-class image classification** problem: given a
# greyscale photograph of a face, predict *whose* face it is.  We work with the
# **Labeled Faces in the Wild (LFW)** dataset, a standard benchmark containing
# real-world photos of public figures collected from the web.
#
# The pipeline has two stages:
#
# 1. **Dimensionality reduction via PCA** — raw pixel vectors are high-dimensional
#    and highly correlated.  PCA compresses each image into a compact set of
#    coefficients along the directions of maximum variance (the *eigenfaces*),
#    discarding noise and reducing the computational burden for the classifier.
#
# 2. **Classification via a kernel SVM** — a Support Vector Machine with an RBF
#    kernel is trained on the PCA-projected features.  Its hyperparameters are
#    selected automatically by cross-validated grid search.
#
# ## Why PCA First?
#
# An image of size $h \times w$ pixels is represented as a vector
# $\mathbf{x} \in \mathbb{R}^{h \cdot w}$.  With $h \cdot w \sim 10^3$–$10^4$
# and only a few hundred training samples, directly feeding raw pixels to an SVM
# would be both slow and prone to the *curse of dimensionality*.
#
# PCA finds the orthonormal basis $\{\mathbf{u}_k\}_{k=1}^{K}$ that maximises
# retained variance:
#
# $$
# \mathbf{u}_k = \underset{\|\mathbf{u}\|=1}{\arg\max}\;
# \operatorname{Var}\!\left(\mathbf{u}^\top \mathbf{X}\right),
# \quad \mathbf{u}_k \perp \mathbf{u}_j \; \forall\, j < k.
# $$
#
# Each image is then represented by $K \ll h \cdot w$ scalar projections
# (the **principal components**), yielding a much lower-dimensional feature
# vector while retaining most of the discriminative information.
#
# When applied to face images the principal components are themselves
# face-shaped images — hence the name **eigenfaces**.
#
# ## Why an RBF-SVM?
#
# After PCA the feature space is dense and relatively low-dimensional.  An SVM
# with the RBF kernel:
#
# $$
# k(\mathbf{z}, \mathbf{z}') = \exp\!\left(-\gamma\,\|\mathbf{z}-\mathbf{z}'\|^2\right)
# $$
#
# can model non-linear, multi-modal class boundaries in this space via the
# dual representation:
#
# $$
# f(\mathbf{z}) = \sum_{i \in \mathcal{SV}} \alpha_i\, y_i\, k(\mathbf{z}_i, \mathbf{z}) + b.
# $$
#
# The two free hyperparameters — the soft-margin penalty $C$ and the kernel
# bandwidth $\gamma$ — are tuned by **5-fold cross-validated grid search**.
# 

# %%
# ── Suppress non-critical deprecation warnings ────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

# %%
# ── Standard library ──────────────────────────────────────────────────────────
from time import time

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ── Scikit-learn: data, preprocessing, modelling, evaluation ─────────────────
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# %%
# ── Utility: gallery plot ─────────────────────────────────────────────────────
def plot_gallery(images, h, w, n_row=3, n_col=4, titles=None):
    """
    Display a rectangular grid of greyscale images.

    Parameters
    ----------
    images  : array-like, shape (N, h*w) — flattened image vectors
    h, w    : int — image height and width in pixels
    n_row   : int — number of rows in the grid
    n_col   : int — number of columns in the grid
    titles  : list of str or None — optional per-image caption
    """
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(1.8 * n_col, 2.4 * n_row))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99,
                        top=0.90, hspace=0.35)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis('off')
            continue
        ax.imshow(images[idx].reshape((h, w)), cmap='gray')
        if titles is not None:
            ax.set_title(titles[idx], size=9, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# %% [markdown]
# ---
# ## 1 – Dataset: Labeled Faces in the Wild (LFW)
#
# The **LFW** dataset contains JPEG photographs of famous people scraped from
# the web.  Each image is aligned and centred on the face.  We keep only
# subjects with at least **70** images so that each class has enough examples
# for reliable training and evaluation.  Images are downscaled by a factor of
# $0.4$ to reduce dimensionality while preserving enough detail for recognition.
#
# After loading, the dataset is characterised by three integers:
# - $N$ — total number of images (samples)
# - $h \times w$ — spatial resolution of each image
# - $F = h \cdot w$ — number of raw pixel features per image
# - $C$ — number of identity classes (people)
# 

# %%
# ── Load the LFW dataset ──────────────────────────────────────────────────────
# min_faces_per_person=70  → keeps only the 7 most photographed subjects,
#                            giving a challenging but tractable 7-class problem.
# resize=0.4               → downscales from ~250×250 to ~50×37 pixels.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# %%
# ── Extract dataset dimensions ────────────────────────────────────────────────
n_samples, h, w = lfw_people.images.shape   # (N, height, width)

X = lfw_people.data                         # flattened pixel matrix, shape (N, h*w)
n_features = X.shape[1]                     # F = h * w

y = lfw_people.target                       # integer class labels, shape (N,)
target_names = lfw_people.target_names      # string names, shape (C,)
n_classes = target_names.shape[0]

# %%
# ── Dataset summary ───────────────────────────────────────────────────────────
print("═" * 40)
print("  LFW dataset summary")
print("═" * 40)
print(f"  Subjects (classes) : {n_classes}")
print(f"  Samples            : {n_samples}")
print(f"  Image resolution   : {h} × {w} px")
print(f"  Raw features / img : {n_features}")
print("═" * 40)
print(f"\n  Class labels:\n  {list(target_names)}")

# %%
# ── Figure 1 – Sample images from the dataset ─────────────────────────────────
plot_gallery(X, h, w, n_row=6, n_col=8)

# %%
# ── Train / test split (75% train, 25% test) ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# %% [markdown]
# ---
# ## 2 – Dimensionality Reduction: Eigenfaces via PCA
#
# ### Rationale
#
# With $F \approx 1{,}850$ pixel features and only ~1{,}000 training images,
# training an SVM directly on raw pixels is computationally expensive and
# statistically inefficient — many pixel dimensions carry no discriminative
# signal and introduce noise.
#
# PCA projects each image onto the $K$ directions of maximum variance in the
# training set.  These directions, reshaped as $h \times w$ images, are the
# **eigenfaces**: abstract face-like patterns that collectively span the subspace
# most informative for distinguishing identities.
#
# ### Whitening
#
# We also apply **ZCA whitening**, which rescales each principal component to
# unit variance:
#
# $$
# \tilde{z}_k = \frac{z_k}{\sqrt{\lambda_k}},
# $$
#
# where $\lambda_k$ is the $k$-th eigenvalue.  Whitening removes the
# variance difference between components, so the subsequent SVM treats all
# directions equally — beneficial since the RBF kernel is isotropic.
#
# ### Choosing $K$
#
# We retain $K = 50$ components.  The fraction of total variance explained is:
#
# $$
# \text{EVR}(K) = \frac{\sum_{k=1}^{K} \lambda_k}{\sum_{k=1}^{F} \lambda_k}.
# $$
#
# We print this ratio below; in practice $K = 50$ captures $\gtrsim 80\%$ of
# the variance while reducing the feature dimension by a factor of $\sim 37$.
# 

# %%
# ── Fit PCA on training images ────────────────────────────────────────────────
# svd_solver='randomized'  → Halko et al. randomised SVD, much faster than
#                            exact SVD when K << F.
# whiten=True              → ZCA whitening (unit variance per component).
K = 50
print(f"Fitting PCA: retaining top {K} components "
      f"from {X_train.shape[0]} training images …")
t0 = time()
pca = PCA(n_components=K, svd_solver='randomized', whiten=True)
pca.fit(X_train)
print(f"Done in {time() - t0:.2f}s")

# Fraction of total variance retained
evr = pca.explained_variance_ratio_.sum() * 100
print(f"Explained variance ratio (K={K}): {evr:.1f}%")

# %%
# ── Project data onto eigenface basis ─────────────────────────────────────────
X_train_pca = pca.transform(X_train)   # shape: (N_train, K)
X_test_pca  = pca.transform(X_test)    # shape: (N_test,  K)

# %%
# ── Extract eigenfaces (principal components as images) ───────────────────────
eigenfaces = pca.components_.reshape((K, h, w))   # each row is one eigenface

# Titles show the variance explained by each component
eigenface_titles = [f"λ={v:.2f}" for v in pca.explained_variance_]

# %%
# ── Figure 2 – Top-25 eigenfaces with explained variance ──────────────────────
plot_gallery(eigenfaces, h, w, n_row=5, n_col=5, titles=eigenface_titles)

# %% [markdown]
# ---
# ## 3 – Image Decomposition in the Eigenface Basis
#
# ### Mathematical Foundation
#
# Let $\bar{\mathbf{x}} \in \mathbb{R}^F$ be the **mean face** and let
# $\mathbf{u}_k \in \mathbb{R}^F$ ($\|\mathbf{u}_k\|=1$) be the $k$-th
# principal eigenvector of the sample covariance matrix of the training set,
# with associated eigenvalue $\lambda_k$ (the variance of the data along
# $\mathbf{u}_k$):
#
# $$
# \mathbf{S}\,\mathbf{u}_k = \lambda_k\,\mathbf{u}_k,
# \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_K > 0.
# $$
#
# The eigenfaces $\mathbf{u}_k$ form an orthonormal basis for the principal
# subspace.  Any centred image $\tilde{\mathbf{x}} = \mathbf{x} - \bar{\mathbf{x}}$
# is approximated by its truncated projection:
#
# $$
# \tilde{\mathbf{x}} \approx \hat{\mathbf{x}}_K
# = \sum_{k=1}^{K} z_k\, \mathbf{u}_k,
# \qquad z_k = \tilde{\mathbf{x}}^\top \mathbf{u}_k,
# $$
#
# so the full reconstruction in pixel space is
# $\mathbf{x} \approx \bar{\mathbf{x}} + \sum_{k=1}^{K} z_k\,\mathbf{u}_k$.
#
# ### The Role of Eigenvalues: Whitening and Reconstruction
#
# When `whiten=True`, scikit-learn stores **whitened** components in
# `pca.components_`:
#
# $$
# \tilde{\mathbf{u}}_k = \frac{\mathbf{u}_k}{\sqrt{\lambda_k}}
# \quad \Longrightarrow \quad
# \texttt{pca.components\_}[k] = \frac{\mathbf{u}_k}{\sqrt{\lambda_k}}.
# $$
#
# The whitened projection $\tilde{z}_k = \tilde{\mathbf{x}}^\top \tilde{\mathbf{u}}_k
# = z_k / \sqrt{\lambda_k}$ has unit variance across the training set, which
# benefits the downstream RBF-SVM (an isotropic kernel).
#
# To reconstruct the image in pixel space we must **undo the whitening**,
# multiplying back by $\sqrt{\lambda_k}$ to recover the true eigenvectors
# $\mathbf{u}_k = \sqrt{\lambda_k}\,\tilde{\mathbf{u}}_k$:
#
# $$
# \hat{\mathbf{x}}_K
# = \bar{\mathbf{x}}
#   + \sum_{k=1}^{K} z_k\,\mathbf{u}_k
# = \bar{\mathbf{x}}
#   + \sum_{k=1}^{K}
#     \underbrace{\left(\tilde{\mathbf{x}}^\top \tilde{\mathbf{u}}_k\right)}_{\tilde{z}_k}
#     \cdot
#     \underbrace{\sqrt{\lambda_k}\,\tilde{\mathbf{u}}_k}_{\mathbf{u}_k}.
# $$
#
# In code: `components_unwhitened = pca.components_ * sqrt(λ_k)`, which is
# exactly what `pca.inverse_transform` does internally.  **Omitting the
# $\sqrt{\lambda_k}$ factor would shrink each eigenface by its whitening
# scale, producing a blurred, incorrectly-scaled reconstruction.**
#
# ### Reconstruction Error
#
# The **relative reconstruction error** measures the fraction of image energy
# lost by retaining only $K$ components:
#
# $$
# \epsilon_K
# = \frac{\|\tilde{\mathbf{x}} - \hat{\mathbf{x}}_K\|^2}
#        {\|\tilde{\mathbf{x}}\|^2}
# = 1 - \frac{\displaystyle\sum_{k=1}^{K} z_k^2}
#            {\|\tilde{\mathbf{x}}\|^2}.
# $$
#
# Note that the numerator involves $z_k^2$ — the squared **unwhitened**
# coefficients — not $\tilde{z}_k^2$.  Since $z_k = \sqrt{\lambda_k}\,\tilde{z}_k$,
# components with large eigenvalues contribute proportionally more to the
# reconstructed energy, reflecting the fact that high-variance directions
# carry more structural information about the image.
#
# ### Interpreting the Coefficients
#
# The bar chart of $z_k^2 = \lambda_k\,\tilde{z}_k^2$ reveals **which eigenfaces
# contribute most** to a specific image.  A large $z_k^2$ can arise either
# because the eigenvalue $\lambda_k$ is large (a globally dominant direction,
# e.g. overall illumination) or because $\tilde{z}_k^2$ is large (the image
# is strongly aligned with that particular eigenface).  Separating the two
# effects — by plotting $\tilde{z}_k^2$ alongside $\lambda_k$ — gives a
# richer picture of what makes each face distinctive.
# 

# %%
# ── Parameters for the decomposition visualisation ────────────────────────────
IMG_IDX   = 42    # index of the image to decompose (change freely)
K_SHOW    = 16    # number of top eigenfaces to display in the reconstruction

# %%
# ── Helper: clip and rescale a float image to [0, 1] for display ──────────────
def normalise(img):
    """Min-max normalise to [0, 1] so imshow renders correctly."""
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-12)

# %%
# ── Decompose a single image in the eigenface basis ───────────────────────────
import numpy as np

# Select the image and centre it (subtract the mean face learned by PCA)
x_orig   = X[IMG_IDX]                            # raw pixel vector, shape (F,)
x_mean   = pca.mean_                             # mean face, shape (F,)
x_centred = x_orig - x_mean                      # centred image

# ── Recover true (unwhitened) eigenvectors u_k ───────────────────────────────
# With whiten=True, scikit-learn stores  u_k / sqrt(λ_k)  in pca.components_.
# Multiplying back by sqrt(λ_k) restores the unit-norm eigenvectors u_k,
# which are needed both for geometrically correct coefficients z_k = <x̃, u_k>
# and for pixel-space reconstruction  x̂ = x̄ + Σ z_k · u_k.
# Skipping this step would produce incorrectly scaled (blurred) reconstructions.
components_unwhitened = pca.components_ * np.sqrt(pca.explained_variance_)[:, None]
#   shape: (K, F)  — each row is the true eigenface u_k

# Unwhitened projection coefficients z_k = x̃ᵀ u_k
z = x_centred @ components_unwhitened.T          # shape (K,)

# Sort by |z_k|^2 (contribution to this specific image)
order      = np.argsort(z**2)[::-1]              # indices, most to least important
z_sorted   = z[order]
ef_sorted  = components_unwhitened[order]        # eigenfaces in contribution order

# %%
# ── Figure 3a – Progressive reconstruction as K increases ─────────────────────
# We show the original image and reconstructions using 1, 2, 4, 8, K_SHOW, K
# components, giving an intuitive sense of how each eigenface adds detail.
checkpoints = [1, 2, 4, 8, K_SHOW, K]
n_panels    = 1 + len(checkpoints)               # original + one panel per K value

fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 3.2))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Progressive eigenface reconstruction  |  image index {IMG_IDX}  "
    f"|  true label: {target_names[y[IMG_IDX]]}",
    fontsize=11, y=1.02
)

# Panel 0: original image
axes[0].imshow(normalise(x_orig.reshape(h, w)), cmap='gray')
axes[0].set_title("Original", fontsize=9)
axes[0].axis('off')

# Panels 1…n: partial reconstructions
for ax, k in zip(axes[1:], checkpoints):
    # Reconstruct using the top-k most-contributing eigenfaces for THIS image
    x_rec = x_mean + (z_sorted[:k] @ ef_sorted[:k])

    # Relative reconstruction error
    err = np.linalg.norm(x_centred - (x_rec - x_mean))**2 / (np.linalg.norm(x_centred)**2 + 1e-12)

    ax.imshow(normalise(x_rec.reshape(h, w)), cmap='gray')
    ax.set_title(f"K={k}\nε={err:.2f}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
# ── Figure 3b – Top-K_SHOW contributing eigenfaces and their coefficients ─────
fig, axes = plt.subplots(2, K_SHOW, figsize=(2.0 * K_SHOW, 5.0))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"Top-{K_SHOW} eigenfaces by contribution  |  image index {IMG_IDX}",
    fontsize=11, y=1.01
)

for col in range(K_SHOW):
    # Top row: eigenface image
    ef_img = normalise(ef_sorted[col].reshape(h, w))
    axes[0, col].imshow(ef_img, cmap='gray')
    axes[0, col].set_title(f"rank {col+1}\nz={z_sorted[col]:.1f}", fontsize=8)
    axes[0, col].axis('off')

    # Bottom row: weighted contribution  z_k * u_k  (what it adds to the face)
    contrib = normalise((z_sorted[col] * ef_sorted[col]).reshape(h, w))
    axes[1, col].imshow(contrib, cmap='gray')
    axes[1, col].set_title(f"z·u", fontsize=8)
    axes[1, col].axis('off')

plt.tight_layout()
plt.show()

# %%
# ── Figure 3c – Coefficient magnitude bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor('white')

colors = ['steelblue' if i < K_SHOW else 'lightgrey' for i in range(K)]
ax.bar(range(1, K + 1), z_sorted**2, color=colors, edgecolor='none')
ax.axvline(K_SHOW + 0.5, color='tomato', linewidth=1.5, linestyle='--',
           label=f'K_SHOW = {K_SHOW}')
ax.set_xlabel('Rank (sorted by contribution $z_k^2$)', fontsize=11)
ax.set_ylabel('$z_k^2$  (squared coefficient)', fontsize=11)
ax.set_title(
    f"Eigenface coefficient spectrum — image {IMG_IDX} "
    f"({target_names[y[IMG_IDX]]})",
    fontsize=11
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# %%
# ── Numerical summary of the decomposition ────────────────────────────────────
cumulative_energy = np.cumsum(z_sorted**2) / (np.linalg.norm(x_centred)**2 + 1e-12)

print(f"Image index : {IMG_IDX}  |  True label : {target_names[y[IMG_IDX]]}")
print(f"{'Rank':<6} {'|z_k|':>8} {'z_k^2':>10} {'Cum. energy':>14}")
print("─" * 42)
for rank in range(K_SHOW):
    print(f"{rank+1:<6} {abs(z_sorted[rank]):>8.3f} "
          f"{z_sorted[rank]**2:>10.3f} {cumulative_energy[rank]:>13.1%}")
print("─" * 42)
print(f"  Total energy captured by top-{K_SHOW}: {cumulative_energy[K_SHOW-1]:.1%}")
print(f"  Total energy captured by all  K={K}: {cumulative_energy[-1]:.1%}")

# %% [markdown]
# ---
# ## 4 – SVM Classifier with Hyperparameter Tuning
#
# ### Model
#
# We train a **soft-margin SVM** with an RBF kernel on the PCA-projected
# training set.  The class-frequency imbalance in LFW (some people have far
# more images than others) is handled by setting `class_weight='balanced'`,
# which rescales each class's penalty:
#
# $$
# C_i = C \cdot \frac{N}{C \cdot N_i},
# $$
#
# where $N_i$ is the number of training samples for class $i$.
#
# ### Grid Search
#
# Two hyperparameters jointly control model complexity:
#
# | Parameter | Role |
# |-----------|------|
# | $C$ | Soft-margin penalty — trades off margin width against training error |
# | $\gamma$ | RBF bandwidth — controls locality of the kernel |
#
# We perform an exhaustive **5-fold stratified cross-validated grid search**
# over the Cartesian product of candidate values, selecting the pair
# $(C^*, \gamma^*)$ that maximises mean validation accuracy.
# 

# %%
# ── Hyperparameter grid ────────────────────────────────────────────────────────
param_grid = {
    'C':     [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1],
}

# ── Grid search with 5-fold cross-validation ──────────────────────────────────
print("Running grid search (5-fold CV) …")
t0 = time()
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'),
    param_grid,
    cv=5,
    n_jobs=-1,           # use all available CPU cores
    verbose=0
)
clf.fit(X_train_pca, y_train)
print(f"Done in {time() - t0:.2f}s")

# ── Best hyperparameters found ─────────────────────────────────────────────────
print(f"\nBest estimator : {clf.best_estimator_}")
print(f"Best CV score  : {clf.best_score_:.4f}")

# %% [markdown]
# ---
# ## 5 – Evaluation on the Test Set
#
# ### Metrics
#
# For a $C$-class problem we report:
#
# - **Per-class precision, recall, and F1-score** via `classification_report`.
#   For class $c$:
#   $$
#   \text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad
#   \text{Recall}_c    = \frac{TP_c}{TP_c + FN_c}, \qquad
#   F_1^{(c)}          = 2\,\frac{\text{Prec}_c \cdot \text{Rec}_c}{\text{Prec}_c + \text{Rec}_c}.
#   $$
#
# - **Confusion matrix** $\mathbf{M} \in \mathbb{Z}^{C \times C}$, where
#   $M_{ij}$ counts how many samples of true class $i$ were predicted as class
#   $j$.  A perfect classifier yields a diagonal matrix.
# 

# %%
# ── Predictions on the held-out test set ──────────────────────────────────────
print("Evaluating on the test set …")
y_pred = clf.predict(X_test_pca)

# %%
# ── Per-class report ──────────────────────────────────────────────────────────
print("\n── Classification report ─────────────────────────────")
print(classification_report(y_test, y_pred, target_names=target_names))

# ── Confusion matrix ──────────────────────────────────────────────────────────
print("── Confusion matrix ──────────────────────────────────")
cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
print(cm)

# %% [markdown]
# ---
# ## 6 – Qualitative Results: Predicted vs. True Labels
#
# We display a random subset of test images annotated with both the **predicted**
# identity and the **ground-truth** identity.  Correct predictions appear as
# matching names; misclassifications reveal the face pairs the model finds most
# confusable — typically subjects with similar lighting, pose, or expression
# in the training set.
# ---

# %%
# ── Helper: build per-image prediction caption ────────────────────────────────
def make_title(y_pred, y_test, target_names, i):
    """Return a two-line caption showing predicted and true last name."""
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    match = "✓" if pred_name == true_name else "✗"
    return f"{match} pred: {pred_name}\n   true: {true_name}"

# %%
# ── Figure 3 – Gallery of predictions on the test set ─────────────────────────
prediction_titles = [
    make_title(y_pred, y_test, target_names, i)
    for i in range(len(y_pred))
]
plot_gallery(X_test, h, w, titles=prediction_titles, n_row=4, n_col=6)

# %% [markdown]
# ---
# ## Summary
#
# | Stage | Method | Key parameters |
# |-------|--------|----------------|
# | Dimensionality reduction | PCA (randomised SVD) + whitening | $K = 50$ components |
# | Classification | RBF-SVM | $C, \gamma$ via 5-fold grid search |
#
# **Key takeaways:**
#
# 1. **PCA is essential here.** Raw pixel space ($F \approx 1{,}850$) is large
#    relative to the training set size; PCA compresses features by $\sim 37\times$
#    while retaining $> 80\%$ of variance, dramatically reducing SVM training
#    time and improving generalisation.
#
# 2. **Eigenfaces encode global face structure.** The first eigenfaces capture
#    lighting and overall shape; later ones encode finer, identity-specific
#    details.  The explained-variance labels in Figure 2 confirm this decay.
#
# 3. **The RBF-SVM generalises well** in the compressed PCA space.  Grid search
#    is crucial: a poor choice of $C$ or $\gamma$ can degrade accuracy by
#    10–20 percentage points.
#
# 4. **Class imbalance requires attention.** `class_weight='balanced'` prevents
#    the model from being biased towards the over-represented subjects
#    (e.g. George W. Bush has far more images than others in LFW).
#
# 5. **Failure modes** visible in Figure 3 are typically caused by unusual
#    lighting or pose not well represented in the training set — limitations
#    inherent to a purely appearance-based (no geometry, no depth) approach.
# 

