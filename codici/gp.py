
# %% [markdown]
# ### Regressione mediante processo gaussiano

# %%
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')


# %%
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# plt.style.use("ggplot")                         # clean grid-based style
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
# Generazione del dataset secondo la funzione $y=\sin{(x-2.5)^2}$

# %%
rng = np.random.RandomState(4)
X_tot = rng.uniform(0, 5, 50).reshape(-1,1)
y_tot = np.sin((X_tot[:,0] - 2.5) ** 2)

# %%
x = np.linspace(0,5,100)
y = np.sin((x - 2.5) ** 2)
plt.figure(figsize=(16, 8))
plt.plot(x,y, color=COLORS[1])
plt.scatter(X_tot,y_tot, color=COLORS[4],zorder=9)
plt.show()

# %% [markdown]
# Definizione di kernel RBF con parametri $\alpha$ e $l$ 

# %%
alpha = 1
l = .5
kernel = alpha * RBF(length_scale=l, length_scale_bounds=(1e-1, 10.0))

# %% [markdown]
# Creazione regressore corrispondente, con $\sigma$ misura della varianza dei dati assunta nella likelihood

# %%
sigma = .1
gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma)

# %% [markdown]
# Plot di insieme di funzioni generate dal processo, a priori rispetto al dataset, e della relativa media e deviazione standard

# %%
# Plot prior
plt.figure(figsize=(16, 8))
x = np.linspace(0, 5, 200)
y_mean, y_std = gp.predict(x.reshape(-1,1), return_std=True)
plt.plot(x, y_mean, color=COLORS[3], lw=2, zorder=1)
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, color=COLORS[3])
y_samples = gp.sample_y(x.reshape(-1,1), 8)
plt.plot(x, y_samples, color=COLORS[0], lw=1.5)
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.title("Prior (kernel:  %s)" % kernel, fontsize=14)
plt.show()

# %% [markdown]
# Apprendimento su un insieme di valori, mediante derivazione della distribuzione delle funzioni a posteriori rispetto ai valori stessi

# %%
n = 10
X_tr = X_tot[:n]
y_tr = y_tot[:n]
gp = gp.fit(X_tr, y_tr)

# %% [markdown]
# Plot di insieme di funzioni generate dal processo, a posteriori rispetto ai valori considerati, con relativa media e deviazione standard

# %%
# Plot posterior
plt.figure(figsize=(16, 8))
x = np.linspace(0, 5, 100)
y_mean, y_std = gp.predict(x.reshape(-1,1), return_std=True)
plt.plot(x, y_mean, color=COLORS[3], lw=2, zorder=9)
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, color=COLORS[3])
y_samples = gp.sample_y(x.reshape(-1,1), 8)
plt.plot(x, y_samples, color=COLORS[0], lw=1.5)
plt.scatter(X_tr[:, 0], y_tr, c=COLORS[4], s=50, zorder=10)
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.title("Posterior (kernel: %s)"
              % (gp.kernel_),
              fontsize=14)
plt.show()

# %% [markdown]
# Apprendimento su i valori del training set

# %%
gp = gp.fit(X_tot, y_tot)

# %% [markdown]
# Plot delle predizioni e della funzione originaria

# %%
plt.figure(figsize=(16, 8))
x = np.linspace(0, 5, 100)
y_mean, y_std = gp.predict(x.reshape(-1,1), return_std=True)
plt.plot(x, y_mean, color=COLORS[3], lw=2, zorder=9, label='Predizioni')
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=COLORS[3])
plt.scatter(X_tot[:, 0], y_tot, c=COLORS[4], s=50, zorder=10)
plt.plot(x,y, color=COLORS[1], label='Funzione')
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.legend(fontsize=10)
plt.title("Posterior (kernel: %s)" % gp.kernel_,fontsize=14)
plt.show()

# %%



