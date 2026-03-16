#%%
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

#conda install -c conda-forge pymc
import pymc as pm
import arviz as az

# %%
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
size = 200
w0 = 1
w1 = 2
sigma = .5
x = np.linspace(0,1,size)
regression_line = w0+w1*x
y = regression_line + np.random.normal(scale=sigma, size= size)
data = dict(x=x, y=y)

# %%
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.scatter(x, y, marker='.', s=80, color=COLORS[2],label='dataset')
ax.plot(x, regression_line, color=COLORS[1], label='regression line', lw=2.)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Dataset and underlying model', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.show()

# %% [markdown]
# ### Modello bayesiano gerarchico utilizzato

# %% [markdown]
# ![Diagramma modello regressione lineare gaussiana.](./lin_regr.png)

# %% [markdown]
# $t_i$ viene assunto avere distribuzione gaussiana con valore atteso $\theta^T\overline{x}_i$ e deviazione standard $\sigma$. I coefficienti in $\theta$ si assumono estratti da una distribuzione gaussiana di media $M$ e deviazione standard $S$. $\sigma$ si assume invece estratto da una distribuzione esponenziale con parametro $L$

# %%
M = 0
S = 20
L = 1

# %%
fig = plt.figure(figsize=(16, 4))
dist = stats.norm(loc=M, scale=S)
ax = fig.add_subplot(121)
xx = np.linspace(-20,20, 100)
ax.plot(xx, dist.pdf(xx))
plt.title('Normal, $\mu=${0:5.2f}, $\sigma=${1:5.2f}'.format(M,S))
ax = fig.add_subplot(122)
dist1 = stats.expon(scale=1.0/L)
xx = np.linspace(0,5, 100)
ax.plot(xx, dist1.pdf(xx))
plt.title('Exponential, $\lambda=${0:5.2f}'.format(1.0/L))
plt.show()

# %% [markdown]
# Definizione del modello in Pymc

# %%
with pm.Model() as model:
    # distribuzioni variabili random originali nel modello
    sigma = pm.Exponential('sigma', lam=L)
    theta_0 = pm.Normal('theta_0', mu=M, sigma=S)
    theta_1 = pm.Normal('theta_1', mu=M, sigma=S)
    # distribuzione della variabile random di output, di cui si osservano le instanziazioni nei dati
    y = pm.Normal('y', mu=theta_0+theta_1*x, sigma=sigma, observed=data['y'])

# %% [markdown]
# Sampling

# %%
trace = pm.sample(draws=5000, model=model, chains=2)

#%%
print("\nRiassunto dei risultati:")
print(az.summary(trace, var_names=["theta_0"]))
print(az.summary(trace, var_names=["theta_1"]))
print(az.summary(trace, var_names=["sigma"]))
# %%
post = trace.posterior
theta_0_samples = post['theta_0'].values.flatten()
theta_1_samples = post['theta_1'].values.flatten()

# %%
plt.figure(figsize=(8, 5))

# Plot dell'istogramma
plt.hist(theta_0_samples, bins=50, density=True, color='skyblue', alpha=0.7, label='Posterior samples')

# Aggiungiamo una linea per la media
media = theta_0_samples.mean()
plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')

plt.title('Distribuzione a posteriori di $\\theta_0$')
plt.xlabel('Valore di $\\theta_0$')
plt.ylabel('Densità')
plt.legend()
plt.show()

# %%
az.plot_trace(trace, var_names=['sigma', 'theta_0', 'theta_1'])

# %%
# Mostra la distribuzione con l'intervallo di credibilità (HDI)
az.plot_posterior(trace, var_names=['sigma', 'theta_0', 'theta_1'], hdi_prob=0.95)

# %%
az.plot_forest(trace, var_names=['sigma', 'theta_0', 'theta_1'], combined=True)

# %%


# Crea il grafico
plt.scatter(data['x'], data['y'], color='black', alpha=0.5, label='Dati osservati')

# Disegna, ad esempio, le prime 100 rette campionate
for i in range(100):
    y_line = theta_0_samples[i] + theta_1_samples[i] * data['x']
    plt.plot(data['x'], y_line, color='blue', alpha=0.05)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Incertezza della Regressione Bayesiana')
plt.show()

# %%
#pm.save_trace(trace, 'linregr.trace', overwrite=True)
az.to_netcdf(trace, filename="linregr.nc")
#with model:
#  trace = pm.load_trace('linregr.trace') 

#trace = az.from_netcdf("linregr.nc")

# %%
trace.posterior.isel(draw=slice(0, 100))["theta_0"][0].values

# %%
plt.figure(figsize=(16,8))
az.plot_trace(trace[100:], lines={'theta_0':w0, 'theta_1':w1}, combined=True)
plt.tight_layout()
plt.show()

#%%
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.kdeplot(theta_0_samples, fill=True, burn=1000, color='purple')
plt.title('Densità stimata di $\\theta_0$')
plt.show()
# %%
fig=plt.figure(figsize=(16,6))
ax=sns.distplot(trace.get_values('theta_0', burn=1000, combine=False)[0])
sns.distplot(trace.get_values('theta_0', burn=1000, combine=False)[1])
ax.axvline(w0, color=colors[3], label='w0')
plt.title(r'$p(\theta_0)$', fontsize=16)
plt.legend()
plt.show()

# %%
fig=plt.figure(figsize=(16,6))
ax=sns.distplot(trace.get_values('theta_1', burn=1000, combine=False)[0])
sns.distplot(trace.get_values('theta_1', burn=1000, combine=False)[1])
ax.axvline(w1, color=colors[3], label='w1')
plt.title(r'$p(\theta_1)$', fontsize=16)
plt.legend()
plt.show()

# %%
fig=plt.figure(figsize=(16,6))
ax=sns.distplot(trace.get_values('sigma', burn=1000, combine=False)[0])
sns.distplot(trace.get_values('sigma', burn=1000, combine=False)[1])
#ax.axvline(sigma, color=colors[3], label='$\sigma$')
plt.title(r'$p(\sigma)$', fontsize=16)
plt.legend()
plt.show()

# %%
plt.figure(figsize=(16, 10))
plt.scatter(data['x'], data['y'], marker='x', color=colors[0],label='sampled data')
t0 = []
t1 = []
for i in range(100):
    ndx = np.random.randint(0, len(trace))
    theta_0, theta_1 = trace[ndx]['theta_0'], trace[ndx]['theta_1']
    t0.append(theta_0)
    t1.append(theta_1)
    p = theta_0+theta_1*data['x']
    plt.plot(x, p, c=colors[3], alpha=.1)
plt.plot(data['x'], regression_line, color=colors[1], label='retta di regressione', lw=3.)
theta_0_mean = np.array(t0).mean()
theta_1_mean = np.array(t1).mean()
plt.plot(data['x'], theta_0_mean+theta_1_mean*data['x'], color=colors[8], label='retta di regressione da media su posterior', lw=3.)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Rette di regressione da posterior', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.show()

# %%
x_ = 3.9
x_test = np.array([0,x_])
x_shared.set_value(x_test)

# %%
ppc = pm.sample_posterior_predictive(trace, model=model, samples=1000)

# %%
fig = plt.subplots(figsize=(12, 6))
ax = sns.distplot(ppc['y'][:,1], color=colors[1])
ax.axvline(ppc['y'][:,1].mean())
ax.set(title='Posterior predictive for x={0:5.2f}'.format(x_), xlabel='y', ylabel='p(y)');

# %% [markdown]
# ![Diagramma modello regressione lineare t-Student.](assets/lin_regr_1.png)

# %%
fig = plt.figure(figsize=(16, 4))
dist = stats.halfcauchy()
ax = fig.add_subplot(121)
xx = np.linspace(0,5, 100)
ax.plot(xx, dist.pdf(xx), color=colors[1], label='Half Cauchy')
ax.plot(xx, stats.expon.pdf(xx), label='Exponential')
plt.legend()
ax = fig.add_subplot(122)
dist1 = stats.t(2)
xx = np.linspace(-5,5, 100)
ax.plot(xx, dist1.pdf(xx), color=colors[1], label='Student')
ax.plot(xx, stats.norm.pdf(xx),label='Gaussian')
plt.legend()
plt.show()

# %%
with pm.Model() as model_1:
    # a priori
    sigma = pm.HalfCauchy('sigma', beta=1)
    theta_0 = pm.Normal('theta_0', mu=0, sd=20)
    theta_1 = pm.Normal('theta_1', mu=0, sd=20)
    # likelihood
    likelihood = pm.StudentT('y', mu=theta_0+theta_1*x, sd=sigma, nu=1.0, observed=y)
    trace_1 = pm.sample(3000)

# %%
pm.save_trace(trace_1, 'linregr1.trace', overwrite=True)
#with model_1:
#  trace_1 = pm.load_trace('linregr1.trace') 

# %%
plt.figure(figsize=(16,8))
pm.traceplot(trace_1[100:], lines={'theta_0':w0, 'theta_1':w1}, combined=True)
plt.tight_layout()
plt.show()

# %%
fig = plt.figure(figsize=(12,4))
ax = sns.distplot(trace_1['theta_0'], color=colors[0])
ax.axvline(w0, color=colors[1], label='True value')
plt.title(r'$p(\theta_0)$', fontsize=16)
plt.legend()
plt.show()

# %%
fig = plt.figure(figsize=(12,4))
ax = sns.distplot(trace_1['theta_1'], color=colors[0])
ax.axvline(w1, color=colors[1], label='True value')
plt.title(r'$p(\theta_1)$', fontsize=16)
plt.legend()
plt.show()

# %%
fig = plt.figure(figsize=(12,4))
ax = sns.distplot(trace_1['sigma'], color=colors[0])
plt.title(r'$p(\sigma)$', fontsize=16)
plt.show()

# %%
plt.figure(figsize=(16, 10))
plt.scatter(x, y, marker='x', color=colors[0],label='sampled data')
t0 = []
t1 = []
for i in range(100):
    ndx = np.random.randint(0, len(trace_1))
    theta_0, theta_1 = trace_1[ndx]['theta_0'], trace_1[ndx]['theta_1']
    t0.append(theta_0)
    t1.append(theta_1)
    p = theta_0+theta_1*x 
    plt.plot(x, p, c=colors[3], alpha=.1)
plt.plot(x, regression_line, color=colors[1], label='true regression line', lw=3.)
theta_0_mean = np.array(t0).mean()
theta_1_mean = np.array(t1).mean()
plt.plot(x, theta_0_mean+theta_1_mean*x, color=colors[8], label='average regression line', lw=3.)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Posterior predictive regression lines', fontsize=16)
plt.legend(loc=0, fontsize=14)
plt.show()

# %%



