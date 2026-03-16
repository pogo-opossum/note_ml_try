# %% [markdown]
# # Funzioni di costo, rischio e ottimizzazione per gradient descent
#
# Questo notebook introduce i concetti fondamentali della teoria dell'apprendimento
# automatico supervisionato, partendo dalla definizione di rischio e funzione di
# costo fino alle principali varianti dell'algoritmo di discesa del gradiente.
#
# Gli argomenti trattati sono:
#
# 1. **Funzioni di costo e rischio** — come misurare la qualità di un predittore
# 2. **Rischio empirico** — approssimazione del rischio reale sul training set
# 3. **Minimizzazione parametrica del rischio empirico** — famiglie di funzioni e coefficienti
# 4. **Gradient descent** — ricerca analitica e numerica dell'ottimo
# 5. **Batch, Stochastic e Mini-batch GD** — varianti con diversi compromessi tra
#    accuratezza e costo computazionale
# 6. **Momento e Nesterov** — tecniche per accelerare la convergenza
# 7. **Adagrad e Adadelta** — learning rate adattativi per parametro
# 8. **Metodi del secondo ordine** — Newton-Raphson e matrice Hessiana

# %% [markdown]
# ## Import e configurazione grafica

# %%
# Librerie standard di calcolo scientifico e visualizzazione
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scipy
import scipy.special as sp

# %%
# Palette di colori e stile dei grafici
COLORS = [
    "xkcd:dusty blue", "xkcd:dark peach", "xkcd:dark seafoam green",
    "xkcd:dusty purple", "xkcd:watermelon", "xkcd:dusky blue", "xkcd:amber",
    "xkcd:purplish", "xkcd:dark teal", "xkcd:orange", "xkcd:slate",
]

plt.style.use("ggplot")
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

# %%
# Funzione di utilità per caricare i dati: da disco locale o da URL remoto
FILEPATH = "../dataset/"
URL      = "https://tvml.github.io/ml2324/dataset/"

def get_file(filename, local):
    """
    Restituisce il percorso al file richiesto.

    Se `local=True`, accede al filesystem locale; altrimenti scarica il file
    dall'URL del corso e lo salva nella directory corrente.
    """
    if local:
        return FILEPATH + filename
    else:
        urllib.request.urlretrieve(URL + filename, filename)
        return filename

# %%
# Funzione di visualizzazione del dataset di classificazione binaria.
# Mostra i punti delle due classi e, opzionalmente, la retta di separazione
# definita da coefficiente angolare `m` e intercetta `q`.
def plot_ds(data, m=None, q=None):
    """
    Visualizza il dataset di classificazione binaria su piano (x1, x2).

    Parametri
    ----------
    data : DataFrame con colonne x1, x2, t (t in {0,1})
    m    : float, coefficiente angolare della retta di separazione (opzionale)
    q    : float, intercetta della retta di separazione (opzionale)
    """
    fig = plt.figure(figsize=(16, 8))
    minx, maxx = data.x1.min(), data.x1.max()
    x_range = np.linspace(minx - 0.1 * (maxx - minx),
                          maxx + 0.1 * (maxx - minx), 1000)
    ax = fig.gca()
    ax.scatter(data[data.t == 0].x1, data[data.t == 0].x2,
               s=40, edgecolor="k", alpha=0.7, label="Classe 0")
    ax.scatter(data[data.t == 1].x1, data[data.t == 1].x2,
               s=40, edgecolor="k", alpha=0.7, label="Classe 1")
    if m is not None:
        ax.plot(x_range, m * x_range + q, lw=2, color=COLORS[5],
                label="Retta di separazione")
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.title("Dataset", fontsize=12)
    plt.legend()
    plt.show()


# %%
# Funzione di visualizzazione dell'andamento di costo e traiettoria dei parametri
# durante l'ottimizzazione. Utile per confrontare le diverse varianti di GD.
def plot_all(cost_history, m, q, low, high, step):
    """
    Visualizza su due pannelli:
      - Sinistra: andamento del costo al variare delle iterazioni
      - Destra:   traiettoria dei parametri (m, q) nello spazio 2D

    Parametri
    ----------
    cost_history : array (N,1), storia dei valori di costo
    m, q         : array (N,), storia di coefficiente angolare e intercetta
    low, high    : indici di inizio e fine della finestra visualizzata
    step         : passo di campionamento degli indici
    """
    idx = range(low, high, step)
    ch  = cost_history[idx]
    th1 = m[idx]
    th0 = q[idx]

    fig = plt.figure(figsize=(18, 6))

    # Pannello sinistro: costo vs iterazioni
    ax = fig.add_subplot(1, 2, 1)
    c_min, c_max = ch.min(), ch.max()
    dc = 0.1 * (c_max - c_min)
    ax.plot(range(len(ch)), ch, alpha=1, color=COLORS[0], linewidth=2)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, pos: f"{x * step + low:0.0f}")
    )
    plt.xlabel("Iterazioni")
    plt.ylabel("Costo")
    plt.xlim(-0.1 * len(ch), 1.1 * len(ch))
    plt.ylim(c_min - dc, c_max + dc)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Pannello destro: traiettoria (q, m) nello spazio dei parametri
    ax = fig.add_subplot(1, 2, 2)
    p_minx, p_maxx = th0.min(), th0.max()
    p_miny, p_maxy = th1.min(), th1.max()
    dp_x = 0.1 * (p_maxx - p_minx)
    dp_y = 0.1 * (p_maxy - p_miny)
    ax.plot(th0, th1, alpha=1, color=COLORS[1], linewidth=2, zorder=1)
    ax.scatter(th0[-1], th1[-1], color=COLORS[5], marker="o", s=40, zorder=2,
               label="Punto finale")
    plt.xlabel(r"$q$ (intercetta)")
    plt.ylabel(r"$m$ (coeff. angolare)")
    plt.xlim(p_minx - dp_x, p_maxx + dp_x)
    plt.ylim(p_miny - dp_y, p_maxy + dp_y)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## 1. Rischio e funzione di costo
#
# ### Definizione di funzione di costo
#
# Dato un algoritmo che associa a ogni input $x$ una previsione $f(x)$,
# la qualità della previsione si misura mediante una **funzione di costo**
# (o *loss function*) $L(x_1, x_2)$, dove $x_1$ è il valore predetto e
# $x_2$ è il valore corretto. Il valore $L(f(x), y)$ quantifica quanto
# costa prevedere $f(x)$ quando il valore reale è $y$.
#
# ### Rischio atteso
#
# Poiché il costo dipende dalla coppia $(x, y)$, una valutazione globale
# richiede di considerarne il **valore atteso** rispetto alla distribuzione
# congiunta $p(x, y)$:
#
# $$
# \mathcal{R}(f) = \mathbb{E}_p[L(f(x), y)]
#   = \int_{D_x}\int_{D_y} L(f(x), y)\, p(x, y)\, dx\, dy
# $$
#
# Il rischio $\mathcal{R}(f)$ misura quindi il costo atteso delle predizioni
# di $f$ quando:
#
# 1. $x$ viene estratto dalla distribuzione marginale
#    $p(x) = \int_{D_y} p(x,y)\, dy$
# 2. il target $y$ viene estratto dalla distribuzione condizionata
#    $p(y \mid x) = p(x,y)/p(x)$
# 3. il costo è valutato dalla funzione $L$
#
# **Nota importante**: non si assume che esista una relazione funzionale
# deterministica tra $x$ e $y$. Si ammette invece che a uno stesso valore
# $x$ possano corrispondere valori diversi di $y$ con probabilità non nulla.
# Questo è il modo naturale di modellare la presenza di rumore nelle
# osservazioni: il modello probabilistico $p(y \mid x)$ cattura questa
# incertezza intrinseca.
#
# ### Esempio: previsione della pioggia
#
# Vogliamo prevedere la pioggia (T/F) in base alle condizioni del cielo
# (S=sereno, N=nuvoloso, C=coperto). La funzione di costo $L: \{T,F\}^2 \to \mathbb{R}$
# codifica le preferenze dell'utente.
#
# **Funzione di costo simmetrica $L_1$** (penalizzazione uguale per i due errori):
#
# | $\hat{y}$ / $y$ | T | F |
# |:---:|:---:|:---:|
# | T | 0 | 1 |
# | F | 1 | 0 |
#
# **Funzione di costo asimmetrica $L_2$** (bagnarsi è molto peggio che portare
# l'ombrello inutilmente):
#
# | $\hat{y}$ / $y$ | T | F |
# |:---:|:---:|:---:|
# | T | 0 | 1 |
# | F | 25 | 0 |
#
# La distribuzione congiunta $p(x, y)$ sull'esempio è:
#
# | $x$ / $y$ | T | F |
# |:---:|:---:|:---:|
# | S | .05 | .20 |
# | N | .25 | .25 |
# | C | .20 | .05 |
#
# Consideriamo due predittori $f_1$ e $f_2$:
#
# | $x$ | $f_1(x)$ | $f_2(x)$ |
# |:---:|:---:|:---:|
# | S | F | F |
# | N | F | T |
# | C | T | T |
#
# Con $L_1$: $\mathcal{R}(f_1) = 0.65$, $\mathcal{R}(f_2) = 0.40$ → $f_2$ è preferibile.
# Con $L_2$: $\mathcal{R}(f_1) = 1.55$, $\mathcal{R}(f_2) = 7.55$ → $f_1$ è preferibile.
#
# Questo esempio illustra due principi fondamentali:
#
# - La scelta della funzione di costo determina quale predittore è ottimale:
#   **non esiste un predittore universalmente migliore**, ma solo il migliore
#   rispetto a un dato costo.
# - Anche la distribuzione $p(x,y)$ influenza il confronto: una diversa
#   distribuzione, con la stessa $L_1$, potrebbe rendere $f_1$ preferibile a $f_2$.
#
# Nella pratica, sia la funzione di costo che la distribuzione richiedono
# scelte progettuali consapevoli: la funzione di costo deve riflettere le
# priorità dell'applicazione, e la distribuzione è quella (sconosciuta) del
# contesto reale.

# %% [markdown]
# ---
# ## 2. Rischio empirico
#
# Poiché la distribuzione reale $p(x, y)$ è sconosciuta, il calcolo diretto
# del rischio $\mathcal{R}(f)$ è impossibile. Si ricorre quindi a un'
# **approssimazione campionaria**: il **rischio empirico**, che stima il
# valore atteso sostituendo la distribuzione ignota con la media aritmetica
# sul training set $\mathcal{D} = \{(x_1,y_1), \ldots, (x_n,y_n)\}$:
#
# $$
# \overline{\mathcal{R}}(f;\, \mathcal{D})
#   = \frac{1}{n} \sum_{i=1}^{n} L\!\left(f(x_i),\, y_i\right)
# $$
#
# Il predittore ottimo viene scelto minimizzando il rischio empirico
# sull'insieme di funzioni disponibile $\mathcal{F}$:
#
# $$
# f^* = \underset{f \in \mathcal{F}}{\arg\min}\;
#       \overline{\mathcal{R}}(f;\, \mathcal{D})
# $$
#
# ### Quando il rischio empirico approssima bene il rischio reale?
#
# La speranza è che minimizzare $\overline{\mathcal{R}}$ dia risultati
# simili a minimizzare $\mathcal{R}$. Questo dipende da quattro fattori:
#
# 1. **Dimensione del training set**: per la legge dei grandi numeri,
#    $\overline{\mathcal{R}}(f; \mathcal{D}) \to \mathcal{R}(f)$ al crescere
#    di $n$, per ogni $f$ fissata.
# 2. **Complessità della distribuzione $p(x,y)$**: distribuzioni più
#    complesse richiedono più dati per essere ben approssimate.
# 3. **Forma della funzione di costo $L$**: se $L$ assegna penalità molto
#    alte a eventi rari, il rischio empirico può sottostimare sistematicamente
#    il rischio reale.
# 4. **Capacità della classe di funzioni $\mathcal{F}$**: classi troppo ricche
#    (elevata capacità di Vapnik-Chervonenkis) richiedono molti dati per
#    garantire che il minimo empirico sia vicino al minimo reale, pena
#    l'overfitting. D'altro canto, classi troppo ristrette portano a
#    underfitting: il minimo su $\mathcal{F}$ può essere molto lontano dal
#    minimo assoluto.

# %% [markdown]
# ---
# ## 3. Minimizzazione parametrica del rischio empirico
#
# ### Famiglie parametriche di funzioni
#
# Nella pratica, l'insieme $\mathcal{F}$ è definito **parametricamente**:
# $\mathcal{F} = \{f(x;\,\theta) \mid \theta \in D_\theta\}$, dove il
# vettore di parametri $\theta \in \mathbb{R}^m$ identifica la funzione
# specifica all'interno della classe strutturale.
#
# Un esempio paradigmatico è la **regressione lineare**: la classe è formata
# da tutte le funzioni lineari $f_\mathbf{w}(x) = w_0 + w_1 x_1 + \cdots +
# w_d x_d$, e il parametro è $\theta = \mathbf{w} = (w_0, \ldots, w_d)$.
#
# ### Rischio empirico come funzione di $\theta$
#
# Fissata la famiglia $\mathcal{F}$, il rischio empirico diventa una funzione
# ordinaria di $\theta$:
#
# $$
# \overline{\mathcal{R}}(\theta;\, \mathcal{D})
#   = \frac{1}{n} \sum_{i=1}^{n} L\!\left(f(x_i;\,\theta),\, y_i\right)
# $$
#
# e il problema di apprendimento si riduce a un problema di ottimizzazione:
#
# $$
# \theta^* = \underset{\theta \in D_\theta}{\arg\min}\;
#            \overline{\mathcal{R}}(\theta;\, \mathcal{D})
# $$
#
# da cui si ricava $f^* = f(x;\,\theta^*)$.
#
# ### Come trovare il minimo?
#
# Sono disponibili due strategie principali:
#
# **Soluzione analitica** — si cercano i $\theta$ per cui tutte le derivate
# parziali si annullano simultaneamente:
#
# $$
# \frac{\partial \overline{\mathcal{R}}(\theta;\, \mathcal{D})}{\partial \theta_i}
# \Bigg|_{\theta = \theta^*} = 0, \qquad i = 1, \ldots, m
# $$
#
# Questo sistema di $m$ equazioni in $m$ incognite ammette soluzione in
# forma chiusa solo in casi particolari (es. regressione lineare con loss
# quadratica). In generale è necessario ricorrere a metodi numerici.
#
# **Metodi numerici iterativi** — si parte da un valore iniziale $\theta^{(0)}$
# e si aggiornano iterativamente i parametri seguendo una direzione che
# riduce il costo. La **discesa del gradiente** è il metodo di gran lunga
# più utilizzato in machine learning.

# %% [markdown]
# ---
# ## 4. Discesa del gradiente (Gradient Descent)
#
# La discesa del gradiente minimizza la funzione obiettivo
# $J(\theta)$, con $\theta \in \mathbb{R}^d$, aggiornando iterativamente
# $\theta$ nella direzione opposta al gradiente, che è la direzione di
# massima decrescita locale:
#
# $$
# \theta^{(k+1)} = \theta^{(k)} - \eta\, \nabla J(\theta^{(k)})
# $$
#
# Il parametro $\eta > 0$ è detto **learning rate** e controlla l'ampiezza
# dei passi:
# - $\eta$ troppo piccolo → convergenza lentissima
# - $\eta$ troppo grande → oscillazioni o divergenza
#
# Geometricamente, l'algoritmo simula lo spostamento di un punto sulla
# superficie di $J(\theta)$, sempre in discesa lungo la direzione di
# massima pendenza, fino a raggiungere un minimo locale (globale se $J$
# è convessa).
#
# ### Struttura additiva e additività del gradiente
#
# In machine learning, la funzione obiettivo è quasi sempre additiva
# rispetto agli elementi del dataset:
#
# $$
# J(\theta;\, \mathcal{D}) = \frac{1}{n} \sum_{i=1}^{n} J(\theta;\, x_i)
# $$
#
# Per linearità della derivata, anche il gradiente è additivo:
#
# $$
# \nabla J(\theta;\, \mathcal{D}) = \frac{1}{n} \sum_{i=1}^{n} \nabla J(\theta;\, x_i)
# $$
#
# Questa proprietà è alla base delle tre varianti dell'algoritmo, che
# differiscono per la **quantità di dati utilizzata** a ogni iterazione
# per calcolare il gradiente.

# %% [markdown]
# ---
# ## 5. Esempio applicativo: logistic regression su dataset 2D
#
# Per illustrare concretamente le varianti di gradient descent, utilizziamo
# un problema di **classificazione binaria** su un dataset bidimensionale.
# Il metodo scelto è la **regressione logistica**, che cerca un iperpiano
# (in questo caso una retta) di separazione minimizzando la *cross-entropy*.
#
# ### Modello
#
# Il predittore è la funzione sigmoide applicata alla combinazione lineare
# delle feature:
#
# $$
# y = \sigma(\theta^\top x)
#   = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2)}}
#   \in (0, 1)
# $$
#
# Il valore $y$ viene interpretato come la probabilità stimata che
# l'elemento $x$ appartenga alla classe positiva ($t=1$).
# La retta di separazione si ottiene ponendo $\sigma = 0.5$, ovvero
# $\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$, da cui:
#
# $$
# x_2 = -\frac{\theta_1}{\theta_2} x_1 - \frac{\theta_0}{\theta_2}
#      \equiv m\, x_1 + q
# $$
#
# ### Funzione di costo: cross-entropy
#
# Per un singolo elemento $(x, t)$ la **cross-entropy** è:
#
# $$
# J(\theta, x) = -\Bigl[t \log y + (1-t) \log(1-y)\Bigr]
# $$
#
# Questa funzione vale $0$ quando la predizione è perfetta ($y=t$),
# e diverge quando la predizione è completamente errata. A differenza
# della loss quadratica, la cross-entropy è convessa nei parametri
# $\theta$ per modelli logistici, garantendo l'assenza di minimi locali.
#
# Il **rischio empirico** sul dataset è la media delle cross-entropy
# individuali:
#
# $$
# J(\theta, \mathcal{D}) = -\frac{1}{n} \sum_{i=1}^{n}
#   \Bigl[t_i \log \sigma(\theta^\top x_i)
#         + (1-t_i) \log(1 - \sigma(\theta^\top x_i))\Bigr]
# $$
#
# ### Gradiente della cross-entropy
#
# Il gradiente rispetto ai singoli parametri ha una forma elegante e
# sorprendentemente semplice:
#
# $$
# \frac{\partial J(\theta, x)}{\partial \theta_i}
#   = -\bigl(t - \sigma(\theta^\top x)\bigr) x_i, \quad i = 1,\ldots,d
# $$
#
# Il termine $(t - y)$ è il **residuo** tra target e predizione: il gradiente
# è grande quando il modello sbaglia di molto, piccolo quando sbaglia poco.
# Il gradiente del rischio empirico è quindi:
#
# $$
# \frac{\partial J(\theta, \mathcal{D})}{\partial \theta_i}
#   = -\frac{1}{n} \sum_{j=1}^{n} \bigl(t_j - \sigma(\theta^\top x_j)\bigr) x_{ji}
# $$

# %%
# Caricamento del dataset di classificazione binaria
data = pd.read_csv(
    get_file("testSet.txt", local=False),
    sep=r"\s+", header=None, names=["x1", "x2", "t"]
)

# %%
# Visualizzazione del dataset: due classi nel piano (x1, x2)
plot_ds(data)

# %%
# Preparazione delle matrici di input e target per l'ottimizzazione.
# La colonna di 1 in testa a X rappresenta il termine bias theta_0.
n         = len(data)
nfeatures = len(data.columns) - 1

X = np.array(data[["x1", "x2"]])
t = np.array(data["t"]).reshape(-1, 1)
X = np.column_stack((np.ones(n), X))   # X ha forma (n, 3): [1, x1, x2]

# %%
# Funzione sigmoide: mappa la combinazione lineare theta^T x in (0,1).
# Si usa scipy.special.expit per stabilità numerica (evita overflow in exp).
def sigma(theta, X):
    """
    Calcola la funzione sigmoide applicata a X @ theta.

    Ritorna
    -------
    y : array (n, 1), valori in (0, 1) — probabilità stimate della classe 1
    """
    return sp.expit(np.dot(X, theta))


# %%
# Funzione di costo: cross-entropy media sul dataset.
# Il clipping a eps evita log(0) quando sigma è numericamente 0 o 1.
def cost(theta, X, t):
    """
    Calcola il rischio empirico cross-entropy su (X, t).

        J = -(1/n) * [t^T log(y) + (1-t)^T log(1-y)]

    Parametri
    ----------
    theta : array (d, 1), parametri attuali
    X     : array (n, d), design matrix con bias
    t     : array (n, 1), target binari

    Ritorna
    -------
    J : float, valore della funzione di costo
    """
    eps = 1e-50
    y   = np.clip(sigma(theta, X), eps, 1.0 - eps)     # clipping per stabilità numerica
    return (-np.dot(np.log(y).T, t)
            - np.dot(np.log(1.0 - y).T, 1 - t))[0][0] / len(X)


# %%
# Gradiente della cross-entropy rispetto a theta.
# Forma compatta: grad = -(1/n) * X^T (t - y)
def gradient(theta, X, t):
    """
    Calcola il gradiente del rischio empirico cross-entropy.

        grad_J = -(1/n) * X^T (t - sigma(X @ theta))

    Ritorna
    -------
    g : array (d, 1), gradiente
    """
    return -np.dot(X.T, (t - sigma(theta, X))) / len(X)


# %%
# Valori ottimali dei parametri (di riferimento per misurare la convergenza)
M_STAR = 0.62595499
Q_STAR = 7.3662299

def convergence_iterations(m, q, tol=1e-2):
    """
    Restituisce il numero di iterazioni necessarie affinché la soluzione
    entri in un intorno di raggio `tol` attorno alla soluzione ottima.
    """
    dist = np.sqrt((M_STAR - m) ** 2 + (Q_STAR - q) ** 2)
    return np.argmin(dist > tol) + 1


# %% [markdown]
# ---
# ## 6. Batch Gradient Descent (BGD)
#
# Nel **Batch Gradient Descent** il gradiente viene calcolato a ogni
# iterazione sull'**intero training set**. L'aggiornamento al passo $k$ è:
#
# $$
# \theta^{(k+1)} = \theta^{(k)} - \eta \sum_{i=1}^{n} \nabla J(\theta^{(k)};\, x_i)
# $$
#
# oppure, per i singoli coefficienti:
#
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)}
#   - \eta \sum_{i=1}^{n} \frac{\partial J(\theta;\, x_i)}{\partial \theta_j}
#     \Bigg|_{\theta = \theta^{(k)}}
# $$
#
# Nel caso della logistic regression, sostituendo il gradiente calcolato
# sopra, si ottiene:
#
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)}
#   + \frac{\eta}{n} \sum_{i=1}^{n} \bigl(t_i - \sigma(\theta^{(k)\top} x_i)\bigr) x_{ij}
# $$
#
# ```python
# for epoch in range(n_epochs):
#     g = sum(gradient(theta, x_i) for x_i in dataset)
#     theta = theta - eta * g
# ```
#
# **Vantaggi**: convergenza regolare e garantita al minimo globale se $J$
# è convessa; aggiornamenti stabili e privi di rumore.
#
# **Svantaggi**: ogni iterazione richiede una passata completa sul dataset,
# il che è computazionalmente proibitivo per dataset di grandi dimensioni.
# In presenza di dati che non entrano in memoria, il metodo è inapplicabile.

# %%
def batch_gd(X, t, eta=0.1, epochs=10000):
    """
    Batch Gradient Descent per la logistic regression.

    A ogni iterazione aggiorna theta usando il gradiente calcolato
    su tutti gli n elementi del training set.

    Parametri
    ----------
    X      : array (n, d), design matrix
    t      : array (n, 1), target
    eta    : float, learning rate
    epochs : int, numero di iterazioni

    Ritorna
    -------
    cost_history   : array (epochs, 1), storia dei valori di costo
    theta_history  : array (epochs, d), storia dei parametri
    m, q           : array (epochs,), storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        theta = theta - eta * gradient(theta, X, t)
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    # Ricava m e q dalla retta di separazione: x2 = -(theta1/theta2)*x1 - theta0/theta2
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m, q


# %%
# Esecuzione di BGD sul dataset
cost_history_bgd, theta_history_bgd, m_bgd, q_bgd = batch_gd(
    X, t, eta=0.1, epochs=100000
)

# %%
# Visualizzazione: andamento del costo e traiettoria dei parametri nelle prime 5000 iterazioni.
# La discesa regolare e monotona del costo è la caratteristica distintiva di BGD.
plot_all(cost_history_bgd, m_bgd, q_bgd, low=0, high=5000, step=10)

# %%
# Numero di iterazioni necessarie per entrare nell'intorno della soluzione ottima
print(f"BGD: iterazioni alla convergenza = {convergence_iterations(m_bgd, q_bgd)}")

# %%
# Retta di separazione finale ottenuta con BGD
plot_ds(data, m_bgd[-1], q_bgd[-1])


# %% [markdown]
# ---
# ## 7. Stochastic Gradient Descent (SGD)
#
# Nel **Stochastic Gradient Descent** il gradiente viene approssimato a
# ogni iterazione usando **un solo elemento** $x_i$ del training set,
# scelto casualmente:
#
# $$
# \theta^{(k+1)} = \theta^{(k)} - \eta\, \nabla J(\theta^{(k)};\, x_i)
# $$
#
# Per la logistic regression:
#
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)}
#   + \eta\, \bigl(t_i - \sigma(\theta^{(k)\top} x_i)\bigr) x_{ij}
# $$
#
# ```python
# for epoch in range(n_epochs):
#     np.random.shuffle(data)
#     for x_i in data:
#         g = gradient(theta, x_i)
#         theta = theta - eta * g
# ```
#
# ### Vantaggi e svantaggi rispetto a BGD
#
# **Pro**: ogni aggiornamento è computazionalmente economico (un solo
# campione). Il numero di aggiornamenti per epoca è $n$ volte maggiore
# che in BGD, il che spesso accelera la convergenza pratica, soprattutto
# nelle fasi iniziali. Inoltre, le oscillazioni intorno al minimo possono
# aiutare a **sfuggire da minimi locali**, rendendo SGD utile per
# funzioni obiettivo non convesse.
#
# **Contro**: il gradiente stimato da un solo campione è un **estimatore
# rumoroso** del gradiente vero. Ne consegue un andamento molto irregolare
# del costo: anziché decrescere monotonamente, oscilla intorno a un trend
# di discesa globale. Questo rende difficile stabilire un criterio di
# convergenza affidabile.
#
# > **Osservazione**: se si considera la sequenza dei valori di costo
# > *al termine di ogni epoca* (anziché a ogni singola iterazione), emerge
# > chiaramente la tendenza di decrescita di fondo, analoga a BGD ma
# > con maggior rumore.

# %%
def stochastic_gd(X, t, eta=0.01, epochs=1000):
    """
    Stochastic Gradient Descent per la logistic regression.

    A ogni iterazione aggiorna theta usando il gradiente calcolato
    su un singolo elemento del training set.

    Parametri
    ----------
    X      : array (n, d), design matrix
    t      : array (n, 1), target
    eta    : float, learning rate
    epochs : int, numero di epoche (ogni epoca processa tutti gli n elementi)

    Ritorna
    -------
    cost_history  : array (epochs*n, 1), storia dei valori di costo
    theta_history : array (epochs*n, d), storia dei parametri
    m, q          : array (epochs*n,), storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        for i in range(n):
            # Residuo per il singolo campione i
            e     = (t[i] - sigma(theta, X[i, :]))[0]
            theta = theta + eta * e * X[i, :].reshape(-1, 1)
            theta_history.append(theta.copy())
            cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m, q


# %%
# Esecuzione di SGD — si usa un learning rate più piccolo rispetto a BGD
# poiché la varianza degli aggiornamenti è più alta
cost_history_sgd, theta_history_sgd, m_sgd, q_sgd = stochastic_gd(
    X, t, eta=0.01, epochs=10000
)

# %%
# Visualizzazione: le oscillazioni del costo sono la firma di SGD.
# Si noti come la traiettoria dei parametri sia molto più caotica rispetto a BGD.
plot_all(cost_history_sgd, m_sgd, q_sgd, low=0, high=150 * n, step=30)

# %%
# Convergenza di SGD
print(f"SGD: iterazioni alla convergenza = {convergence_iterations(m_sgd, q_sgd)}")

# %%
# Retta di separazione finale ottenuta con SGD
plot_ds(data, m_sgd[-1], q_sgd[-1])

# %%
# Visualizzando solo i valori al termine di ogni epoca (step=n),
# emerge la tendenza di decrescita regolare anche in SGD.
plot_all(cost_history_sgd, m_sgd, q_sgd, low=0, high=1000 * n, step=n)


# %% [markdown]
# ---
# ## 8. Mini-batch Gradient Descent (MBGD)
#
# Il **Mini-batch Gradient Descent** è una soluzione di compromesso tra BGD
# e SGD. A ogni iterazione, anziché usare tutti gli $n$ elementi o uno
# solo, si utilizza un sottoinsieme (**mini-batch**) di dimensione fissa $s$.
#
# All'inizio di ogni epoca il dataset viene rimescolato e diviso in
# $\lceil n/s \rceil$ mini-batch. L'aggiornamento per il mini-batch $MB_k$ è:
#
# $$
# \theta^{(k+1)} = \theta^{(k)}
#   - \eta \sum_{x_i \in MB_k} \nabla J(\theta^{(k)};\, x_i)
# $$
#
# Per la logistic regression:
#
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)}
#   + \eta \sum_{x_i \in MB_k} \bigl(t_i - y_i\bigr) x_{ij}
# $$
#
# ```python
# for epoch in range(n_epochs):
#     np.random.shuffle(data)
#     for batch in get_batches(data, batch_size=s):
#         g = sum(gradient(theta, x_i) for x_i in batch)
#         theta = theta - eta * g
# ```
#
# ### Perché MBGD è lo standard per le reti neurali?
#
# - **Bilanciamento varianza/costo**: un mini-batch fornisce una stima del
#   gradiente meno rumorosa di SGD (varianza ridotta di un fattore $s$)
#   ma molto più veloce da calcolare rispetto a BGD.
# - **Sfruttamento del parallelismo hardware**: le moderne GPU eseguono
#   operazioni su matrici in modo massivamente parallelo. Calcolare il
#   gradiente su un mini-batch si traduce in operazioni matriciali efficienti,
#   sfruttando al meglio l'hardware disponibile.
# - **Dimensione tipica**: valori di $s$ tra 32 e 512 rappresentano il
#   compromesso ottimale nella maggioranza delle applicazioni di deep learning.
#
# L'oscillazione del costo è intermedia tra BGD e SGD: tanto più
# pronunciata quanto più piccolo è $s$.

# %%
def mb_gd(X, t, eta=0.01, epochs=1000, minibatch_size=5):
    """
    Mini-batch Gradient Descent per la logistic regression.

    A ogni iterazione calcola il gradiente su un mini-batch di dimensione
    `minibatch_size` e aggiorna theta di conseguenza.

    Parametri
    ----------
    X              : array (n, d), design matrix
    t              : array (n, 1), target
    eta            : float, learning rate
    epochs         : int, numero di epoche
    minibatch_size : int, dimensione di ogni mini-batch

    Ritorna
    -------
    cost_history : array, storia dei valori di costo (un valore per mini-batch)
    m, q         : array, storia di coeff. angolare e intercetta
    """
    n_batches = int(np.ceil(n / minibatch_size))
    idx       = np.arange(n)
    np.random.shuffle(idx)

    theta         = np.zeros((nfeatures + 1, 1))
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        # Processa tutti i mini-batch tranne l'ultimo (che può essere incompleto)
        for k in range(n_batches - 1):
            batch_idx = idx[k * minibatch_size : (k + 1) * minibatch_size]
            g = sum(
                (t[i] - sigma(theta, X[i, :]))[0] * X[i, :]
                for i in batch_idx
            )
            theta = theta + eta * g.reshape(-1, 1)
            theta_history.append(theta.copy())
            cost_history.append(cost(theta, X, t))

        # Ultimo mini-batch (eventualmente più piccolo)
        last_batch_idx = idx[(n_batches - 1) * minibatch_size : n]
        g = sum(
            (t[i] - sigma(theta, X[i, :]))[0] * X[i, :]
            for i in last_batch_idx
        )
        theta = theta + eta * g.reshape(-1, 1)
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, m, q


# %%
# Esecuzione di MBGD con mini-batch di dimensione 5
cost_history_mb, m_mb, q_mb = mb_gd(
    X, t, eta=0.01, epochs=10000, minibatch_size=5
)

# %%
# Visualizzazione: l'oscillazione è intermedia tra BGD (regolare) e SGD (caotica)
plot_all(cost_history_mb, m_mb, q_mb, low=0, high=5000, step=10)

# %%
print(f"MBGD: iterazioni alla convergenza = {convergence_iterations(m_mb, q_mb)}")

# %%
# Retta di separazione finale ottenuta con MBGD
plot_ds(data, m_mb[-1], q_mb[-1])


# %% [markdown]
# ---
# ## 9. Criticità dei metodi base di gradient descent
#
# I tre metodi elementari presentati condividono alcune limitazioni strutturali:
#
# - **Scelta del learning rate $\eta$**: è un iperparametro critico da fissare
#   manualmente. Un $\eta$ troppo piccolo rallenta enormemente la convergenza;
#   un $\eta$ troppo grande provoca oscillazioni intorno al minimo o addirittura
#   divergenza. Non esiste un valore universalmente ottimale.
#
# - **Learning rate fisso nel tempo**: un $\eta$ costante non può adattarsi
#   alle diverse fasi dell'ottimizzazione. Nelle fasi iniziali conviene un
#   passo grande (rapido avvicinamento al minimo), nelle fasi finali un passo
#   piccolo (convergenza fine). Si possono usare schemi di *decay* predefiniti,
#   ma richiedono la specificazione di parametri aggiuntivi.
#
# - **Stesso $\eta$ per tutti i parametri**: la stessa ampiezza del passo
#   viene applicata a tutti i coefficienti $\theta_j$, indipendentemente
#   dalla loro curvatura locale. Questo è inefficiente quando le diverse
#   dimensioni dello spazio dei parametri hanno scale o sensibilità molto diverse.
#
# - **Difficoltà con la non-convessità**: per reti neurali profonde, $J(\theta)$
#   è fortemente non convessa, caratterizzata da numerosi minimi locali e
#   **punti di sella** (dove il gradiente è nullo ma non è un minimo).
#   I metodi base faticano a sfuggire dai punti di sella, spesso circondati
#   da regioni in cui il gradiente è molto piccolo (*flat regions*).
#
# I metodi delle sezioni successive affrontano queste criticità con
# approcci complementari.

# %% [markdown]
# ---
# ## 10. Gradient Descent con Momento
#
# ### Motivazione
#
# I metodi base sono inefficienti in presenza di **valli elongate**: la
# funzione di costo scende lentamente lungo il fondo della valle
# (direzione del minimo) ma ha pareti laterali ripide, per cui il gradiente
# punta quasi perpendicolarmente alla direzione ottimale di discesa.
# Il risultato è un percorso a zig-zag con progressione lenta.
#
# ### Analogia fisica
#
# Il **metodo del momento** si ispira alla meccanica classica: si immagina
# $\theta$ come la posizione di un corpo di massa unitaria che si muove
# sulla superficie di $J(\theta)$ soggetto alla forza peso
# $F(\theta) = -\eta\, \nabla J(\theta)$.
#
# Nella discesa del gradiente standard, lo spostamento a ogni passo dipende
# solo dalla forza *istantanea* (il gradiente nel punto attuale).
# Nel metodo del momento, si introduce la **velocità** $v^{(k)}$:
# lo spostamento dipende dall'inerzia accumulata nei passi precedenti.
#
# ### Regola di aggiornamento
#
# $$
# v^{(k+1)} = \gamma\, v^{(k)} - \eta \sum_{i=1}^{n} \nabla J(\theta^{(k)};\, x_i)
# $$
# $$
# \theta^{(k+1)} = \theta^{(k)} + v^{(k+1)}
# $$
#
# Il parametro $\gamma \in (0,1)$ è il coefficiente di **attrito** (tipicamente
# $\gamma \approx 0.9$): determina quanta "memoria" delle direzioni passate
# viene mantenuta. Con $\gamma = 0$ si ottiene la discesa del gradiente standard.
#
# ### Effetto dell'inerzia
#
# Sviluppando la ricorrenza con $v^{(0)} = 0$, si mostra che lo spostamento
# totale è una **media esponenzialmente pesata** dei gradienti passati:
#
# $$
# \theta^{(k+1)} = \theta^{(k)}
#   - \eta \sum_{j=0}^{k} \gamma^j \sum_{i=1}^{n} \nabla J(\theta^{(k-j)};\, x_i)
# $$
#
# Le direzioni di discesa persistenti (come quella lungo il fondo di una valle)
# si accumulano e amplificano, mentre le oscillazioni trasversali si cancellano
# grazie alla media. Il risultato è una traiettoria molto più diretta verso
# il minimo, con oscillazioni laterali ridotte.
#
# Per la logistic regression gli aggiornamenti diventano:
#
# $$
# v_j^{(k+1)} = \gamma\, v_j^{(k)}
#   + \frac{\eta}{n} \sum_{i=1}^{n} \bigl(t_i - \sigma(\theta^{(k)\top} x_i)\bigr) x_{ij}
# $$
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)} + v_j^{(k+1)}
# $$

# %%
def momentum_gd(X, t, eta=0.1, gamma=0.97, epochs=1000):
    """
    Gradient Descent con momento per la logistic regression.

    Mantiene un vettore velocità v che accumula i gradienti passati
    con decadimento esponenziale controllato da gamma.

    Parametri
    ----------
    X      : array (n, d), design matrix
    t      : array (n, 1), target
    eta    : float, learning rate
    gamma  : float, coefficiente di attrito (0 < gamma < 1)
    epochs : int, numero di iterazioni

    Ritorna
    -------
    cost_history : array, storia dei valori di costo
    m, q         : array, storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    v             = np.zeros((nfeatures + 1, 1))   # vettore velocità, inizializzato a zero
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        # Aggiornamento velocità: inerzia + contributo gradiente
        v     = gamma * v - eta * gradient(theta, X, t)
        # Aggiornamento posizione: spostamento nella direzione della velocità
        theta = theta + v
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, m, q


# %%
# Esecuzione di GD con momento
cost_history_mom, m_mom, q_mom = momentum_gd(
    X, t, eta=0.1, gamma=0.97, epochs=10000
)

# %%
# Confronto con BGD: l'inerzia riduce le oscillazioni e accelera la convergenza
plot_all(cost_history_mom, m_mom, q_mom, low=0, high=5000, step=10)

# %%
print(f"Momentum GD: iterazioni alla convergenza = {convergence_iterations(m_mom, q_mom)}")

# %%
plot_ds(data, m_mom[-1], q_mom[-1])


# %% [markdown]
# ---
# ## 11. Accelerazione di Nesterov (NAG)
#
# ### Idea: look-ahead del gradiente
#
# Nel metodo del momento, al passo $k$ si conosce sia la posizione attuale
# $\theta^{(k)}$ sia il vettore velocità $v^{(k)}$. Senza calcolare il
# gradiente, si può già stimare la posizione *approssimata* al passo
# successivo:
#
# $$
# \tilde{\theta}^{(k+1)} = \theta^{(k)} + \gamma\, v^{(k)}
# $$
#
# Il **metodo di Nesterov** sfrutta questa previsione calcolando il
# gradiente non nella posizione attuale $\theta^{(k)}$, ma nella posizione
# anticipata $\tilde{\theta}^{(k+1)}$:
#
# $$
# v^{(k+1)} = \gamma\, v^{(k)}
#   - \eta \sum_{i=1}^{n} \nabla J\!\left(\tilde{\theta}^{(k+1)};\, x_i\right)
# $$
# $$
# \theta^{(k+1)} = \theta^{(k)} + v^{(k+1)}
# $$
#
# ### Perché funziona meglio del momento classico?
#
# Nel momento classico, il gradiente "corregge" la traiettoria *dopo* aver
# già fatto un passo di inerzia. In Nesterov, il gradiente viene valutato
# nel punto dove si "andrà a finire" con l'inerzia, ottenendo una correzione
# anticipata. Questo riduce l'**overshooting** (il tendere ad andare oltre
# il minimo) che si osserva nel momento classico, e permette di usare
# valori di $\gamma$ più aggressivi.
#
# Per la logistic regression la differenza è minima nell'implementazione:
# si sostituisce `gradient(theta, ...)` con `gradient(theta + gamma*v, ...)`.
#
# ```python
# v = 0
# for epoch in range(n_epochs):
#     theta_lookahead = theta + gamma * v
#     g = gradient(theta_lookahead, X, t)
#     v = gamma * v - eta * g
#     theta = theta + v
# ```

# %%
def nesterov_gd(X, t, eta=0.1, gamma=0.97, epochs=1000):
    """
    Nesterov Accelerated Gradient (NAG) per la logistic regression.

    Calcola il gradiente nella posizione anticipata theta + gamma*v
    anziché nella posizione corrente theta.

    Parametri
    ----------
    X, t, eta, gamma, epochs : identici a momentum_gd

    Ritorna
    -------
    cost_history, m, q : identici a momentum_gd
    """
    theta         = np.zeros((nfeatures + 1, 1))
    v             = np.zeros((nfeatures + 1, 1))
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        # Gradiente calcolato nella posizione anticipata (look-ahead)
        v     = gamma * v - eta * gradient(theta + gamma * v, X, t)
        theta = theta + v
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, m, q


# %%
cost_history_nes, m_nes, q_nes = nesterov_gd(
    X, t, eta=0.1, gamma=0.97, epochs=10000
)

# %%
plot_all(cost_history_nes, m_nes, q_nes, low=0, high=5000, step=10)

# %%
print(f"Nesterov GD: iterazioni alla convergenza = {convergence_iterations(m_nes, q_nes)}")

# %%
plot_ds(data, m_nes[-1], q_nes[-1])



# %% [markdown]
# ---
# ## 12. Adagrad: learning rate adattativi per parametro
#
# I metodi visti finora usano lo stesso learning rate $\eta$ per tutti i
# parametri a ogni iterazione. Questo è subottimale quando le feature hanno
# scale diverse o frequenze di aggiornamento molto diverse: i parametri
# relativi a feature *rare* (es. parole infrequenti in NLP) ricevono pochi
# segnali di aggiornamento, ma quando li ricevono dovrebbero essere aggiornati
# in modo più marcato rispetto ai parametri di feature comuni.
#
# **Adagrad** (Duchi et al., 2011) introduce learning rate *adattativi
# per parametro*: la scala dell'aggiornamento di $\theta_j$ al passo $k$
# è modulata dalla storia dei gradienti passati di $\theta_j$.
#
# ### Regola di aggiornamento
#
# Sia $g_{j,k} = \nabla_{\theta_j} J(\theta^{(k)})$ il gradiente di
# $\theta_j$ al passo $k$, e sia:
#
# $$
# G_{j,k} = \sum_{i=0}^{k} g_{j,i}^2
# $$
#
# la somma cumulativa dei quadrati dei gradienti passati per $\theta_j$.
# In forma vettoriale, con $\mathbf{G}_k = \sum_{i=0}^k \mathbf{g}_i \odot
# \mathbf{g}_i$ (con $\odot$ prodotto elemento per elemento):
#
# $$
# \boldsymbol{\theta}^{(k+1)}
#   = \boldsymbol{\theta}^{(k)}
#     - \frac{\eta}{\sqrt{\mathbf{G}_k + \varepsilon}} \odot \mathbf{g}_k
# $$
#
# dove $\varepsilon \approx 10^{-8}$ è un termine di smoothing che evita
# la divisione per zero.
#
# ### Interpretazione
#
# - Parametri con **gradienti storicamente grandi** ricevono un learning
#   rate effettivo più piccolo: $G_{j,k}$ è grande, quindi
#   $\eta/\sqrt{G_{j,k}}$ è piccolo. Il parametro rallenta.
# - Parametri con **gradienti storicamente piccoli** mantengono un
#   learning rate effettivo più alto e accelerano quando ricevono un segnale.
#
# Adagrad è particolarmente efficace con **dati sparsi**, dove feature
# rare vengono aggiornate aggressivamente quando compaiono, e feature
# frequenti vengono frenate per evitare oscillazioni.
#
# ### Limitazione strutturale
#
# Poiché $G_{j,k}$ è **monotonicamente crescente** (si somma il quadrato
# del gradiente a ogni passo, senza mai sottrarre), il learning rate
# effettivo $\eta/\sqrt{G_{j,k}}$ decade verso zero per *tutti* i parametri.
# In addestramenti lunghi, Adagrad può cessare di apprendere molto prima
# di aver raggiunto un ottimo soddisfacente.

# %%
def adagrad(X, t, eta=0.5, eps=1e-8, epochs=10000):
    """
    Adagrad per la logistic regression.

    Accumula i quadrati di tutti i gradienti passati in G e usa sqrt(G)
    per ridimensionare il learning rate separatamente per ogni parametro.

    Regola di aggiornamento:
        G     <- G + g * g                     (somma cumulativa, element-wise)
        theta <- theta - eta / sqrt(G + eps) * g

    Parametri
    ----------
    X      : array (n, d), design matrix con bias
    t      : array (n, 1), target binari
    eta    : float, learning rate globale (default 0.5)
    eps    : float, termine di smoothing per stabilità numerica (default 1e-8)
    epochs : int, numero di iterazioni

    Ritorna
    -------
    cost_history  : array (epochs, 1), storia dei valori di costo
    theta_history : array (epochs, d), storia dei parametri
    m, q          : array (epochs,), storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    G             = np.zeros_like(theta)   # accumulatore: cresce monotonicamente
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        g     = gradient(theta, X, t)
        G     = G + g ** 2                              # accumulo quadrati
        theta = theta - (eta / np.sqrt(G + eps)) * g   # passo adattativo
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m, q


# %%
# Esecuzione di Adagrad.
# Si usa un learning rate iniziale più alto: il ridimensionamento adattativo
# lo ridurrà automaticamente al crescere delle iterazioni.
cost_history_ag, theta_history_ag, m_ag, q_ag = adagrad(
    X, t, eta=0.5, epochs=10000
)

# %%
# Il costo decade rapidamente all'inizio (passi grandi), poi rallenta
# man mano che l'accumulatore G cresce e il learning rate effettivo si riduce.
plot_all(cost_history_ag, m_ag, q_ag, low=0, high=5000, step=10)

# %%
print(f"Adagrad: iterazioni alla convergenza = {convergence_iterations(m_ag, q_ag)}")

# %%
plot_ds(data, m_ag[-1], q_ag[-1])

# %%
# Visualizzazione del decadimento del learning rate effettivo per ogni parametro.
# Tutti e tre i learning rate tendono monotonicamente a zero, confermando
# la limitazione strutturale di Adagrad in addestramenti lunghi.
eta_adagrad = 0.5
G_cumul = np.cumsum(np.diff(theta_history_ag, axis=0, prepend=0) ** 2 * 0 +
                    (theta_history_ag * 0), axis=0)  # placeholder per forma
# Ricalcolo corretto dell'accumulatore G dalla storia dei gradienti
grads_ag = np.array([
    -gradient(theta_history_ag[k].reshape(-1,1), X, t).ravel()
    for k in range(len(theta_history_ag))
])
G_cumul_ag = np.cumsum(grads_ag ** 2, axis=0)
eta_eff_ag = eta_adagrad / np.sqrt(G_cumul_ag + 1e-8)

plt.figure(figsize=(12, 5))
for j, label in enumerate([r"$\theta_0$ (bias)", r"$\theta_1$", r"$\theta_2$"]):
    plt.plot(eta_eff_ag[:3000, j], lw=1.5, color=COLORS[j], label=label)
plt.xlabel("Iterazione")
plt.ylabel(r"Learning rate effettivo $\eta\,/\,\sqrt{G_j + \varepsilon}$")
plt.title("Adagrad: decadimento monotono del learning rate per parametro")
plt.legend()
plt.show()


# %% [markdown]
# ---
# ## 13. Adadelta: learning rate adattativi senza decadimento globale
#
# **Adadelta** (Zeiler, 2012) nasce esplicitamente per risolvere il
# problema di Adagrad: il decadimento monotono e illimitato del learning
# rate. L'idea chiave è sostituire la somma cumulativa dei gradienti al
# quadrato con una **media mobile esponenzialmente pesata** (EMA).
#
# ### Media mobile esponenziale
#
# L'accumulatore $G_{j,k}$ di Adagrad viene sostituito da:
#
# $$
# E[g^2]_{j,k} = \rho\, E[g^2]_{j,k-1} + (1-\rho)\, g_{j,k}^2
# $$
#
# con $\rho \in (0,1)$ tipicamente pari a $0.9$ o $0.95$.
# Il contributo del gradiente al passo $i < k$ decade come $\rho^{k-i}$:
# solo i **gradienti recenti** influenzano il learning rate attuale.
#
# ### Versione completa con doppio accumulatore
#
# Zeiler propone inoltre di applicare la stessa EMA agli **aggiornamenti**
# $\Delta\theta$ passati, con un secondo accumulatore:
#
# $$
# E[\Delta\theta^2]_{j,k}
#   = \rho\, E[\Delta\theta^2]_{j,k-1} + (1-\rho)\, \Delta\theta_{j,k}^2
# $$
#
# L'aggiornamento finale diventa:
#
# $$
# \Delta\theta_{j,k}
#   = -\frac{\sqrt{E[\Delta\theta^2]_{j,k-1} + \varepsilon}}
#           {\sqrt{E[g^2]_{j,k} + \varepsilon}}\, g_{j,k}
# $$
#
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)} + \Delta\theta_{j,k}
# $$
#
# In questa formulazione completa **$\eta$ non è più necessario**: il
# learning rate è determinato automaticamente dal rapporto tra la scala
# degli aggiornamenti passati e la scala dei gradienti passati, rendendo
# Adadelta praticamente privo di iperparametri critici da tarare.
#
# > **Nota**: la versione semplificata di Adadelta (con solo il primo
# > accumulatore e $\eta$ esplicito) coincide essenzialmente con
# > **RMSProp**, presentato nella prossima sezione.

# %%
def adadelta(X, t, rho=0.95, eps=1e-6, epochs=10000):
    """
    Adadelta (versione completa, senza learning rate esplicito) per la
    logistic regression.

    Mantiene due accumulatori EMA:
      - E_g2    : media mobile dei quadrati dei gradienti
      - E_dth2  : media mobile dei quadrati degli aggiornamenti passati

    Il learning rate è determinato automaticamente dal loro rapporto,
    senza bisogno di specificare eta.

    Regola di aggiornamento:
        E_g2    <- rho * E_g2   + (1-rho) * g^2
        dtheta   = -sqrt(E_dth2 + eps) / sqrt(E_g2 + eps) * g
        E_dth2  <- rho * E_dth2 + (1-rho) * dtheta^2
        theta   <- theta + dtheta

    Parametri
    ----------
    X      : array (n, d), design matrix con bias
    t      : array (n, 1), target binari
    rho    : float, coefficiente di decadimento EMA (default 0.95)
    eps    : float, termine di smoothing (default 1e-6)
    epochs : int, numero di iterazioni

    Ritorna
    -------
    cost_history  : array (epochs, 1), storia dei valori di costo
    theta_history : array (epochs, d), storia dei parametri
    m, q          : array (epochs,), storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    E_g2          = np.zeros_like(theta)   # EMA dei quadrati del gradiente
    E_dtheta2     = np.zeros_like(theta)   # EMA dei quadrati degli aggiornamenti
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        g = gradient(theta, X, t)

        # 1. Aggiornamento EMA dei gradienti al quadrato
        E_g2 = rho * E_g2 + (1 - rho) * g ** 2

        # 2. Calcolo dell'aggiornamento: numeratore = scala aggiornamenti passati,
        #    denominatore = scala gradienti correnti
        dtheta = -(np.sqrt(E_dtheta2 + eps) / np.sqrt(E_g2 + eps)) * g

        # 3. Aggiornamento EMA degli aggiornamenti (DOPO aver calcolato dtheta)
        E_dtheta2 = rho * E_dtheta2 + (1 - rho) * dtheta ** 2

        theta = theta + dtheta
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m, q


# %%
# Nessun learning rate da specificare: Adadelta si auto-calibra.
cost_history_ad, theta_history_ad, m_ad, q_ad = adadelta(
    X, t, rho=0.95, epochs=10000
)

# %%
# Il costo decade più regolarmente rispetto ad Adagrad, senza il "blocco"
# nelle iterazioni avanzate dovuto al decadimento illimitato del LR.
plot_all(cost_history_ad, m_ad, q_ad, low=0, high=5000, step=10)

# %%
print(f"Adadelta: iterazioni alla convergenza = {convergence_iterations(m_ad, q_ad)}")

# %%
plot_ds(data, m_ad[-1], q_ad[-1])


# %% [markdown]
# ---
# ## 14. RMSProp: Adagrad con memoria finita e learning rate esplicito
#
# **RMSProp** (*Root Mean Square Propagation*, Hinton, 2012) è stato
# proposto indipendentemente da Adadelta e condivide la stessa intuizione
# di fondo: sostituire la somma cumulativa di Adagrad con una EMA dei
# gradienti al quadrato per evitare il decadimento monotono del learning rate.
#
# ### Regola di aggiornamento
#
# $$
# E[g^2]_{j,k} = \rho\, E[g^2]_{j,k-1} + (1-\rho)\, g_{j,k}^2
# $$
#
# $$
# \theta_j^{(k+1)} = \theta_j^{(k)}
#   - \frac{\eta}{\sqrt{E[g^2]_{j,k} + \varepsilon}}\, g_{j,k}
# $$
#
# ### Differenza rispetto ad Adadelta
#
# La differenza rispetto ad Adadelta è che RMSProp **mantiene $\eta$ come
# iperparametro esplicito**, anziché sostituirlo con la scala adattativa
# degli aggiornamenti passati. Questo offre più controllo diretto, al costo
# di dover scegliere $\eta$.
#
# ### Confronto tra i metodi con LR adattativo
#
# | Metodo   | Accumulatore $G$              | Learning rate effettivo               |
# |:---------|:------------------------------|:--------------------------------------|
# | Adagrad  | $\sum g^2$ (→ ∞)              | $\eta / \sqrt{G}$ → 0                 |
# | RMSProp  | EMA di $g^2$ (stabile)        | $\eta / \sqrt{E[g^2]}$ (stabile)      |
# | Adadelta | EMA di $g^2$ + EMA di $\Delta\theta^2$ | automatico (no $\eta$)   |
#
# RMSProp è particolarmente efficace per le **reti neurali ricorrenti**
# (RNN), dove i gradienti variano velocemente nel tempo e l'adattamento
# rapido alle condizioni correnti è cruciale. Hinton consiglia tipicamente
# $\rho = 0.9$ ed $\eta = 0.001$.

# %%
def rmsprop(X, t, eta=0.01, rho=0.9, eps=1e-8, epochs=10000):
    """
    RMSProp per la logistic regression.

    Mantiene una EMA dei quadrati dei gradienti e la usa per
    ridimensionare il learning rate, evitando il decadimento illimitato
    di Adagrad pur mantenendo eta come iperparametro esplicito.

    Regola di aggiornamento:
        E_g2  <- rho * E_g2 + (1 - rho) * g^2
        theta <- theta - eta / sqrt(E_g2 + eps) * g

    Parametri
    ----------
    X      : array (n, d), design matrix con bias
    t      : array (n, 1), target binari
    eta    : float, learning rate globale (default 0.01)
    rho    : float, coefficiente di decadimento EMA (default 0.9)
    eps    : float, termine di smoothing (default 1e-8)
    epochs : int, numero di iterazioni

    Ritorna
    -------
    cost_history  : array (epochs, 1), storia dei valori di costo
    theta_history : array (epochs, d), storia dei parametri
    m, q          : array (epochs,), storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    E_g2          = np.zeros_like(theta)   # EMA dei quadrati del gradiente
    theta_history = []
    cost_history  = []

    for _ in range(epochs):
        g     = gradient(theta, X, t)
        E_g2  = rho * E_g2 + (1 - rho) * g ** 2         # aggiornamento EMA
        theta = theta - (eta / np.sqrt(E_g2 + eps)) * g  # passo adattativo
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m, q


# %%
cost_history_rms, theta_history_rms, m_rms, q_rms = rmsprop(
    X, t, eta=0.01, rho=0.9, epochs=10000
)

# %%
# A differenza di Adagrad, il learning rate si stabilizza: la curva di costo
# continua a scendere anche nelle iterazioni avanzate.
plot_all(cost_history_rms, m_rms, q_rms, low=0, high=5000, step=10)

# %%
print(f"RMSProp: iterazioni alla convergenza = {convergence_iterations(m_rms, q_rms)}")

# %%
plot_ds(data, m_rms[-1], q_rms[-1])


# %% [markdown]
# ---
# ## 15. Adam: momento adattativo
#
# **Adam** (*Adaptive Moment Estimation*, Kingma & Ba, 2015) è l'ottimizzatore
# più utilizzato nell'addestramento di reti neurali moderne. Combina le idee
# del **metodo del momento** (traccia la direzione media del gradiente) con
# quelle di **RMSProp** (adatta la scala per ogni parametro), aggiungendo
# una **correzione del bias di inizializzazione** che migliora le prime
# iterazioni.
#
# ### Due stime di momento
#
# Adam mantiene due accumulatori EMA, interpretabili come stime del
# primo e del secondo momento della distribuzione del gradiente:
#
# $$
# m_k = \beta_1\, m_{k-1} + (1-\beta_1)\, g_k
#   \quad\leftarrow\text{primo momento: direzione media del gradiente}
# $$
#
# $$
# v_k = \beta_2\, v_{k-1} + (1-\beta_2)\, g_k^2
#   \quad\leftarrow\text{secondo momento: variabilità del gradiente}
# $$
#
# con valori predefiniti $\beta_1 = 0.9$, $\beta_2 = 0.999$.
# Il primo momento accumula la direzione persistente (come il momento
# classico); il secondo momento misura la scala locale (come RMSProp).
#
# ### Correzione del bias di inizializzazione
#
# Poiché $m_0 = v_0 = \mathbf{0}$, nelle prime iterazioni gli accumulatori
# sono fortemente distorti verso zero: per piccoli $k$, il prodotto
# $\beta_1^k$ è ancora vicino a 1, per cui $m_k \approx (1-\beta_1^k) \cdot
# \text{gradiente} \approx 0$.
# Adam corregge esplicitamente questo bias dividendo per i termini di
# normalizzazione:
#
# $$
# \hat{m}_k = \frac{m_k}{1 - \beta_1^k}, \qquad
# \hat{v}_k = \frac{v_k}{1 - \beta_2^k}
# $$
#
# Alle prime iterazioni la correzione è grande (riporta le stime ai valori
# corretti); dopo molti passi $\beta^k \to 0$ e la correzione diventa
# trascurabile.
#
# ### Regola di aggiornamento
#
# $$
# \theta^{(k+1)} = \theta^{(k)}
#   - \frac{\eta}{\sqrt{\hat{v}_k} + \varepsilon}\, \hat{m}_k
# $$
#
# I valori predefiniti degli autori — $\eta = 0.001$, $\beta_1 = 0.9$,
# $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$ — funzionano bene su una
# vasta gamma di problemi senza richiedere taratura estesa.
#
# ### Perché Adam funziona così bene in pratica?
#
# - Il **primo momento** fornisce inerzia direzionale: le direzioni di
#   discesa persistenti vengono amplificate, riducendo le oscillazioni.
# - Il **secondo momento** adatta la scala: i parametri con gradienti
#   variabili ricevono passi più piccoli, quelli stabili passi più grandi.
# - La **correzione del bias** garantisce aggiornamenti significativi
#   sin dalle primissime iterazioni, quando gli accumulatori non sono
#   ancora "riscaldati".
# - Il risultato è un metodo che converge rapidamente, è relativamente
#   insensibile alla scelta di $\eta$, e si comporta bene sia con
#   gradienti sparsi sia con gradienti densi.
#
# ### Principali varianti di Adam
#
# | Variante    | Modifica principale                                          |
# |:------------|:-------------------------------------------------------------|
# | **AdaMax**  | Sostituisce la norma $L^2$ di $v_k$ con la norma $L^\infty$ |
# | **Nadam**   | Incorpora il look-ahead di Nesterov nel primo momento        |
# | **AMSGrad** | Usa il massimo storico di $\hat{v}_k$ per garantire convergenza teorica |
# | **AdamW**   | Separa il weight decay (L2) dall'aggiornamento adattativo    |
#
# **AdamW** è attualmente lo standard per i modelli linguistici di grandi
# dimensioni (LLM): il weight decay corretto (applicato direttamente a
# $\theta$, non al gradiente) migliora significativamente la generalizzazione.

# %%
def adam(X, t, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8, epochs=10000):
    """
    Adam (Adaptive Moment Estimation) per la logistic regression.

    Combina momento del primo ordine (EMA del gradiente) e adattamento
    della scala (EMA del gradiente al quadrato), con correzione del
    bias di inizializzazione a ogni iterazione.

    Regola di aggiornamento (al passo k = 1, 2, ...):
        m      <- beta1 * m + (1 - beta1) * g          # EMA del gradiente
        v      <- beta2 * v + (1 - beta2) * g^2        # EMA di g^2
        m_hat   = m / (1 - beta1^k)                    # correzione bias
        v_hat   = v / (1 - beta2^k)                    # correzione bias
        theta  <- theta - eta / (sqrt(v_hat) + eps) * m_hat

    Parametri
    ----------
    X      : array (n, d), design matrix con bias
    t      : array (n, 1), target binari
    eta    : float, learning rate (default 0.001)
    beta1  : float, decadimento primo momento   (default 0.9)
    beta2  : float, decadimento secondo momento (default 0.999)
    eps    : float, termine di smoothing        (default 1e-8)
    epochs : int, numero di iterazioni

    Ritorna
    -------
    cost_history  : array (epochs, 1), storia dei valori di costo
    theta_history : array (epochs, d), storia dei parametri
    m_line, q_line : array (epochs,), storia di coeff. angolare e intercetta
    """
    theta         = np.zeros((nfeatures + 1, 1))
    m_acc         = np.zeros_like(theta)   # primo momento  (EMA del gradiente)
    v_acc         = np.zeros_like(theta)   # secondo momento (EMA di g^2)
    theta_history = []
    cost_history  = []

    for k in range(1, epochs + 1):
        g = gradient(theta, X, t)

        # Aggiornamento dei momenti (EMA)
        m_acc = beta1 * m_acc + (1 - beta1) * g
        v_acc = beta2 * v_acc + (1 - beta2) * g ** 2

        # Correzione del bias di inizializzazione
        m_hat = m_acc / (1 - beta1 ** k)
        v_hat = v_acc / (1 - beta2 ** k)

        # Aggiornamento dei parametri
        theta = theta - eta * m_hat / (np.sqrt(v_hat) + eps)

        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    m_line = -theta_history[:, 1] / theta_history[:, 2]
    q_line = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m_line, q_line


# %%
cost_history_adam, theta_history_adam, m_adam_r, q_adam_r = adam(
    X, t, eta=0.001, beta1=0.9, beta2=0.999, epochs=10000
)

# %%
# Adam converge in modo stabile e spesso in meno iterazioni rispetto agli
# altri metodi, grazie alla combinazione di inerzia direzionale e scala adattativa.
plot_all(cost_history_adam, m_adam_r, q_adam_r, low=0, high=5000, step=10)

# %%
print(f"Adam: iterazioni alla convergenza = {convergence_iterations(m_adam_r, q_adam_r)}")

# %%
plot_ds(data, m_adam_r[-1], q_adam_r[-1])


# %% [markdown]
# ---
# ### Confronto finale tra tutti gli ottimizzatori
#
# La tabella seguente riassume le caratteristiche dei metodi presentati,
# come riferimento pratico per la scelta dell'ottimizzatore.
#
# | Metodo       | LR adatt. | Memoria         | Momento       | Iperparametri principali          |
# |:-------------|:---------:|:----------------|:-------------:|:----------------------------------|
# | BGD          | ✗         | —               | ✗             | $\eta$                            |
# | SGD          | ✗         | —               | ✗             | $\eta$                            |
# | MBGD         | ✗         | —               | ✗             | $\eta$, $s$                       |
# | Momentum     | ✗         | EMA (1 acc.)    | ✓             | $\eta$, $\gamma$                  |
# | Nesterov     | ✗         | EMA (1 acc.)    | ✓ look-ahead  | $\eta$, $\gamma$                  |
# | Adagrad      | ✓         | Cumulativa (↗∞) | ✗             | $\eta$                            |
# | Adadelta     | ✓         | EMA (2 acc.)    | ✗             | $\rho$                            |
# | RMSProp      | ✓         | EMA (1 acc.)    | ✗             | $\eta$, $\rho$                    |
# | **Adam**     | ✓         | EMA (2 acc.)    | ✓             | $\eta$, $\beta_1$, $\beta_2$      |

# %%
# Confronto visivo delle curve di costo per i sei metodi principali.
# Tutti visualizzati sulla stessa finestra temporale per un confronto equo.
N_SHOW = 3000

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
configs = [
    (cost_history_bgd,  "BGD",      COLORS[0]),
    (cost_history_mom,  "Momentum", COLORS[1]),
    (cost_history_ag,   "Adagrad",  COLORS[2]),
    (cost_history_ad,   "Adadelta", COLORS[3]),
    (cost_history_rms,  "RMSProp",  COLORS[4]),
    (cost_history_adam, "Adam",     COLORS[5]),
]
for ax, (ch, label, color) in zip(axes.ravel(), configs):
    ax.plot(ch[:N_SHOW], lw=1.5, color=color)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Costo")
    ax.tick_params(labelsize=8)

plt.suptitle("Confronto curve di costo — primi 3000 passi", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Riepilogo numerico: iterazioni alla convergenza per ogni metodo
results = {
    "BGD":      convergence_iterations(m_bgd,    q_bgd),
    "SGD":      convergence_iterations(m_sgd,    q_sgd),
    "MBGD":     convergence_iterations(m_mb,     q_mb),
    "Momentum": convergence_iterations(m_mom,    q_mom),
    "Nesterov": convergence_iterations(m_nes,    q_nes),
    "Adagrad":  convergence_iterations(m_ag,     q_ag),
    "Adadelta": convergence_iterations(m_ad,     q_ad),
    "RMSProp":  convergence_iterations(m_rms,    q_rms),
    "Adam":     convergence_iterations(m_adam_r, q_adam_r),
}
print("Iterazioni alla convergenza (distanza < 1e-2 dalla soluzione ottima):")
for name, iters in results.items():
    print(f"  {name:<12}: {iters:>7}")


# %% [markdown]
# ---
# ## 16. Metodi del secondo ordine: Newton-Raphson
#
# ### Motivazione
#
# I metodi del primo ordine (gradient descent e varianti) usano solo
# informazioni sulla pendenza locale (gradiente). I **metodi del secondo
# ordine** sfruttano anche la curvatura locale (derivate seconde), ottenendo
# convergenze molto più rapide, al costo di un maggiore onere computazionale.
#
# ### Newton-Raphson per la ricerca di zeri
#
# Il metodo di Newton-Raphson cerca i **punti stazionari** di $J(\theta)$
# (dove il gradiente si annulla) iterando:
#
# $$
# x^{(k+1)} = x^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})}
#             \qquad \text{(caso univariato)}
# $$
#
# L'idea geometrica è: a ogni passo si approssima $f'$ con la retta
# tangente, e si salta al suo zero. Applicato alla minimizzazione di $J$
# (dove si cerca lo zero della derivata prima $J'$):
#
# $$
# x^{(k+1)} = x^{(k)} - \frac{J'(x^{(k)})}{J''(x^{(k)})}
# $$
#
# ### Caso multivariato: la matrice Hessiana
#
# Per funzioni $J: \mathbb{R}^d \to \mathbb{R}$, la derivata seconda scalare
# è sostituita dalla **matrice Hessiana** $H \in \mathbb{R}^{d \times d}$:
#
# $$
# H_{ij}(J) = \frac{\partial^2 J}{\partial \theta_i\, \partial \theta_j}
# $$
#
# L'aggiornamento di Newton diventa:
#
# $$
# \theta^{(k+1)} = \theta^{(k)}
#   - \left[H(J)^{-1} \nabla J\right]_{\theta = \theta^{(k)}}
# $$
#
# L'Hessiana pre-condiziona il gradiente: invece di scendere di una quantità
# proporzionale al gradiente, lo spostamento è proporzionale a
# $H^{-1} \nabla J$, che tiene conto della curvatura in ogni direzione.
# In zone a curvatura elevata (dove la superficie è "ripida") il passo
# viene ridotto; in zone piatte viene ampliato.
#
# ### Limitazioni dei metodi del secondo ordine
#
# - **Costo computazionale**: calcolare e invertire $H \in \mathbb{R}^{d \times d}$
#   richiede $O(d^2)$ memoria e $O(d^3)$ operazioni. Per reti neurali con
#   milioni di parametri, questo è proibitivo.
# - **Metodi quasi-Newton**: L-BFGS e BFGS approssimano $H^{-1}$ in modo
#   efficiente usando solo la storia dei gradienti, riducendo il costo a
#   $O(d)$ o $O(kd)$ per $k$ iterazioni memorizzate.
# - **Applicabilità in ML**: i metodi del secondo ordine trovano applicazione
#   in problemi con numero moderato di parametri (regressione logistica,
#   SVM, piccole reti neurali) o come step di fine-tuning dopo una fase
#   iniziale con gradient descent stocastico.

# %% [markdown]
# ---
# ### Hessiana della cross-entropy per la logistic regression
#
# Per applicare il metodo di Newton occorre calcolare analiticamente la
# matrice Hessiana della funzione di costo.
#
# Ricordiamo che il gradiente del rischio empirico è:
#
# $$
# \nabla J(\theta) = -\frac{1}{n}\, \mathbf{X}^\top (\mathbf{t} - \boldsymbol{\sigma})
# $$
#
# dove $\boldsymbol{\sigma} = \sigma(\mathbf{X}\theta) \in \mathbb{R}^n$.
# Derivando nuovamente rispetto a $\theta_j$, si ottiene la derivata seconda:
#
# $$
# \frac{\partial^2 J}{\partial \theta_i \, \partial \theta_j}
#   = \frac{1}{n} \sum_{k=1}^{n} \sigma_k (1 - \sigma_k)\, x_{ki}\, x_{kj}
# $$
#
# Il termine $\sigma_k(1-\sigma_k)$ è la varianza di una variabile di
# Bernoulli con parametro $\sigma_k$: vale al massimo $\frac{1}{4}$,
# raggiunto quando il modello è massimamente incerto ($\sigma_k = \frac{1}{2}$).
#
# Definendo la matrice diagonale
# $\mathbf{S} = \mathrm{diag}\!\left(\sigma_k(1-\sigma_k)\right)$,
# l'intera Hessiana si scrive in forma compatta:
#
# $$
# \mathbf{H}(\theta) = \frac{1}{n}\, \mathbf{X}^\top \mathbf{S}\, \mathbf{X}
# $$
#
# Questa matrice è:
#
# - **Simmetrica**, per costruzione ($\mathbf{X}^\top \mathbf{S} \mathbf{X}$
#   è sempre simmetrica).
# - **Semidefinita positiva**, poiché tutti gli elementi di $\mathbf{S}$
#   sono non negativi. Con dati non degeneri risulta **definita positiva**,
#   garantendo che la cross-entropy sia strettamente convessa e ammetta
#   un **unico minimo globale**.
#
# La convessità garantisce la convergenza del metodo di Newton al minimo
# globale da qualsiasi punto di partenza (purché $\mathbf{H}$ rimanga
# invertibile, condizione che nel caso della logistic regression è quasi
# sempre soddisfatta).

# %%
def hessian(theta, X):
    """
    Calcola la matrice Hessiana della cross-entropy per la logistic regression.

    Formula analitica:
        H = (1/n) * X^T diag(sigma * (1 - sigma)) X

    dove sigma * (1 - sigma) sono le varianze delle predizioni bernoulliane,
    con valori in (0, 1/4]. La matrice e' simmetrica e semidefinita positiva
    (definita positiva con dati non degeneri).

    Parametri
    ----------
    theta : array (d, 1), parametri correnti
    X     : array (n, d), design matrix con bias

    Ritorna
    -------
    H : array (d, d), matrice Hessiana
    """
    s     = sigma(theta, X).ravel()    # predizioni in (0, 1)
    s_var = s * (1.0 - s)             # varianze bernoulliane, in (0, 1/4]
    S     = np.diag(s_var)            # matrice diagonale S = diag(s_var)
    return (X.T @ S @ X) / len(X)    # H = (1/n) X^T S X


# %% [markdown]
# ### Il passo di Newton e la convergenza quadratica
#
# L'aggiornamento di Newton al passo $k$ risolve il sistema lineare:
#
# $$
# \mathbf{H}(\theta^{(k)})\, \Delta\theta = \nabla J(\theta^{(k)})
# $$
#
# e aggiorna i parametri con un **damping factor** $\eta \in (0, 1]$:
#
# $$
# \theta^{(k+1)} = \theta^{(k)} - \eta\, \Delta\theta
#   = \theta^{(k)} - \eta\, \mathbf{H}(\theta^{(k)})^{-1} \nabla J(\theta^{(k)})
# $$
#
# Risolvere il sistema lineare tramite fattorizzazione LU (con
# `numpy.linalg.solve`) è preferibile all'inversione esplicita di $\mathbf{H}$:
# stesso costo asintotico $O(d^3)$, ma maggiore stabilità numerica in
# presenza di quasi-singolarità.
#
# Il ruolo del damping factor:
#
# - $\eta = 1$: **Newton puro** — passo completo, convergenza quadratica,
#   ma può essere instabile lontano dall'ottimo.
# - $\eta < 1$: **Newton smorzato** — passo ridotto, più robusto nelle
#   fasi iniziali, a scapito di una convergenza leggermente più lenta.
#
# ### Convergenza quadratica
#
# La caratteristica distintiva di Newton è la **convergenza quadratica**
# in un intorno del minimo. Detto $e_k = \|\theta^{(k)} - \theta^*\|$
# l'errore al passo $k$, vale asintoticamente:
#
# $$
# e_{k+1} \leq C\, e_k^2
# $$
#
# per una costante $C > 0$. Questo significa che il numero di **cifre
# decimali corrette raddoppia a ogni passo**:
#
# $$
# e_0 \sim 10^{-1}
# \;\longrightarrow\;
# e_1 \sim 10^{-2}
# \;\longrightarrow\;
# e_2 \sim 10^{-4}
# \;\longrightarrow\;
# e_3 \sim 10^{-8}
# \;\longrightarrow\;
# e_4 \sim 10^{-16}
# $$
#
# In pratica, Newton converge per la logistic regression in **meno di
# 15--20 passi**, indipendentemente dalla dimensione del dataset, contro
# le decine di migliaia di passi necessari ai metodi del primo ordine.
#
# Su scala logaritmica, la norma del gradiente produce una curva con
# **pendenza crescente** (la firma della convergenza quadratica), a
# differenza della retta a pendenza costante della convergenza lineare
# dei metodi del primo ordine.

# %%
def newton_method(X, t, eta=1.0, epochs=50, tol=1e-10):
    """
    Metodo di Newton smorzato per la logistic regression.

    A ogni iterazione calcola il gradiente e la Hessiana della cross-entropy,
    risolve il sistema lineare H * delta = grad, e aggiorna theta con
    step size eta. Converge quadraticamente al minimo globale (unico,
    per la convessita' della cross-entropy).

    Parametri
    ----------
    X      : array (n, d), design matrix con bias
    t      : array (n, 1), target binari
    eta    : float, damping factor in (0, 1] (default 1.0 = Newton puro)
    epochs : int, numero massimo di iterazioni (default 50)
    tol    : float, soglia ||grad|| per il criterio di convergenza (default 1e-10)

    Ritorna
    -------
    cost_history  : array (iters, 1), storia dei valori di costo
    theta_history : array (iters, d), storia dei parametri
    m, q          : array (iters,), storia di coeff. angolare e intercetta
    grad_norms    : array (iters,), norma del gradiente a ogni passo
    """
    theta         = np.zeros((nfeatures + 1, 1))
    theta_history = []
    cost_history  = []
    grad_norms    = []

    for k in range(epochs):
        g         = gradient(theta, X, t)    # gradiente corrente: (d, 1)
        H         = hessian(theta, X)        # Hessiana corrente:  (d, d)
        grad_norm = np.linalg.norm(g)

        # Salva stato prima dell'aggiornamento
        theta_history.append(theta.copy())
        cost_history.append(cost(theta, X, t))
        grad_norms.append(grad_norm)

        # Criterio di arresto: gradiente numericamente trascurabile
        if grad_norm < tol:
            print(f"  Convergenza all'iterazione {k}  "
                  f"(||grad|| = {grad_norm:.2e} < tol = {tol:.0e})")
            break

        # Passo di Newton: risolve H * delta = g (piu' stabile di inv(H) @ g)
        delta = np.linalg.solve(H, g)
        theta = theta - eta * delta          # aggiornamento smorzato

    theta_history = np.array(theta_history).reshape(-1, 3)
    cost_history  = np.array(cost_history).reshape(-1, 1)
    grad_norms    = np.array(grad_norms)
    m = -theta_history[:, 1] / theta_history[:, 2]
    q = -theta_history[:, 0] / theta_history[:, 2]
    return cost_history, theta_history, m, q, grad_norms


# %%
# Newton puro: attesi meno di 20 passi contro le 10^4-10^5 dei metodi GD.
print("Metodo di Newton (eta=1.0):")
cost_history_nw, theta_history_nw, m_nw, q_nw, grad_norms_nw = newton_method(
    X, t, eta=1.0, epochs=50, tol=1e-10
)
print(f"  Iterazioni eseguite : {len(cost_history_nw)}")
print(f"  Costo finale        : {cost_history_nw[-1, 0]:.10f}")
print(f"  ||grad|| finale     : {grad_norms_nw[-1]:.2e}")

# %%
# Traiettoria: pochi punti discreti, ciascuno corrispondente a un passo Newton.
# La rapida convergenza rende il grafico molto "scarno" rispetto ai metodi GD.
plot_all(cost_history_nw, m_nw, q_nw,
         low=0, high=len(cost_history_nw), step=1)

# %%
# Retta di separazione ottenuta con il metodo di Newton
plot_ds(data, m_nw[-1], q_nw[-1])

# %%
# Visualizzazione della convergenza quadratica.
#
# Pannello sinistro (scala lineare): il crollo del costo e' brusco e in
# pochi passi; i valori di costo sono annotati per evidenziare la rapidita'.
#
# Pannello destro (scala logaritmica): la norma del gradiente mostra una
# curva con pendenza CRESCENTE — firma distintiva della convergenza
# quadratica. Confrontare con la retta a pendenza costante tipica dei
# metodi del primo ordine.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(cost_history_nw, marker="o", lw=2, color=COLORS[0], markersize=7)
for i, c in enumerate(cost_history_nw.ravel()):
    ax.annotate(f"{c:.4f}", (i, c), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7)
ax.set_xlabel("Iterazione Newton")
ax.set_ylabel("Costo (cross-entropy)")
ax.set_title("Discesa del costo — Newton puro")
ax.tick_params(labelsize=8)

ax = axes[1]
ax.semilogy(grad_norms_nw, marker="s", lw=2, color=COLORS[1], markersize=7)
for i, gn in enumerate(grad_norms_nw):
    ax.annotate(f"{gn:.1e}", (i, gn), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7)
ax.set_xlabel("Iterazione Newton")
ax.set_ylabel(r"$\|\nabla J(\theta)\|$ (scala logaritmica)")
ax.set_title("Norma del gradiente — convergenza quadratica")
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()

# %%
# Newton puro vs. Newton smorzato (eta=0.5).
# Il metodo smorzato richiede piu' passi nella fase iniziale, ma converge
# comunque rapidamente grazie alla struttura quadratica del problema.
print("Newton smorzato (eta=0.5):")
cost_nw_05, _, m_nw_05, q_nw_05, gn_nw_05 = newton_method(
    X, t, eta=0.5, epochs=100, tol=1e-10
)
print(f"  Iterazioni eseguite : {len(cost_nw_05)}")
print(f"  Costo finale        : {cost_nw_05[-1, 0]:.10f}")
print(f"  ||grad|| finale     : {gn_nw_05[-1]:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.semilogy(grad_norms_nw, marker="o", lw=2, color=COLORS[0], markersize=6,
            label=r"Newton puro ($\eta=1$)")
ax.semilogy(gn_nw_05, marker="s", lw=2, color=COLORS[1], markersize=6,
            label=r"Newton smorzato ($\eta=0.5$)")
ax.set_xlabel("Iterazione")
ax.set_ylabel(r"$\|\nabla J(\theta)\|$ (scala log)")
ax.set_title("Norma del gradiente: puro vs. smorzato")
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)

ax = axes[1]
ax.plot(cost_history_nw.ravel(), marker="o", lw=2, color=COLORS[0], markersize=6,
        label=r"Newton puro ($\eta=1$)")
ax.plot(cost_nw_05.ravel(), marker="s", lw=2, color=COLORS[1], markersize=6,
        label=r"Newton smorzato ($\eta=0.5$)")
ax.set_xlabel("Iterazione")
ax.set_ylabel("Costo (cross-entropy)")
ax.set_title("Costo: puro vs. smorzato")
ax.legend(fontsize=9)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()

# %%
# Riepilogo numerico: Newton vs. tutti i metodi del primo ordine.
# Si noti la differenza di ordine di grandezza nel numero di iterazioni.
print("=" * 64)
print(f"{'Metodo':<16} {'Passi':>8} {'Costo finale':>16}")
print("-" * 64)
fo_results = [
    ("BGD",       cost_history_bgd,  m_bgd,    q_bgd),
    ("Momentum",  cost_history_mom,  m_mom,    q_mom),
    ("Adagrad",   cost_history_ag,   m_ag,     q_ag),
    ("RMSProp",   cost_history_rms,  m_rms,    q_rms),
    ("Adam",      cost_history_adam, m_adam_r, q_adam_r),
]
for name, ch, m_h, q_h in fo_results:
    iters = convergence_iterations(m_h, q_h)
    print(f"{name:<16} {iters:>8} {ch[-1, 0]:>16.8f}   (r<1e-2 dalla soluzione ottima)")
print(f"{'Newton (eta=1)':<16} {len(cost_history_nw):>8} "
      f"{cost_history_nw[-1, 0]:>16.8f}   (||grad|| < 1e-10)")
print("=" * 64)

# %%
# Confronto visivo delle traiettorie nello spazio (q, m).
#
# - BGD percorre un cammino lungo e serpeggiante (100k passi).
# - Adam e' piu' diretto grazie al momento e all'adattamento (10k passi).
# - Newton raggiunge la soluzione quasi in linea retta (< 20 passi):
#   l'Hessiana pre-condiziona lo spostamento correggendo la curvatura
#   locale, rendendo ogni passo quasi ottimale.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
traj_configs = [
    ("BGD (100k passi)",    m_bgd,    q_bgd,    COLORS[0]),
    ("Adam (10k passi)",    m_adam_r, q_adam_r, COLORS[5]),
    ("Newton (< 20 passi)", m_nw,     q_nw,     COLORS[2]),
]
for ax, (label, m_h, q_h, color) in zip(axes, traj_configs):
    ax.plot(q_h, m_h, lw=1.5, color=color, alpha=0.8)
    ax.scatter(q_h[0],  m_h[0],  color=color, marker="x",
               s=120, zorder=5, linewidths=2.5, label="Inizio")
    ax.scatter(q_h[-1], m_h[-1], color="k", marker="o",
               s=80, zorder=5, label="Fine")
    ax.set_xlabel(r"$q$ (intercetta)", fontsize=10)
    ax.set_ylabel(r"$m$ (coeff. angolare)", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.suptitle("Traiettoria nello spazio dei parametri $(q, m)$", fontsize=12)
plt.tight_layout()
plt.show()
