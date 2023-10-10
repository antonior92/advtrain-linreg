from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn import linear_model
import tqdm
from advtrain import AdversarialTraining


def get_lasso_path(X, y, eps_lasso=1e-5):
    alphas, coefs, _ = lasso_path(X, y, eps=eps_lasso)
    coefs= np.concatenate([np.zeros([X.shape[1], 1]), coefs], axis=1)
    alphas = np.concatenate([1e2 * np.ones([1]), alphas], axis=0)
    return alphas, coefs, []


def get_path(X, y, estimator, amax, eps=1e-5, n_alphas=200):
    amin = eps * amax
    alphas = np.logspace(np.log10(amin), np.log10(amax), n_alphas)
    coefs_ = []
    for a in tqdm.tqdm(alphas):
        coefs = estimator(X, y, a)
        coefs_.append(coefs if coefs is not None else np.zeros(m))
    return alphas, np.stack((coefs_)).T


def plot_coefs(alphas, coefs, ax):
    colors = cycle(["b", "r", "g", "c", "k"])
    for coef_l, c in zip(coefs, colors):
        ax.semilogx(1/alphas, coef_l, c=c)


def plot_coefs_l1norm(coefs, ax):
    colors = cycle(["b", "r", "g", "c", "k"])
    l1norm = np.abs(coefs).mean(axis=0)
    for coef_l, c in zip(coefs, colors):
        ax.plot(l1norm, coef_l, c=c)


def diabetes_path():
    X, y = datasets.load_diabetes(return_X_y=True)
    # Standardize data
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    fig, ax = plt.subplots(num='ridge')
    estimator = lambda X, y, a: linear_model.Ridge(alpha=a, fit_intercept=False).fit(X, y).coef_
    alphas_ridge, coefs_ridge = get_path(X, y, estimator, 1e4)
    plot_coefs(alphas_ridge, coefs_ridge, ax)

    fig, ax = plt.subplots(num='lasso')
    alphas_lasso, coefs_lasso, _ = get_lasso_path(X, y)
    plot_coefs_l1norm(coefs_lasso, ax)

    fig, ax = plt.subplots(num='advtrain_l2')
    l2advtrain = AdversarialTraining(X, y, p=2)
    estimator = lambda X, y, a:  l2advtrain(adv_radius=a)
    alphas_adv, coefs_advtrain_l2 = get_path(X, y, estimator, 1e1)
    plot_coefs(alphas_adv, coefs_advtrain_l2, ax)

    fig, ax = plt.subplots(num='advtrain_linf')
    linfadvtrain = AdversarialTraining(X, y, p=np.inf)
    estimator = lambda X, y, a:  linfadvtrain(adv_radius=a)
    alphas_adv, coefs_advtrain_linf  = get_path(X, y, estimator, 1e1)
    plot_coefs_l1norm(coefs_advtrain_linf, ax)


if __name__ == '__main__':
    diabetes_path()
    plt.show()