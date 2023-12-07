from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as plticker
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn import linear_model
import tqdm
import os
from training import adversarial_training


######  JUST FOR GETTING THE RIGHT STYLE ##########
# You can comment this out if you are not generating the figures for the paper
def initialize_plots():
    plt.style.use(['../mystyle.mplsty'])
    mpl.rcParams['figure.figsize'] = 7, 3
    mpl.rcParams['figure.subplot.left'] = 0.15
    mpl.rcParams['figure.subplot.bottom'] = 0.25
    mpl.rcParams['figure.subplot.right'] = 0.99
    mpl.rcParams['figure.subplot.top'] = 0.95
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['legend.handlelength'] = 1
    mpl.rcParams['legend.handletextpad'] = 0.01
    mpl.rcParams['xtick.major.pad'] = 10


def prepare_for_paper(ax, ylabel, xlabel, set_yticks):
    ax.set_xlabel(xlabel)
    loc = plticker.LogLocator(base=10, numticks=10)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(4))
    locmin = plticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    if set_yticks:
        ax.set_ylim(-40, 40)
        ax.set_yticks([-40, -20, 0, 20, 40])
    if ylabel:
        ax.set_ylabel("coefficients")
    plt.grid(ls=':')


def prepare_for_paper_l1(ax, ylabel, set_yticks):
    if ylabel:
        ax.set_ylabel('coefficients')
    ax.set_xlabel(r'$||\widehat \beta||_1$')
    if set_yticks:
        ax.set_ylim(-40, 40)
        ax.set_yticks([-40, -20, 0, 20, 40])
    plt.subplots_adjust(bottom=0.3)
######  END ##########


def plot_coefs(alphas, coefs, name, xlabel=r'$1/\delta$', ylabel=True):
    fig, ax = plt.subplots()

    colors = cycle(["b", "r", "g", "c", "k"])
    for coef_l, c in zip(coefs, colors):
        plt.semilogx(1/alphas, coef_l, c=c)

    if args.save:
        prepare_for_paper(ax, ylabel, xlabel, set_yticks=(args.dset == 'diabetes'))
        plt.savefig(os.path.join(args.save,'{}_{}.pdf'.format(args.dset, name)))
    else:
        plt.title(name)
        plt.show()


def plot_area(coefs, xaxis):
    """Plot the area where  y - X @ coefs > 0"""
    mask = (y[:, None] - X @ coefs > 0).all(axis=0)
    xaxism = xaxis[mask]
    plt.axvspan(xaxism[-1], xaxism[0], color='red', alpha=0.2)


def plot_coefs_l1(coefs, name, ylabel=True, add_area=False):
    fig, ax = plt.subplots()
    alphas = np.abs(coefs).mean(axis=0)
    colors = cycle(["b", "r", "g", "c", "k"])
    if add_area:
        plot_area(coefs, alphas)
    for coef_l, c in zip(coefs, colors):
        ax.plot(alphas, coef_l, c=c)
    if args.save:
        prepare_for_paper_l1(ax, ylabel, set_yticks=(args.dset == 'diabetes'))
        plt.savefig(os.path.join(args.save,'{}_{}_l1.pdf'.format(args.dset, name)))
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot parameters profile')
    parser.add_argument('--save', default='',
                        help='save plot in the given folder (do not write extension). By default just show it.')
    parser.add_argument('--dset', choices=['diabetes', 'gaussian'], default='diabetes')
    parser.add_argument('--l1_xaxis', action='store_true',
                        help='If true also plot x axis.')
    parser.add_argument('--area', action='store_true',
                        help='Plot area')
    args, unk = parser.parse_known_args()

    if args.save:
        initialize_plots()

    # Path lengh
    eps_lasso = 1e-5
    eps_ridge = 1e-6
    eps_adv = 1e-5
    # alpha_max
    amax_ridge = 1e4
    amax_adv = 1
    # number of points along the path
    n_alphas = 200

    if args.dset == 'diabetes':
        X, y = datasets.load_diabetes(return_X_y=True)
        n, m = X.shape
        # Standardize data (easier to set the l1_ratio parameter)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    elif args.dset == 'gaussian':
        amax_adv = 10
        n = 200
        m = 10
        rng = np.random.RandomState(1)
        X = rng.randn(n, m)
        X /= X.std(axis=0)
        beta = 2 * rng.rand(m) - 1
        eps = rng.randn(n)
        y = X @ beta + eps
    else:
        raise NotImplementedError

    # alpha_min path (automatically computed)
    amin_ridge = eps_ridge * amax_ridge
    amin_adv = eps_adv * amax_adv

    # Compute lasso paths
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps_lasso)
    coefs_lasso = np.concatenate([np.zeros([X.shape[1], 1]), coefs_lasso], axis=1)
    alphas_lasso = np.concatenate([1e2 * np.ones([1]), alphas_lasso], axis=0)

    # Compute ridge paths
    alphas_ridge = np.logspace(np.log10(amin_ridge), np.log10(amax_ridge), n_alphas)
    coefs_ridge_ = []
    for a in tqdm.tqdm(alphas_ridge):
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs_ridge_.append(ridge.coef_)
    coefs_ridge = np.stack((coefs_ridge_)).T

    alphas_adv = np.logspace(np.log10(amin_adv), np.log10(amax_adv), n_alphas)
    coefs_advtrain_l2_ = []
    coefs_advtrain_linf_ = []
    for a in tqdm.tqdm(alphas_adv):
        coefs = adversarial_training(X, y, p=2, eps=a)
        coefs_advtrain_l2_.append(coefs if coefs is not None else np.zeros(m))
        coefs = adversarial_training(X, y, p=np.inf, eps=a)
        coefs_advtrain_linf_.append(coefs if coefs is not None else np.zeros(m))  # p = infty seems ill conditioned
    coefs_advtrain_l2 = np.stack((coefs_advtrain_l2_)).T
    coefs_advtrain_linf = np.stack((coefs_advtrain_linf_)).T

    # Display results
    plot_coefs(alphas_lasso, coefs_lasso, 'lasso', r'$1/\lambda$')
    plot_coefs(alphas_ridge, coefs_ridge, 'ridge', r'$1/\lambda$')
    plot_coefs(alphas_adv, coefs_advtrain_l2, 'advtrain_l2', ylabel=False)
    plot_coefs(alphas_adv, coefs_advtrain_linf, 'advtrain_linf', ylabel=False)

    # Plot results
    plot_coefs_l1(coefs_lasso, 'lasso')
    plot_coefs_l1(coefs_advtrain_linf, 'advtrain_linf', ylabel=False)

    # Plot results with area
    if args.area:
        plot_coefs_l1(coefs_lasso, 'lasso_area', add_area=True)
        plot_coefs_l1(coefs_advtrain_linf, 'advtrain_linf_area', ylabel=False, add_area=True)
