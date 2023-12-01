#%% imports
import numpy as np
import tqdm
from advtrain import compute_q
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib as mpl

plt.style.use(['../mystyle.mplsty'])
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.left'] = 0.2
mpl.rcParams['figure.subplot.bottom'] = 0.2
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['xtick.major.pad'] = 7

def get_max_alpha(X, S, y, p=2):
    n, m = X.shape
    q = compute_q(p)

    var = cp.Variable(n)

    U, _, Vh = svd(S, full_matrices=False)

    obj = cp.Maximize(var @ y)
    constr = [cp.pnorm(Vh.T @ Vh @ X.T @ var, p=p) <= 1,]
    prob = cp.Problem(obj, constr)
    result = prob.solve()
    return 1 / (n * np.max(np.abs(var.value)))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Plot parameters profile')
    parser.add_argument('--save', default='',
                        help='save plot in the given folder (do not write extension). By default just show it.')
    parser.add_argument('-p', '--ord', default=np.inf, type=float,
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('--n_alphas', type=int, default=5,
                        help='number of values of alpha')
    parser.add_argument('--n_reps', type=int, default=2,
                        help='number of repetitions of the experiment')
    parser.add_argument('--alpha_max', type=float, default=1,
                       help='maximum regularization parameter is 10 ** alpha_max')
    parser.add_argument('--alpha_range', type=float, default=5,
                       help='minimum regularization parameter is 10**(alpha_max - alpha_range)')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--n_features', type=int, default=200,
                        help='standard deviation of the additive noise added.')
    args, unk = parser.parse_known_args()





    #%% Define constants
    n_train = 60
    n_test = 60
    inp_dim = 1000
    param_norm = 1
    noise_std = 0.1
    seed = 1
    pnorm = np.inf

    #%%  Define problem
    rng = np.random.RandomState(seed)

    # Generate data
    X = rng.randn(n_train, inp_dim)
    X_test = rng.randn(n_test, inp_dim)
    true_param = param_norm / np.sqrt(inp_dim) * rng.randn(inp_dim)
    e = rng.randn(n_train)
    e_test = rng.randn(n_test)
    y = X @ true_param + args.noise_std * e
    y_test = X_test @ true_param + args.noise_std * e_test

    #%% Compute thresholds
    all_nfeatures = np.linspace(110, inp_dim, 10)

    max_alphas_l2 = []
    max_alphas_linf = []
    for n_features in tqdm.tqdm(all_nfeatures):
        max_alphas_l2_ = []
        max_alphas_linf_ = []
        for i in range(args.n_reps):
            S = rng.normal(size=(int(n_features), inp_dim))
            S = np.sign(S)
            max_alphas_l2_.append(get_max_alpha(X, S, y, p=2))
            max_alphas_linf_.append(get_max_alpha(X, S, y, p=np.inf))
        max_alphas_l2.append(max_alphas_l2_)
        max_alphas_linf.append(max_alphas_linf_)


    #%% Plot thresholds
    fig, ax = plt.subplots()
    max_alphas = max_alphas_l2
    m = np.median(max_alphas, axis=1)
    lerr = m - np.quantile(max_alphas, 0.25, axis=1)
    uerr = np.quantile(max_alphas, 0.75, axis=1) - m
    ax.errorbar(all_nfeatures, m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                marker='o', markersize=3.5, ls='', label=r'min. $\ell_2$-norm', color='blue')
    ax.set_xlabel(r'\# features, $p$')
    ax.set_ylabel(r'$\bar \delta$')
    plt.legend()
    if args.save:
        plt.savefig(args.save + f'/thresholdl2_prop_randproj.pdf')
    else:
        plt.show()

    fig, ax = plt.subplots()
    max_alphas = max_alphas_linf
    m = np.median(max_alphas, axis=1)
    lerr = m - np.quantile(max_alphas, 0.25, axis=1)
    uerr = np.quantile(max_alphas, 0.75, axis=1) - m
    ax.errorbar(all_nfeatures, m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                marker='o', markersize=3.5, ls='', label=r'min. $\ell_1$-norm', color='green')
    ax.set_xlabel(r'\# features, $p$')
    ax.set_ylabel(r'$\bar \delta$')
    plt.legend()
    if args.save:
        plt.savefig(args.save + f'/thresholdlinf_prop_randproj.pdf')
    else:
        plt.show()

