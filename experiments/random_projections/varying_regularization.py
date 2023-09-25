#%% imports
import numpy as np
import tqdm
import cvxpy as cp
import matplotlib.pyplot as plt
from adversarial_attack import compute_q
import matplotlib as mpl
from scipy.linalg import svd

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
    constr = [cp.pnorm(Vh.T @ Vh @ X.T @ var, p=p) <= 1, ]
    prob = cp.Problem(obj, constr)
    result = prob.solve()
    return 1 / (n * np.max(np.abs(var.value)))


def adversarial_training_rp(X, S, y, p, eps, **kwargs):
    """Compute parameter for linear model trained adversarially with unitary p-norm.

    :param X:
        A numpy array of shape = (n_points, inp_dim) containing the inputs
    :param S:
        A numpy array of shape = (n_features, inp_dim) containing the random projection matrix
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param p:
        The p-norm the adversarial attack is bounded. `p` gives which p-norm is used
        p = 2 is the euclidean norm. `p` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :param eps:
        The magnitude of the attack during the trainign
    :return:
        An array containing the adversarially estimated parameter.
    """
    n_points, inp_dim = X.shape
    n_features, inp_dim = S.shape

    q = compute_q(p)

    # Formulate problem
    param = cp.Variable(n_features)
    param_d = S.T @ param
    param_norm = cp.pnorm(param_d, p=q)
    abs_error = cp.abs(X @ param_d - y)
    adv_loss = 1 / n_points * cp.sum((abs_error + eps * param_norm) ** 2)

    prob = cp.Problem(cp.Minimize(adv_loss))
    try:
        prob.solve(**kwargs)
        param0 = param.value
    except:
        param0 = np.zeros(n_features)

    return param0


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

    if args.save:
        def display(name):
            plt.savefig(args.save + f'{name}.pdf')
    else:
        def display(_name):
            plt.show()

    n_train = 60
    n_test = 60
    inp_dim = 1000
    param_norm = 1
    n_features = 200
    alphas = np.logspace(args.alpha_max - args.alpha_range, args.alpha_max, args.n_alphas)
    n_coefs = len(alphas)


    def compute_coefs(seed):
        # Define problem
        rng = np.random.RandomState(seed)
        # Generate data
        X = rng.randn(n_train, inp_dim)
        X_test = rng.randn(n_test, inp_dim)
        true_param = param_norm / np.sqrt(inp_dim) * rng.randn(inp_dim)
        e = rng.randn(n_train)
        e_test = rng.randn(n_test)
        y = X @ true_param + args.noise_std * e
        y_test = X_test @ true_param + args.noise_std * e_test
        S = rng.normal(size=(n_features, inp_dim))
        S = np.sign(S)

        #  Compute coeffs
        coefs_ = []
        for a in tqdm.tqdm(alphas):
            theta = adversarial_training_rp(X, S, y, args.ord, a)
            coefs_.append(theta)
        coefs = np.stack((coefs_)).T

        coefs_d = S.T @ coefs
        mse_train = np.mean((X @ coefs_d - y[:, None]) ** 2, axis=0)
        test_error = X_test @ coefs_d - y_test[:, None]
        mse_test = np.mean(test_error ** 2, axis=0)
        max_alpha = get_max_alpha(X, S, y, p=args.ord)

        return mse_train, mse_test, max_alpha

    max_alphas_list = []
    mse_train_list = []
    mse_test_list = []
    for seed in range(args.n_reps):
        mse_train, mse_test, max_alpha = compute_coefs(seed)
        max_alphas_list.append(max_alpha)
        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)

    #%% Plot alpha
    if args.ord == np.inf:
        label = r'$\ell_{\infty}$-adv. train'
        color = 'green'
    elif args.ord == 2:
        label = r'$\ell_2$-adv. train'
        color = 'blue'
    else:
        raise ValueError(f'ord={args.ord} not supported')

    fig, ax = plt.subplots()
    m = np.median(mse_train_list, axis=0)
    lerr = m - np.quantile(mse_train_list, 0.25, axis=0)
    uerr = np.quantile(mse_train_list, 0.75, axis=0) - m
    ax.errorbar(1/alphas, m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                marker='o', markersize=3.5, ls='', label=label, color=color)
    plt.axvline(1/np.mean(max_alphas_list), ls='--', color='k')
    ax.set_xlabel(r'$1/\delta$')
    ax.set_ylabel(r'Train MSE')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    display('train_mse')


    fig, ax = plt.subplots()
    m = np.median(mse_test_list, axis=0)
    lerr = m - np.quantile(mse_test_list, 0.25, axis=0)
    uerr = np.quantile(mse_test_list, 0.75, axis=0) - m
    ax.errorbar(1/alphas, m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                marker='o', markersize=3.5, ls='', label=label, color=color)
    ax.set_xlabel(r'$1/\delta$')
    ax.set_ylabel(r'Test MSE')
    ax.set_xscale('log')
    plt.legend()
    display('test_mse')




