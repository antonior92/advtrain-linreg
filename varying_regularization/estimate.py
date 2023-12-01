import numpy as np
import tqdm
from training import Ridge
from advtrain import AdversarialTraining
from datasets import get_dataset, DSETS
from sklearn.linear_model import lasso_path
from training import get_max_alpha
from utils import get_evaluation_dataframe

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate on dataset for grid of points.')
    parser.add_argument('-o', '--output_file', default='./output.csv',
                        help='output file.')
    parser.add_argument('-p', '--ord',  default=[2.0, np.inf], type=float, nargs='+',
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('-e', '--epsilon', default=[0.0001, 0.001, 0.01, 0.1, 1.0], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial attack')
    parser.add_argument('-g', '--grid', type=int, default=0,
                        help='number of points each solver will be evaluated on')
    parser.add_argument('--n_alpha', type=int, default=100,
                        help='number of values of alpha')
    parser.add_argument('--alpha_max', type=float, default=1,
                       help='maximum regularization parameter is 10 ** alpha_max')
    parser.add_argument('--alpha_range', type=float, default=5,
                       help='minimum regularization parameter is 10**(alpha_max - alpha_range)')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed.')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--n_features', type=int, default=200,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--dset', choices=DSETS, default='gaussian')
    parser.add_argument('-m', '--method', choices=['advtrain_l2', 'advtrain_linf', 'lasso', 'ridge'],
                        default='advtrain_l2')
    args = parser.parse_args()
    print(args)

    X, y, _ = get_dataset(args.dset, seed=args.seed, test=False, noise_std=args.noise_std, n_features=args.n_features)
    X_test, y_test, _ = get_dataset(args.dset, seed=args.seed, test=True, noise_std=args.noise_std, n_features=args.n_features)

    max_ainf = get_max_alpha(X, y, np.inf)
    max_a2 = get_max_alpha(X, y, 2)

    alphas = np.logspace(args.alpha_max - args.alpha_range, args.alpha_max, args.n_alpha)
    if 'lasso' in args.method:
        print(len(alphas))
        alphas_lasso, coefs, _ = lasso_path(X, y, fit_intercept=False,
                                                  normalize=False, tol=1e-5, max_iter=10000, alphas=alphas)
        alphas =alphas_lasso
    else:
        if args.method == 'advtrain_l2':
            f = AdversarialTraining(X, y, 2)
        elif args.method == 'advtrain_linf':
            f = AdversarialTraining(X, y, np.inf)
        else:
            f = Ridge(X, y)
        coefs_ = []
        for a in tqdm.tqdm(alphas):
            theta = f(a)
            coefs_.append(theta)
        coefs = np.stack((coefs_)).T
    df = get_evaluation_dataframe(coefs, X, y, X_test, y_test)
    df['method'] = np.array(args.n_alpha * [args.method])
    df['noise_std'] = np.array(args.n_alpha * [args.noise_std])
    df['max_ainf'] = np.array(args.n_alpha * [max_ainf])
    df['max_a2'] = np.array(args.n_alpha * [max_a2])
    df['alpha'] = alphas
    df.to_csv(args.output_file, index=False)
