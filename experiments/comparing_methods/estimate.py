import numpy as np
from experiments.datasets import get_dataset, DSETS
from sklearn.linear_model import ElasticNetCV
import tqdm
from experiments.utils import get_dataframe_fields, get_evaluation_dataframe
import pandas as pd
from experiments.training import MinimumNorm, sqrt_lasso
from advtrain import AdversarialTraining


def estimate_adv_radius(X, _y, seed=1, n_samples=10):
    """Estimate the adversarial radius of a dataset X."""

    n_train, n_features = X.shape
    rng = np.random.RandomState(seed)

    eps = rng.randn(n_train, n_samples)
    eps /= np.linalg.norm(eps, axis=0, keepdims=True, ord=1)

    adv_radius = np.max(np.abs(eps.T @ X), axis=0).mean()
    return adv_radius


def estimate_sqrt_lasso_reg(X, _y, seed=1, n_samples=10):
    """Estimate the square root of the lasso regularization parameter of a dataset X."""

    n_train, n_features = X.shape
    rng = np.random.RandomState(seed)

    eps = rng.randn(n_train, n_samples)
    eps /= np.linalg.norm(eps, axis=0, keepdims=True, ord=2)

    reg_param = np.max(np.abs(eps.T @ X), axis=0).mean() / np.sqrt(n_train)
    return reg_param


def comput_advradius_zero(X, y):
    """Estimate the adversarial radius of a dataset (X, y) that yield zero solution"""

    return np.max(np.abs(y.T @ X), axis=0) / np.linalg.norm(y, axis=0, keepdims=True, ord=1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate on dataset for grid of points.')
    parser.add_argument('-o', '--output_file', default='./results.csv',
                        help='output file.')
    parser.add_argument('-p', '--ord',  default=[2.0, np.inf], type=float, nargs='+',
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed.')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--max_range', type=int, default=0.1,
                        help='log(max_range) * n_train will be the maximum number of features')
    parser.add_argument('--dset', choices=DSETS, default='magic')
    parser.add_argument('--n_rep', type=int, default=5,
                        help='number of repetitions for each point')
    parser.add_argument('--n_features', default=[100, 300, 500, 1000, 2000], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial attack')
    parser.add_argument('--n_samples', default=60, type=int,
                        help='set to -1 to use all samples')
    args = parser.parse_args()
    if args.n_samples == -1:
        args.n_samples = None

    df = pd.DataFrame(columns=get_dataframe_fields())
    df2 = pd.DataFrame(columns=['n_features', 'seed', 'method',
                                'min_norm_radius', 'radius', 'zero_radius', 'reg_param'])
    for n_features in args.n_features:
        print(f'---- n_features = {n_features} ----')
        for i in tqdm.tqdm(range(args.n_rep)):
            X, y, _ = get_dataset(args.dset, seed=i, test=False, n_features=n_features, n_samples=args.n_samples)
            X_test, y_test, _ = get_dataset(args.dset, seed=i, test=True, n_features=n_features, n_samples=args.n_samples)

            # Lasso regression
            lasso = ElasticNetCV(cv=5, random_state=0, l1_ratio=1)
            lasso.fit(X, y)
            df = df.append(get_evaluation_dataframe(lasso.coef_[:, None], X, y, X_test, y_test).iloc[0], ignore_index=True)
            df2 = df2.append({'n_features': n_features, 'seed': i, 'method': 'lasso'}, ignore_index=True)

            # Minimum L1 norm
            minl1_norm = MinimumNorm(X, y, 1)
            minl1_norm_coefs = minl1_norm()
            df = df.append(get_evaluation_dataframe(minl1_norm_coefs[:, None], X, y, X_test, y_test).iloc[0],
                           ignore_index=True)
            df2 = df2.append({'n_features': n_features, 'seed': i, 'method': 'minl1_norm'}, ignore_index=True)

            # Adversarial training
            opt_radius = estimate_adv_radius(X, y)
            radius_0 = comput_advradius_zero(X, y)[0]
            advtrainlinf = AdversarialTraining(X, y, np.inf)

            for factor in [0.1, 0.5, 1]:
                advtrainlinf_coefs = advtrainlinf(eps=factor * opt_radius)
                df = df.append(get_evaluation_dataframe(advtrainlinf_coefs[:, None], X, y, X_test, y_test).iloc[0],
                               ignore_index=True)
                df2 = df2.append({'n_features': n_features, 'seed': i, 'method': 'advtrainlinf',
                                  'min_norm_radius': minl1_norm.adv_radius(), 'opt_radius': opt_radius,
                                  'zero_radius': radius_0, 'factor': factor}, ignore_index=True)

            # Squared root lasso
            opt_reg_param = estimate_sqrt_lasso_reg(X, y)
            for factor in [0.05, 0.1, 0.5]:
                sqrt_lasso_coefs = sqrt_lasso(X, y, factor * opt_reg_param)
                df = df.append(get_evaluation_dataframe(sqrt_lasso_coefs[:, None], X, y, X_test, y_test).iloc[0], ignore_index=True)
                df2 = df2.append({'n_features': n_features, 'seed': i, 'method': 'sqrt_lasso',
                                  'opt_reg_param': opt_reg_param, 'factor': factor}, ignore_index=True)

        df_all = pd.concat([df2, df], axis=1)
        df_all.to_csv(args.output_file, index=False)