import numpy as np
from datasets import get_dataset, DSETS
from training import MinimumNorm
from tqdm import tqdm
from utils import get_dataframe_fields, get_evaluation_dataframe
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot parameters profile')
    parser.add_argument('-o', '--output_file', default='./output.csv',
                        help='output file.')
    parser.add_argument('--dset', choices=DSETS, default='gaussian')
    parser.add_argument('--save', default='',
                        help='save plot in the given folder (do not write extension). By default just show it.')
    parser.add_argument('--n_points', type=int, default=30,
                        help='number of points in the plot')
    parser.add_argument('--n_rep', type=int, default=5,
                        help='number of repetitions for each point')
    parser.add_argument('--use_asymptotics', action='store_true',
                        help='use asymptotics for the computation of the reference')
    parser.add_argument('--max_range', type=int, default=2,
                        help='log(max_range) * n_train will be the maximum number of features')
    parser.add_argument('--ref_alpha', type=float, default=0.01,
                       help='reference alpha for the asymptotics as a proportion of Expectation of x')
    args, unk = parser.parse_known_args()


    # Check dataset size
    X, y, _ = get_dataset(args.dset, seed=0, n_features=20)
    n_train = X.shape[0]


    all_proportions = np.logspace(0.2, args.max_range, args.n_points)
    df = pd.DataFrame(columns=get_dataframe_fields())
    df2 = pd.DataFrame(columns=['prop', 'seed', 'method'])
    for prop in tqdm(all_proportions):
        for i in range(args.n_rep):
            n_features = int(prop * n_train)
            X, y, _ = get_dataset(args.dset, seed=i, n_features=n_features, test=False)
            X_test, y_test, _ = get_dataset(args.dset, seed=i, n_features=n_features, test=True)
            # compute min norm interpolators
            minnorm2 = MinimumNorm(X, y, 2)
            minnorminf = MinimumNorm(X, y, 1)
            param2 = minnorm2()
            paraminf = minnorminf()
            # Evaluate
            df = df.append(get_evaluation_dataframe(param2[:, None], X, y, X_test, y_test).iloc[0], ignore_index=True)
            df2 = df2.append({'prop': prop, 'seed': i, 'method': 'minl2norm',
                              'threshold': minnorm2.adv_radius()}, ignore_index=True)
            df = df.append(get_evaluation_dataframe(paraminf[:, None], X, y, X_test, y_test).iloc[0], ignore_index=True)
            df2 = df2.append({'prop': prop, 'seed': i, 'method': 'minlinfnorm',
                              'threshold': minnorminf.adv_radius()}, ignore_index=True)

    df = pd.concat([df2, df], axis=1)
    df.to_csv(args.output_file, index=False)