import os
import numpy as np
import pandas as pd
import itertools
from experiments.training import minl1norm_solution
import scipy.linalg as linalg

ls = ['-', ':', '--', '-.']
if __name__ == '__main__':
    import tqdm
    seeds = range(5)
    n_features = [1000, 2000, 4000, 8000, 16000]
    output_file = "results/magic_results.csv"
    folders = [(m, s, 'results/magic_m{}_seed{}'.format(m, s)) for s, m in itertools.product(seeds, n_features)]
    # Load file
    list_df = []
    for m, s, f in folders:
        df_aux = pd.read_csv(os.path.join(f, 'experiments.csv'))
        dset_path = os.path.join(f, 'dataset.npz')
        dset = np.load(dset_path)
        # compute min norm solutions
        X_train, X_test = dset['X_train'], dset['X_test']
        y_train, y_test = dset['y_train'], dset['y_test']
        coefs_minl2norm, _, _, _ = linalg.lstsq(X_train, y_train)
        coefs_minl1norm = minl1norm_solution(X_train, y_train)

        np.save(os.path.join(f, 'minl2norm'), coefs_minl2norm)
        np.save(os.path.join(f, 'minl1norm'), coefs_minl1norm)
        df_aux = df_aux.append({'method': 'minl2norm', 'file': 'minl2norm', 'alpha': 0, 'cv': False}, ignore_index=True)
        df_aux = df_aux.append({'method': 'minl1norm', 'file': 'minl1norm', 'alpha': 0, 'cv': False}, ignore_index=True)

        df_aux['file'] = f + '/' + df_aux['file']
        df_aux['dataset'] = os.path.join(f, 'dataset.npz')
        df_aux['seed'] = s
        df_aux['n_features'] = m
        if 'nmse_test' not in df_aux.keys():
            df_aux['nmse_test'] = -1  # Create columns that will be filled latter on
        if 'nmse_train' not in df_aux.keys():
            df_aux['nmse_train'] = -1
        if 'distance_minl2norm' not in df_aux.keys():
            df_aux['distance_minl2norm'] = -1
        if 'distance_minl1norm' not in df_aux.keys():
            df_aux['distance_minl1norm'] = -1
        list_df.append(df_aux)
    df = pd.concat(list_df, ignore_index=True)


    # filter method
    prev_dset_path = ''
    dset = 0
    for i, l in tqdm.tqdm(list(df.iterrows())):
        dset_path = l['dataset']
        if dset_path != prev_dset_path:
            dset = np.load(dset_path)
            prev_dset_path = dset_path
            X_train, X_test = dset['X_train'], dset['X_test']
            y_train, y_test = dset['y_train'], dset['y_test']
            coefs_minl2norm, _, _, _ = linalg.lstsq(X_train, y_train)
            coefs_minl1norm = minl1norm_solution(X_train, y_train)

        fname = l['file']

        theta = np.load(fname + '.npy')
        y_pred = X_test @ theta
        nmse = np.mean((y_test - y_pred) ** 2) / np.mean((y_test) ** 2)
        df.loc[i, 'nmse_test'] = nmse

        y_pred_train = X_train @ theta
        nmse_train = np.mean((y_train - y_pred_train) ** 2) / np.mean((y_test) ** 2)
        df.loc[i, 'nmse_train'] = nmse_train
        df.to_csv(output_file, index=False)

        df.loc[i, 'distance_minl2norm'] = (theta - coefs_minl2norm).std()
        df.loc[i, 'distance_minl1norm'] = (theta - coefs_minl1norm).std()


