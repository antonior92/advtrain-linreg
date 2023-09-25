# You can download and extract the MAGIC dataset by
# > wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
# > tar -xvf BASIC_GWAS.tar.gz
import os
import pandas as pd
import sklearn.model_selection
import numpy as np
import numpy.random as rnd
import tqdm
from training import adversarial_training, lasso_cvx, ridge

from sklearn.linear_model import ElasticNetCV

l2advtrain = lambda xx, yy, ee: adversarial_training(xx, yy, 2, ee)
linfadvtrain = lambda xx, yy, ee: adversarial_training(xx, yy, np.Inf, ee)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate on MAGIC dataset for grid of points.')
    parser.add_argument('-i', '--input_folder', default='../../WEBSITE/DATA',
                        help='input folder containing magic dataset.')
    parser.add_argument('-o', '--output_folder', default='./out/results/magic',
                        help='output folder.')
    parser.add_argument('--test_size', type=int, default=252,
                       help='number of test samples in the experiment. The number of training points will be 504 minus'
                            'this quantity.')
    parser.add_argument('-g', '--grid', type=int, default=0,
                        help='number of points each solver will be evaluated on')
    parser.add_argument('-m', '--num_features', type=int, default=2000,
                        help='How many features to use. Naturally deal with 55067 features (genotypes).'
                             'When this is not the case use subset of the features.')
    parser.add_argument('-p', '--output_phenotype', default='HET_2', type=str,
                        help='which phenotype will be predicted')
    parser.add_argument('-r', '--random_state', default=0, type=int,
                        help='random seed.')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    founder_names = ["Banco", "Bersee", "Brigadier", "Copain", "Cordiale", "Flamingo",
                     "Gladiator", "Holdfast", "Kloka", "MarisFundin", "Robigus", "Slejpner",
                     "Soissons", "Spark", "Steadfast", "Stetson"]


    # Genotype
    genotype = pd.read_csv(os.path.join(args.input_folder, 'MAGIC_IMPUTED_PRUNED/MAGIC_imputed.pruned.traw'), sep='\t')
    genotype.set_index('SNP', inplace=True)
    genotype = genotype.iloc[:, 5:]
    colnames = genotype.keys()
    new_colnames = [c.split('_')[0] for c in colnames]
    genotype.rename(columns={c: new_c for c, new_c in zip(colnames, new_colnames)}, inplace=True)
    genotype = genotype.transpose()

    # Phenotype
    phenotype = pd.read_csv(os.path.join(args.input_folder, 'PHENOTYPES/NDM_phenotypes.tsv'), sep='\t')
    phenotype.set_index('line_name', inplace=True)
    phenotype.drop(founder_names, inplace=True)
    del phenotype['line_code']

    # Make genotype have the same index as phenotype
    genotype = genotype.reindex(phenotype.index,)

    # Replace NAs
    genotype = genotype.fillna(genotype.mean(axis=0))
    phenotype = phenotype.fillna(phenotype.mean(axis=0))

    # Formulate problem
    X = genotype.values
    y = phenotype[args.output_phenotype].values
    n_samples, n_genes = X.shape

    # Reduce size (just for testing quickly)
    rng = rnd.RandomState(seed=args.random_state)
    num_features = np.minimum(args.num_features, n_genes)
    selected_features_before = np.arange(n_genes) < args.num_features
    selected_features = rnd.permutation(selected_features_before)
    X = X[:, selected_features]

    # Train-val split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=args.test_size, random_state=0)

    # Rescale
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std


    np.savez(os.path.join(args.output_folder, 'dataset'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    def compute_all_values(f, name, df, min_scale=-3, max_scale=6):
        # Compute ridge paths
        alphas = np.logspace(min_scale, max_scale, args.grid)
        for a in tqdm.tqdm(alphas):
            theta = f(X_train, y_train, a)
            fname = name+'_{:0.8}'.format(a)
            np.save(os.path.join(args.output_folder, fname), theta)
            df = df.append({'method': name, 'alpha': a, 'file': fname, 'cv': False}, ignore_index=True)
            df.to_csv(os.path.join(args.output_folder, 'experiments.csv'), index=False)
        return df

    df = pd.DataFrame(columns=['method', 'alpha', 'file', 'cv'])
    df = compute_all_values(ridge, 'ridge', df, min_scale=-1, max_scale=6)
    df = compute_all_values(lasso_cvx, 'lasso', df, min_scale=-5, max_scale=1)
    df = compute_all_values(linfadvtrain, 'linfadvtrain', df, min_scale=-6, max_scale=0)
    df = compute_all_values(l2advtrain, 'l2advtrain', df, min_scale=-4, max_scale=1)

    # Add values related to the cross validation
    regr = ElasticNetCV(cv=10, random_state=0, l1_ratio=1)
    regr.fit(X_train, y_train)
    fname = 'lasso_{:0.8}'.format(regr.alpha_)
    np.save(os.path.join(args.output_folder, fname), regr.coef_)
    y_pred = X_test @ regr.coef_
    mse = np.mean((y_test - y_pred) ** 2)
    print(mse)
    print(sum(np.abs(regr.coef_ > 0)))
    df = df.append({'method': 'lasso', 'alpha': regr.alpha_, 'file': fname, 'cv': True}, ignore_index=True)
    df.to_csv(os.path.join(args.output_folder, 'experiments.csv'), index=False)

    theta = lasso_cvx(X_train, y_train, regr.alpha_)
    y_pred = X_test @ theta
    mse = np.mean((y_test - y_pred) ** 2)
    print(mse)
    print(sum(np.abs(theta > 0)))
    print(max(np.abs(regr.coef_ - theta)))

