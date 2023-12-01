from sklearn import datasets
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import pandas as pd
import numpy.random as rnd
import os
import sklearn
import sklearn.model_selection

DSETS = ['gaussian', 'latent', 'rff', 'random_projections', 'gaussian_sparse', 'magic']

magic = None


class LoadMagic():

    def __init__(self, input_folder='WEBSITE/DATA', output_phenotype = 'HET_2', test_size=252):
        # You can download and extract the MAGIC dataset by
        # > wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
        # > tar -xvf BASIC_GWAS.tar.gz

        # Load data
        founder_names = ["Banco", "Bersee", "Brigadier", "Copain", "Cordiale", "Flamingo",
                         "Gladiator", "Holdfast", "Kloka", "MarisFundin", "Robigus", "Slejpner",
                         "Soissons", "Spark", "Steadfast", "Stetson"]

        # Genotype
        genotype = pd.read_csv(os.path.join(input_folder, 'MAGIC_IMPUTED_PRUNED/MAGIC_imputed.pruned.traw'),
                               sep='\t')
        genotype.set_index('SNP', inplace=True)
        genotype = genotype.iloc[:, 5:]
        colnames = genotype.keys()
        new_colnames = [c.split('_')[0] for c in colnames]
        genotype.rename(columns={c: new_c for c, new_c in zip(colnames, new_colnames)}, inplace=True)
        genotype = genotype.transpose()

        # Phenotype
        phenotype = pd.read_csv(os.path.join(input_folder, 'PHENOTYPES/NDM_phenotypes.tsv'), sep='\t')
        phenotype.set_index('line_name', inplace=True)
        phenotype.drop(founder_names, inplace=True)
        del phenotype['line_code']

        # Make genotype have the same index as phenotype
        genotype = genotype.reindex(phenotype.index, )

        # Replace NAs
        genotype = genotype.fillna(genotype.mean(axis=0))
        phenotype = phenotype.fillna(phenotype.mean(axis=0))

        # Formulate problem
        X = np.array(genotype.values)
        y = np.array(phenotype[output_phenotype].values)

        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=0)
        self.info = {}


    def __call__(self, test=False, n_samples=None):
        X = self.X_test if test else self.X_train
        y = self.y_test if test else self.y_train
        if n_samples is None:
            return X, y, self.info
        else:
            return X[:n_samples], y[:n_samples], self.info


def generate_random_ortogonal(p, d, rng):
    """Generate random W with shape (p, d) such that `W.T W = p / d I_d`."""
    aux = rng.randn(p, d)
    q, r = np.linalg.qr(aux, mode='reduced')
    return q


def get_dataset(dset, seed=1, parameter_norm=1, noise_std=1, test=False, n_features=200, n_samples=60, sparsity=1, input_folder='WEBSITE/DATA'):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    input_folder = os.path.join(script_dir, input_folder)  # absolute path for dataset

    if test and dset in ['gaussian', 'latent', 'random_projections', 'gaussian_sparse']:
        seed = seed + 9999999  # For artificial datasets, use different seed for test set

    rng = np.random.RandomState(seed)

    if dset == 'gaussian':
        beta = parameter_norm / np.sqrt(n_features) * rng.randn(n_features)
        X = rng.randn(n_samples, n_features)
        rng = np.random.RandomState(seed)
        # Generate output with random additive noise
        e = rng.randn(n_samples)
        y = X @ beta + noise_std * e
        # Save internal features
        info = {'beta': beta, 'noise': e}

    elif dset == 'gaussian_sparse':
        s = sparsity
        beta = np.zeros(n_features)
        beta[:s] = parameter_norm / np.sqrt(s) * rng.randn(s)
        X = rng.randn(n_samples, n_features)
        rng = np.random.RandomState(seed)
        # Generate output with random additive noise
        e = rng.randn(n_samples)
        e = np.sign(e)
        y = X @ beta + noise_std * e
        # Save internal features
        info = {'beta': beta, 'noise': e}

    elif dset == 'latent':
        n_latent = 1
        factor = np.sqrt(n_features / n_latent)
        theta = parameter_norm / np.sqrt(n_latent) * rng.randn(n_latent)
        w = factor * generate_random_ortogonal(n_features, n_latent, rng)
        z = rng.randn(n_samples, n_latent)
        u = rng.randn(n_samples, n_features)
        e = rng.randn(n_samples)
        y = z @ theta + noise_std * e
        X = z @ w.T + u
        # Save internal features
        info = {'theta': theta, 'w': w, 'z': z, 'u': u, 'noise': e}

    elif dset == 'random_projections':
        X_, y_ = datasets.load_diabetes(return_X_y=True)
        X_ = (X_ - X_.mean(axis=0)) / X_.std(axis=0)
        y = (y_ - y_.mean()) / y_.std()

        # Random rademacher projection
        seed = 1
        rng = np.random.RandomState(seed)
        # Generate rademacher variable with scipy
        S = rng.normal(size=(10, n_features))
        S = np.sign(S)
        X = X_ @ S
        X = X[:n_samples]
        y = y[:n_samples]
        # Save internal features
        info = {'S': S, 'X_': X_}

    elif dset == 'rff':
        X_, y_ = datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X_, y_, test_size=100, random_state=0)
        X_ = X_test if test else X_train
        y_ = y_test if test else y_train
        X_ = (X_ - X_.mean(axis=0)) / X_.std(axis=0)
        y = (y_ - y_.mean()) / y_.std()

        rbf_feature = RBFSampler(n_components=n_features, gamma=0.1, random_state=seed)
        X_all = rbf_feature.fit_transform(X_)
        # use only subsample
        X = X_all[:n_samples]
        y = y[:n_samples]
        # Save internal features
        info = {'X_': X_, 'RBF': rbf_feature}

    elif dset == 'magic':
        global magic
        if magic is None:
            magic = LoadMagic(input_folder=input_folder);
        X, y, info = magic(n_samples=n_samples, test=test)
        n_samples, n_genes = X.shape
        # Reduce size (just for testing quickly)
        rng = rnd.RandomState(seed=seed)
        n_features = np.minimum(n_features, n_genes)
        selected_features_before = np.arange(n_genes) < n_features
        selected_features = rng.permutation(selected_features_before)
        X = X[:, selected_features]

    else:
        raise ValueError('Unknow dataset')
    return X, y, info