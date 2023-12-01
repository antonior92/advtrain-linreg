import numpy as np
import pandas as pd
from adversarial_attack import compute_adv_attack


eps = [0.0001, 0.001, 0.01, 0.1, 1.0]
ords = [2.0, np.inf]

def get_quantiles(xaxis, r, quantileslower=0.25, quantilesupper=0.75):
    new_xaxis, inverse, counts = np.unique(xaxis, return_inverse=True, return_counts=True)

    r_values = np.zeros([len(new_xaxis), max(counts)])
    secondindex = np.zeros(len(new_xaxis), dtype=int)
    for n in range(len(xaxis)):
        i = inverse[n]
        j = secondindex[i]
        r_values[i, j] = r[n]
        secondindex[i] += 1
    m = np.median(r_values, axis=1)
    lerr = m - np.quantile(r_values, quantileslower, axis=1)
    uerr = np.quantile(r_values, quantilesupper, axis=1) - m
    return new_xaxis, m, lerr, uerr


def plot_errorbar(df, xname, yname, ax, label, color='blue'):
    x = df[xname].values
    y = df[yname].values
    new_x, m, lerr, uerr = get_quantiles(x, y)
    ax.errorbar(new_x, m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                marker='o', markersize=3.5, ls='', label=label, color=color)


def get_quantiles_df(df, xname, yname):
    x = df[xname].values
    y = df[yname].values
    new_x, m, lerr, uerr = get_quantiles(x, y)
    return new_x, m, lerr, uerr



def get_dataframe_fields():
    return ['mse_train', 'mse_test'] + ['param_l2norm', 'param_l1norm'] + ['advrisk-{:.1f}-{:.10f}'.format(p, e) for p in ords for e in eps]


def get_evaluation_dataframe(coefs, X, y, X_test, y_test):
    n_features, n_coefs = coefs.shape
    mse_train = np.mean((X @ coefs - y[:, None]) ** 2, axis=0)
    test_error = X_test @ coefs - y_test[:, None]
    mse_test = np.mean((X_test @ coefs - y_test[:, None]) ** 2, axis=0)
    input_l2norm = np.mean(np.linalg.norm(X, axis=1, ord=2))
    input_linfnorm = np.mean(np.max(np.abs(X), axis=1))

    df_aux = pd.DataFrame({'mse_train': mse_train, 'mse_test': mse_test, 'input_l2norm': input_l2norm,'input_linfnorm': input_linfnorm})

    df_aux['param_l2norm'] = np.linalg.norm(coefs, axis=0, ord=2)
    df_aux['param_l1norm'] = np.linalg.norm(coefs, axis=0, ord=1)

    # Compute adversarial risk
    for p in ords:
        for e in eps:
            df_aux['advrisk-{:.1f}-{:.10f}'.format(p, e)] = np.zeros(n_coefs)
    for p in ords:
        normalization = input_l2norm if p == 2.0 else input_linfnorm
        # Compute ord
        for i in range(n_coefs):
            delta_x = compute_adv_attack(test_error[:, i], coefs[:, i], ord=p)
            for e in eps:
                # Estimate adversarial arisk
                delta_X = normalization * e * delta_x
                r = np.mean((y_test - (X_test + delta_X) @ coefs[:, i]) ** 2, axis=0)
                df_aux['advrisk-{:.1f}-{:.10f}'.format(p, e)][i] = r
    return df_aux