import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datasets import get_dataset


if __name__ == '__main__':
    import argparse
    import matplotlib as mpl

    parser = argparse.ArgumentParser(description='Evaluate on dataset for grid of points.')
    parser.add_argument('-o', '--output_file', default='',
                        help='output file, if empty just plot the result.')
    parser.add_argument('-i', '--input_file', default='./results.csv',
                        help='output file.')
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    # Additional style
    mpl.rcParams['figure.figsize'] = 7, 5
    mpl.rcParams['figure.subplot.bottom'] = 0.2
    mpl.rcParams['figure.subplot.right'] = 0.99
    mpl.rcParams['figure.subplot.top'] = 0.95
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams['xtick.major.pad'] = 7

    plt.style.use(['../mystyle.mplsty'])

    # normalize
    X_test, y_test, _ = get_dataset('magic', test=True, n_samples=-1)

    df = df[((df['method'] == 'sqrt_lasso') & (df['factor'] == 0.1)) | (df['method'] != 'sqrt_lasso')]
    df = df[((df['method'] == 'advtrainlinf') & (df['factor'] == 0.5)) | (df['method'] != 'advtrainlinf')]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.boxplot(x='n_features', y='mse_test', hue='method',
                data=df, palette="Set3",
                hue_order=['minl1_norm', 'lasso cross-valid.', 'sqrt_lasso', 'advtrainlinf'])
    ax.set_xlabel(r'\# features, $p$')
    ax.set_ylabel(r' Test MSE')
    handles, _ = ax.get_legend_handles_labels()
    ax.set_xticklabels([400, 1000, 2000, 4000, 8000])
    ax.legend(handles, ['min $\ell_1$-norm', 'lasso CV', '$\sqrt{\mathrm{lasso}}$', 'adv. train $\ell_\infty$'], loc='upper right')
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()

