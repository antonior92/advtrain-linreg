import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as plticker


def get_median_and_quantiles(x, y):
    new_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    y_values = np.zeros([len(new_x), max(counts)])
    secondindex = np.zeros(len(new_x), dtype=int)
    for n in range(len(x)):
        i = inverse[n]
        j = secondindex[i]
        y_values[i, j] = y[n]
        secondindex[i] += 1
    m = np.median(y_values, axis=1)
    lerr = m - np.quantile(y_values, 0.25, axis=1)
    uerr = np.quantile(y_values, 0.75, axis=1) - m
    return new_x, m, lerr, uerr


def plot_error_bar(new_x, m, lerr, uerr, ax, lbl, logy=False):
    if logy:
        l, = ax.plot(new_x, np.log10(m))
        ax.fill_between(new_x, np.log10(m - lerr), np.log10(m + uerr), color=l.get_color(), alpha=0.4, label=lbl)
    else:
        l, = ax.plot(new_x, m)
        ax.fill_between(new_x, m - lerr, m + uerr, color=l.get_color(), alpha=0.4, label=lbl)


if __name__ == '__main__':
    import pandas as pd
    import os
    import matplotlib as mpl
    df = pd.read_csv('results/magic_results.csv')
    plt.style.use(['../mystyle.mplsty', 'figstyle.mplsty'
                   ])
    save = 'plots'


    def show(name):
        if save:
            plt.savefig(os.path.join(save, name))
        else:
            plt.show()

    methods_pretty_names = {'minl2norm': 'min $\ell_2$-norm',
                            'ridge': 'ridge',
                            'l2advtrain': 'adv. train $\ell_2$',
                            'minl1norm': 'min $\ell_1$-norm',
                            'lasso': 'lasso',
                            'linfadvtrain': 'adv. train $\ell_\infty$'
                            }

    df_filtered = df[df['n_features'] == 16000]
    methods = ['ridge', 'l2advtrain','lasso', 'linfadvtrain']
    methods_extended = ['minl2norm', 'ridge', 'l2advtrain', 'minl1norm','lasso', 'linfadvtrain']
    tps = ['nmse_test', 'nmse_train']
    log_x = {t: True for t in ['mse_test', 'mse_train']}
    log_y = {'nmse_test': False, 'nmse_train': True}
    offsets = {'ridge': 1, 'lasso': 1e-6, 'l2advtrain': 1e-5, 'linfadvtrain': 1e-6}

    # Plot test error
    plt.style.use(['../../plot_style_files/stacked.mplsty'])
    plt.style.use('seaborn-dark-palette')
    mpl.rcParams['hatch.linewidth'] = 0.25
    fig, ax = plt.subplots()
    for m in methods:
        df_aux = df_filtered[df_filtered['method'] == m]
        x_axis = np.array(df_aux['alpha'])
        y_axis = np.array(df_aux['nmse_test'])
        new_x, med, lerr, uerr = get_median_and_quantiles(x_axis, y_axis)
        plot_error_bar(1/new_x * offsets[m] , med, lerr, uerr, ax, None)
        ax.set_xscale('log')
        ax.set_ylabel('Test NMSE')
    ax.set_xlim((10**(-7), 10))
    ax.set_ylim((0.47, 1.1))
    min_l2_norm = df_filtered[df_filtered['method'] == 'minl2norm']['nmse_test']
    min_l1_norm = df_filtered[df_filtered['method'] == 'minl1norm']['nmse_test']
    plt.fill_between([10**(-7), 10], 2 * [np.quantile(min_l2_norm, 0.25)],  2 * [np.quantile(min_l2_norm, 0.75)], hatch=r'///', facecolor='w', alpha=0, label='min. $\ell_2$ sol.', rasterized=True)
    plt.fill_between([10 ** (-7), 10], 2 * [np.quantile(min_l1_norm, 0.25)], 2 * [np.quantile(min_l1_norm, 0.75)],
                     hatch='\\\\\\', facecolor='w', alpha=0, label=r'min. $\ell_1$ sol.', rasterized=True)
    plt.axhline(np.quantile(min_l2_norm, 0.25), lw=0.15, color='black', rasterized=True)
    plt.axhline(np.quantile(min_l2_norm, 0.75), lw=0.15, color='black', rasterized=True)
    plt.axhline(np.quantile(min_l1_norm, 0.25), lw=0.15, color='black', rasterized=True)
    plt.axhline(np.quantile(min_l1_norm, 0.75), lw=0.15, color='black', rasterized=True)
    plt.subplots_adjust(left=0.14)
    plt.xticks([1e-7, 1e-5,  1e-3, 1e-1,10])
    plt.xlim([1e-7, 20])
    legend = plt.legend(loc='upper right')
    mpl.rcParams['savefig.dpi'] = 300
    legend.set_rasterized(True)
    loc = plticker.LogLocator(base=10, numticks=10)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    locmin = plticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), numticks=1000)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(4))
    show('magic_test_vs_regul.pdf')


    # Plot train error
    #plt.style.use(['../../plot_style_files/stacked_bottom2.mplsty'])
    fig, ax = plt.subplots()
    for m in methods:
        df_aux = df_filtered[df_filtered['method'] == m]
        x_axis = np.array(df_aux['alpha'])
        y_axis = np.array(df_aux['nmse_train'])
        new_x, med, lerr, uerr = get_median_and_quantiles(x_axis, y_axis)
        plot_error_bar(1/new_x * offsets[m], med, lerr, uerr, ax, methods_pretty_names[m])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Train. NMSE')
        ax.set_xlabel('$$1/\delta$$')
    plt.subplots_adjust(left=0.14, bottom=0.4)
    plt.legend(bbox_to_anchor=(0.43, -0.75), loc='lower center', ncol=4)
    plt.xticks([1e-7, 1e-5,  1e-3, 1e-1,10])
    plt.xlim([1e-7, 20])
    loc = plticker.LogLocator(base=10, numticks=10)  # this locator puts ticks at regular intervals

    loc = plticker.LogLocator(base=10, numticks=10)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    locmin = plticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), numticks=1000)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    loc = plticker.LogLocator(base=10, numticks=6)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    locmin = plticker.LogLocator(base=10.0, subs=(1.0,), numticks=1000)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    show('magic_train_vs_regul.pdf')

    # Filter by alpha
    all_df = []
    for m in list(methods_extended):
        for n_features in np.unique(df['n_features']):
            if m in methods:
                df_filtered = df[df['n_features'] == n_features]
            else:
                df_filtered = df
            df_aux = df_filtered[df_filtered['method'] == m]
            x_axis = np.array(df_aux['alpha'])
            y_axis = np.array(df_aux['nmse_test'])
            new_x, med, lerr, uerr = get_median_and_quantiles(x_axis, y_axis)
            best_alpha = new_x[np.argmin(med)]
            all_df.append(df_aux[df_aux['alpha'] == best_alpha])
    df_concat = pd.concat(all_df)

    # Plot barplot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="method", hue="n_features", y="nmse_test",
               data=df_concat, ax=ax, palette="Set3")
    ax.set_ylim((0.4, 1.45))
    ax.set_ylabel('Test NMSE')
    ax.set_xlabel('')
    ax.set_xticklabels(methods_pretty_names.values(), rotation=55, ha='right')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_yticks([0.4,  0.6, 0.8, 1.0, 1.2, 1.4])
    plt.subplots_adjust(bottom=0.35, right=0.7)
    plt.legend(title="\# features, $p$", bbox_to_anchor=(0.98, 1), loc='upper left')
    show('magic_test_vs_size.pdf')