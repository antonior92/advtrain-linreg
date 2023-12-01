import pandas as pd
from utils import get_quantiles_df, plot_errorbar
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# Basic style
plt.style.use(['../mystyle.mplsty'])

# Additional style
mpl.rcParams['figure.figsize'] = 7, 3
mpl.rcParams['figure.subplot.bottom'] = 0.3
mpl.rcParams['figure.subplot.right'] = 0.99
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.01
mpl.rcParams['xtick.major.pad'] = 7

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot parameters profile')
    parser.add_argument('-i', '--input', default='./results/gaussian.csv',
                        help='output file.')
    parser.add_argument('-f', '--folder', default='',
                        help='output file.')
    args, unk = parser.parse_known_args()

    dset_name = os.path.split(args.input)[-1].split('.')[0]  # Get name to be used from input file

    def display(name):
        if args.folder:
            plt.savefig(os.path.join(args.folder, name + '.pdf'))
        else:
            plt.show()




    df = pd.read_csv(args.input)

    ######### Plot threshold vs prop #########
    mpl.rcParams['figure.subplot.left'] = 0.16
    fig, ax = plt.subplots()
    # l2 adv attack
    plot_errorbar(df[df['method'] == 'minl2norm'], 'prop', 'threshold', ax=ax, label=r'min. $\ell_2$-norm', color='blue')
    # Plot reference
    xv, yv, _, _ = get_quantiles_df(df[df['method'] == 'minlinfnorm'], 'prop', 'input_l2norm')
    plt.plot(xv, 0.01*yv, color='blue', ls=':', marker='.')
    # l1 adv attack
    plot_errorbar(df[df['method'] == 'minlinfnorm'], 'prop', 'threshold', ax=ax, label=r'min. $\ell_1$-norm',
                  color='green')
    # Plot reference
    xv, yv, _, _ = get_quantiles_df(df[df['method'] == 'minlinfnorm'], 'prop', 'input_linfnorm')
    plt.plot(xv, 0.01*yv, color='green', ls=':', marker='.')
    # Extra plot settings
    ax.set_xlabel('$p / n$')
    ax.set_ylabel(r'$\bar \delta$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    display(f'threshold_prop_{dset_name}')

    ######### Plot test error (l2 norm) #########
    mpl.rcParams['figure.subplot.left'] = 0.15
    fig, ax = plt.subplots()
    dfl2 = df[df['method'] == 'minl2norm']
    plot_errorbar(dfl2, 'prop', 'advrisk-2.0-0.0100000000', ax=ax, label='Adv. MSE', color='green')
    plot_errorbar(dfl2, 'prop', 'mse_test', ax=ax, label='Test MSE', color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('$p / n$')
    ax.set_ylabel(r'MSE')
    plt.legend()
    display(f'advriskl2_prop_{dset_name}')

    ######### Plot test error (linf norm) #########
    mpl.rcParams['figure.subplot.left'] = 0.15
    fig, ax = plt.subplots()
    dflinf = df[df['method'] == 'minlinfnorm']
    plot_errorbar(dflinf, 'prop', 'advrisk-inf-0.0100000000', ax=ax, label='Adv. MSE', color='green')
    plot_errorbar(dflinf, 'prop', 'mse_test', ax=ax, label='Test', color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('$p / n$')
    ax.set_ylabel(r'MSE')
    plt.legend()
    display(f'advrisklinf_prop_{dset_name}')