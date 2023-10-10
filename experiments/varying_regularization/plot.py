import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as plticker
from os import listdir
from os.path import isfile, join

from experiments.utils import get_quantiles

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate on MAGIC dataset for grid of points.')
    parser.add_argument('input_folder', nargs='+',
                        help='input csv files.')
    parser.add_argument('--fields', default=['mse_train'], nargs='+',
                        help='which csv field to plot.')
    parser.add_argument('--filter_fields', default=['mse_train'], nargs='+',
                        help='which csv field to plot.')
    parser.add_argument('--labels', default=[], nargs='+',
                        help='labels. Should have the same size as field')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--log_yscale', action='store_true',
                        help='which csv field to plot.')
    parser.add_argument('--plot_rmin', action='store_true',
                        help='plot_rmin.')
    parser.add_argument('--plot_xmin', action='store_true',
                        help='plot_rmin.')
    parser.add_argument('--plot_a2max', action='store_true',
                        help='plot_rmin.')
    parser.add_argument('--plot_ainfmax', action='store_true',
                        help='plot_rmin.')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--x_min', default=None, type=float,
                        help='inferior limit to x-axis in the plot.')
    parser.add_argument('--x_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--ylabel', default='',
                        help='ylabel')
    parser.add_argument('--only_median', action='store_true',
                        help='plot only median')
    parser.add_argument('--no_ytickformater', action='store_true',
                        help='no tick formater')
    parser.add_argument('--loc', default='upper right',
                        help='loc')
    args = parser.parse_args()
    print(args)

    def show():
        if args.save:
            plt.savefig(args.save)
        else:
            plt.show()

    if args.plot_style:
        plt.style.use(args.plot_style)

    a = []
    for folder in args.input_folder:
        a += [pd.read_csv(join(folder, f)) for f in listdir(folder) if isfile(join(folder, f))]
    df_all = pd.concat(a, ignore_index=False)
    methods = np.unique(df_all['method'])

    i = 0
    fig, ax = plt.subplots()
    for mm in methods:
        df = df_all[df_all['method'] == mm]
        for field in args.fields:
            xaxis = np.array(df['alpha'])
            yaxis = np.array(df[field])
            new_xaxis, m, lerr, uerr = get_quantiles(xaxis, yaxis)

            try:
                lbl = args.labels[i]
            except:
                lbl = mm
            if args.only_median:
                ax.errorbar(x=1 / new_xaxis, y=m, markersize=3.5, ls='-', label=lbl)
            else:
                ax.errorbar(x=1/new_xaxis, y=m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                        marker='o', markersize=3.5,  ls='', label=lbl)
            ax.set_xscale('log')
            if args.log_yscale:
                ax.set_yscale('log')
            plt.xlabel(r"$1/\delta$")
            plt.ylabel(args.ylabel)
            if args.plot_a2max:
                plt.axvline(np.mean(1/df['max_a2']), color='black', lw=2)
            if args.plot_ainfmax:
                plt.axvline(np.mean(1/df['max_ainf']), color='black', lw=2)
            if args.y_max is not None and args.y_min is not None:
                ax.set_ylim((args.y_min, args.y_max))
            if args.x_max is not None and args.x_min is not None:
                ax.set_xlim((args.x_min, args.x_max))
            i += 1
    plt.legend(loc=args.loc,fontsize='small')

    loc = plticker.LogLocator(base=10, numticks=10)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    locmin = plticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    if not args.no_ytickformater:
        loc = plticker.LogLocator(base=10, numticks=6)  # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc)
        locmin = plticker.LogLocator(base=10.0, subs=(1.0,), numticks=1000)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    show()
