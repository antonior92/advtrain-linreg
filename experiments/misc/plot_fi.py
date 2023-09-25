import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

plt.style.use(['../mystyle.mplsty'])

# Additional style
mpl.rcParams['figure.figsize'] = 7.33, 3.45
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.top'] = 0.95
mpl.rcParams['figure.subplot.bottom'] = 0.2
mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handletextpad'] = 0.01
mpl.rcParams['xtick.major.pad'] = 7

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot parameters profile')
    parser.add_argument('--save', default='',
                        help='save plot in the given folder (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()

    beta = np.linspace(-2, 2, 500)
    x = 1
    y = 1

    for delta in [0.5, 1, 2, 4]:
        fi = lambda beta_: np.abs(x * beta_ - y) + delta * np.abs(beta_)
        plt.plot(beta, fi(beta))
        plt.text(2+0.02, fi(2)-0.5, f"$\delta={delta}$")
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\mathcal{L}(\alpha \boldsymbol{x})$')
    #plt.text(1.03, 10, r"$\alpha = \frac{|y_i|}{\|x_i\|_2}$")
    plt.axvline(1, color='black')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, None, 2, None, 4, None, 6, None, 8, None, 10])
    plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], [-2, None, -1, None, 0, None, 1, None, 2])

    plt.grid(ls=':')
    if args.save:
        plt.savefig(os.path.join(args.save, 'fi_plot.pdf'))
    else:
        plt.show()


