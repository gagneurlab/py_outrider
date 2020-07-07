import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from utilis.stats_func import get_prec_recall
from matplotlib.colors import LogNorm
from utilis.float_limits import replace_zeroes_min


def plot_hist(df, title=None, range = None, bins=30):
    plt.clf()
    plt.hist(df.flatten(), range=range, bins=bins)
    plt.title(title)




def plot_prec_recall(pvalues, outlier_pos, names, title=""):
    plt.clf()
    if len(pvalues) != len(names):
        raise ValueError("len(pvalues) != len(names)")
    if isinstance(pvalues, list) is False:
        pvalues = [pvalues]
        names = [names]
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

    plt.figure()
    for idx, ele in enumerate(pvalues):
        pr = get_prec_recall(pvalues[idx], outlier_pos)
        plt.plot(pr['rec'], pr['pre'], color=colors[idx], lw=2)
        names[idx] = names[idx]+" ["+str(round(pr["auc"],5))+"]"

    plt.legend(names, loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.title(title)





def plot_binhex(x, y, title="", ylab="", xlab="", axis_limit=None, center=False):
    plt.clf()

    x = x[~np.isnan(x)]  # automatically flattens
    y = y[~np.isnan(y)]

    if axis_limit is None:
        axis_limit = max(max(map(abs, x)), max(map(abs, y)))  ## make easier
    else:
        axis_limit = abs(axis_limit)
        x = np.clip(x, -axis_limit, axis_limit)
        y = np.clip(y, -axis_limit, axis_limit)

    fig, ax = plt.subplots()
    h = ax.hist2d(x, y, 100, norm=LogNorm(), cmap='summer')

    fig.colorbar(h[3], ax=ax)

    if center:
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xlim(-axis_limit, axis_limit)
    else:
        ax.set_ylim(0, axis_limit)
        ax.set_xlim(0, axis_limit)

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, c="grey")  # diagonale
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)




def ppoints(n, a=0.5):
    try:
        n = np.float(len(n))
    except TypeError:
        n = np.float(n)
    return (np.arange(n) + 1 - a)/(n + 1 - 2*a)



def plot_qqplot(pval, title="qq-plot"):
    pval = replace_zeroes_min(pval.flatten())
    pval = pval[~np.isnan(pval)]
    points_log = -np.log10(sorted(ppoints(len(pval)), reverse=True))

    pval_log = np.array(sorted(pval, reverse=True))
    pval_log = -np.log10(pval_log)

    print(np.max(pval_log))

    plot_binhex(points_log, pval_log, title=title,
                ylab="observed p-values [-log10]", xlab="expected p-values [-log10]",
                axis_limit=min(8, max(pval_log + 1)), center=False)



# create empty plot with text only
def plot_empty_text(txt):
    plt.clf()
    plt.plot([0, 1], [0, 1], alpha=0)
    plt.text(0.5, 0.5, txt, horizontalalignment='center', verticalalignment='center', fontsize=30)
    plt.axis('off')











