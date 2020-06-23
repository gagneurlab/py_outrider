import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from utilis.stats_func import get_prec_recall



def plot_hist(df, main=None, output_file=None, bins=20):
    plt.hist(df.flatten(), range=(0, 1), bins=bins)
    plt.title(main)
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
    plt.clf()


def plot_prec_recall(pvalues, outlier_pos, names, output_file=None, title=""):
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

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


