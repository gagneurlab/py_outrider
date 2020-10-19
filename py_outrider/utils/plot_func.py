import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.colors import LogNorm
from py_outrider.utils.stats_func import get_prec_recall
from py_outrider.utils.float_limits import replace_zeroes_min
from py_outrider.utils.print_func import np_summary



def plot_hist(df, title=None, range = None, bins=30):
    plt.clf()
    df_plot = df.flatten()
    plt.hist(df_plot, range=range, bins=bins)
    plt.suptitle(title, y=1.00, fontsize=12)
    plt.title(np_summary(df_plot), fontsize=8)




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
        ax.plot(ax.get_xlim(), ax.get_ylim(), c="grey")  # diagonale
    else:
        diag=np.linspace(-500,500,51)
        plt.plot(diag,diag,'k-', c="grey")
        plt.xlim(np.nanmin(x),np.nanmax(x))
        plt.ylim(np.nanmin(y),np.nanmax(y))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)





def ppoints(n, a=0.5):
    """
    ppoints function from R
    ordinates for probability plotting
    :param n: number of observation points
    :param a: offset fraction to be used, typically in (0,1)
    :return: sequence of probability points
    """
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

    plot_binhex(points_log, pval_log, title=title,
                ylab="observed p-values [-log10]", xlab="expected p-values [-log10]",
                axis_limit=min(8, max(pval_log + 1)), center=False)



def plot_empty_text(txt):
    """
    creates an empty plot without axis and only text
    :param txt: shown text in plot
    """
    plt.clf()
    plt.plot([0, 1], [0, 1], alpha=0)
    plt.text(0.5, 0.5, txt, horizontalalignment='center', verticalalignment='center', fontsize=30)
    plt.axis('off')





def plot_hyperpar_fit(hyperpar_df):
    """
    plots the precision-recall performance during the hyperparameter optimisation
    :param hyperpar_df: dataframe of dictionary xrds.attrs["hyperpar_table"]
    """
    plt.clf()

    fig, ax = plt.subplots()
    for label, group in hyperpar_df.groupby('noise_factor'):
        ax.plot(group["encod_dim"], group["prec_rec"], label=label)

    ax.set_ylabel('precision-recall [AUC]')
    ax.set_xlabel('encoding dimension')
    ax.set_title('hyperparameter optimisation')
    ax.legend(title="noise factor")







