from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from ..utils import plot_func as pl
from ..utils.xarray_output import xrds_to_list


class Custom_output():

    def __init__(self, xrds):

        output_path = Path(xrds.attrs["output"]).parent
        print(f'writing custom output into {output_path}')

        if xrds.attrs["output_list"] is True:
            xrds_to_list(xrds, output_file = output_path/ "outrider_result_list.csv" )

        if xrds.attrs["output_plots"] is True:
            self.plot_all(xrds, output_file = output_path/ "plot_summary.pdf")




    def plot_all(self, xrds, output_file):

        pp = PdfPages(output_file)

        ### prediction
        pp.savefig(pl.plot_empty_text("prediction summary"))
        pp.savefig(pl.plot_binhex(xrds["X_trans_pred"].values, (xrds["X_trans"]+xrds["X_center_bias"]).values,
                   title="transformed prediction comparison", xlab="X_pred",ylab="X_true") )
        pp.savefig(pl.plot_binhex(xrds["X_trans_noise"].values, xrds["X_trans"].values,
                   title="noise insertion comparison", xlab="X_noise",ylab="X_true", center=True) )
        pp.savefig(pl.plot_qqplot(xrds["X_pvalue"].values, title="p-value qq-plot global") )
        pp.savefig(pl.plot_hist(xrds["X_pvalue"].values, range=(0,1), title="p-value distribution") )

        if "X_is_outlier" in xrds:
            pp.savefig(pl.plot_prec_recall([xrds["X_pvalue"].values], xrds["X_is_outlier"].values,
                                           names=["py_outrider"], title="precision-recall on artifically injected outliers"))

        ### model summary
        pp.savefig(pl.plot_empty_text("model weights"))
        pp.savefig(pl.plot_hist(xrds["encoder_weights"].values, title="encoder_weights") )
        pp.savefig(pl.plot_hist(xrds["decoder_weights"].values, title="decoder_weights") )
        pp.savefig(pl.plot_hist(xrds["decoder_bias"].values, title="decoder_bias") )

        ### hyperparameter optimisation
        pp.savefig(pl.plot_empty_text("other plots"))
        if "hyperpar_table" in xrds.attrs:
            hyperpar_df = pd.DataFrame.from_dict(xrds.attrs["hyperpar_table"])
            pp.savefig(pl.plot_hyperpar_fit(hyperpar_df))

        pp.close()













