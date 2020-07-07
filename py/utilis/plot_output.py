from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import utilis.plot_func as pl



class Plot_output():


    def __init__(self, xrds):
        if xrds.attrs["output_plots"] is True:
            self.plot_all(xrds)




    def plot_all(self, xrds):
        output_file = Path(xrds.attrs["output"]).parent / "plot_summary.pdf"
        print(f'writing output into {output_file}')
        pp = PdfPages(output_file)

        ### prediction
        pp.savefig(pl.plot_empty_text("prediction summary"))
        pp.savefig(pl.plot_binhex(xrds["X_trans_pred"].values, (xrds["X_trans"]+xrds["X_center_bias"]).values,
                   title="transformed prediction comparison", xlab="X_pred",ylab="X_true") )
        pp.savefig(pl.plot_binhex(xrds["X_trans_noise"].values, xrds["X_trans"].values,
                   title="noise insertion comparison", xlab="X_noise",ylab="X_true", center=True) )

        pp.savefig(pl.plot_qqplot(xrds["X_pvalue"].values, title="qq-plot global") )

        if "X_is_outlier" in xrds:
            pp.savefig(pl.plot_prec_recall([xrds["X_pvalue"].values], xrds["X_is_outlier"].values,
                                           names=["py_outrider"], title="precision-recall on artifically injected outliers"))


        ### model summary
        pp.savefig(pl.plot_empty_text("model weights"))
        pp.savefig(pl.plot_hist(xrds["encoder_weights"].values, title="encoder_weights") )
        pp.savefig(pl.plot_hist(xrds["decoder_weights"].values, title="decoder_weights") )
        pp.savefig(pl.plot_hist(xrds["decoder_bias"].values, title="decoder_bias") )

        pp.close()













