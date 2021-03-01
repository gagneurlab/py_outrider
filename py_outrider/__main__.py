import sys
from pathlib import Path

from . import parser
from .main_run import full_run


def main():
    args = parser.parse_args(sys.argv[1:])

    if args['input'] is not None:
        full_run(args_input=args)

    else:
        # test run options for protrider | outrider on sample data
        test_dir = Path(__file__).parent.absolute().parents[0] / "tests"

        # OUTRIDER test run (RNA-seq gene counts):
        sample_args = {"input": test_dir / "sample_gene.csv",
                       "encod_dim": 10, 'verbose': False, 'num_cpus': 5,
                       'seed': 5, "output_plots": True, "output_list": True,
                       "max_iter": 3, "profile": "outrider",
                       # 'sample_anno': test_dir / 'sample_gene_sa.csv',
                       # 'covariates': ["is_male", "batch"],
                       "output": test_dir / "sample_gene_output"}

        # PROTRIDER test run (protein MS intensities):
        # sample_args = {"input" : test_dir / "sample_protein.csv",
        #                "encod_dim": 10, 'verbose': True, 'num_cpus': 5,
        #                'seed': 5, "output_plots": True, "output_list": True,
        #                "max_iter": 3, "profile": "protrider",
        #                'sample_anno': test_dir / 'sample_protein_sa.csv',
        #                'covariates': ["is_male", "batch"],
        #                "output": test_dir / "sample_protein_output"}

        args.update(sample_args)
        full_run(args_input=args)
