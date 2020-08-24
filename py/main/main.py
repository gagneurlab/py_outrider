import sys

## necessary for running script in shell -> TODO better fix !!
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)
# print(dir_path)

from pathlib import Path
import parser.input_parser as input_parser
import utilis.print_func as print_func
from full_run import Full_run



def main():
    args = input_parser.parse_args(sys.argv[1:])

    print(args)
    if args['file_meas'] is not None:    #
        Full_run(args_input=args)

    else:
        ### test run options for protrider | outrider on sample data
        test_dir = Path(__file__).parent.absolute().parents[1] / "tests"

        sample_args = { "file_meas" : test_dir / "sample_protein.csv", "encod_dim": 10,
                  'verbose':True, 'num_cpus':5, 'seed':5, "output_plots":True,
                  "max_iter": 5, "profile": "protrider",
                'file_sa': test_dir / 'sample_protein_sa.csv', 'covariates': ["is_male", "batch"],
                 "output": test_dir / "sample_prot_output"
                  }

        sample_args = { "file_meas" : test_dir / "sample_gene.csv", "encod_dim": 10,
                  'verbose':False, 'num_cpus':5, 'seed':5, "output_plots":True,
                  "max_iter": 5, "profile": "outrider",
                 "output": test_dir / "sample_gene_output"
                  }

        args.update(sample_args)
        print_func.print_dict(args)
        Full_run(args_input=args)






if __name__ == '__main__':
    main()








