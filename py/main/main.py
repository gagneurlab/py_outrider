import sys

## necessary for running script in shell -> TODO better fix !!
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)
# print(dir_path)

from pathlib import Path
import parser.input_parser as input_parser
import utils.print_func as print_func
from full_run import Full_run



def main():
    args = input_parser.parse_args(sys.argv[1:])

    print(args)
    if args['file_meas'] is not None:    #
        Full_run(args_input=args)

    else:
        ### test run options for protrider | outrider on sample data
        test_dir = Path(__file__).parent.absolute().parents[1] / "tests"

        # sample_args = { "file_meas" : test_dir / "sample_protein.csv", "encod_dim": 10,
        #           'verbose':True, 'num_cpus':5, 'seed':5, "output_plots":True, "output_list":True,
        #           "max_iter": 3, "profile": "protrider",
        #           'file_sa': test_dir / 'sample_protein_sa.csv', 'covariates': ["is_male", "batch"],
        #          "output": test_dir / "sample_protein_output"
        #           }

        sample_args = { "file_meas" : test_dir / "sample_gene.csv", "encod_dim": 10,
                  'verbose':False, 'num_cpus':5, 'seed':5, "output_plots":True, "output_list":True,
                  "max_iter": 3, "profile": "outrider",
                  # 'file_sa': test_dir / 'sample_gene_sa.csv', 'covariates': ["is_male", "batch"],
                 "output": test_dir / "sample_gene_output"
                  }


        # folder_path = '/home/stefan/gagneurlab/home/Documents/mofa_outrider/01_example_blood_subset/'
        # folder_path = '/home/stefan/gagneurlab/home/Documents/mofa_outrider/06_sample_blood_outlier_z3/'
        #folder_path = '/home/stefan/Desktop/py_outrider/gene_subset_01/'
        # folder_path = '/home/stefan/Desktop/py_outrider/protein_neutropenia/'
        #
        #
        # # args2 = { "file_meas" : folder_path+"counts_raw.csv", "encod_dim": 3, 'verbose':True, 'num_cpus':5,
        # args2 = { "file_meas" : folder_path+"counts_raw.csv", "encod_dim": 3, 'verbose':True, 'num_cpus':5, 'seed':5,
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 2, "profile": "outrider"
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 1, "profile": "pca"
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 2, "profile": "protrider"
        #           "max_iter": 1, "profile": "protrider"
        #         # ,'file_sa': folder_path+ 'sa_file_artificially.csv', 'covariates': ["batch", "oneh"]
        #         ,'file_sa': folder_path+ 'san_notechrep.csv', 'covariates': ["sex", "date_processed_y_m_d"]
        #           }

        # folder_path = '/home/stefan/gagneurlab/s/project/protrider/loipf/data/prok_version_P20200317_with_outliers/'
        # args2 = { "file_meas" : folder_path+"X_raw_out.csv", "encod_dim": 25, 'verbose':True, 'num_cpus':5,
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 2, "profile": "outrider"
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 1, "profile": "pca"
        #           'X_is_outlier': folder_path+"X_out_pos.csv", "max_iter": 10, "profile": "protrider"
        #         ,'file_sa': folder_path+ 'prok_batches.csv', 'covariates': ["PROTEOMICS_BATCH","gender","INSTRUMENT"]
        #           }


        # folder_path = '/home/stefan/gagneurlab/s/project/protrider/loipf/results/20200713_min_sample_size/datasets/'
        # args2 = { "file_meas" : folder_path+"tmt_0_1.csv", "encod_dim": 5, 'verbose':True, 'num_cpus':5,
        #           "max_iter": 20, "profile": "protrider" #"protrider_cov1" #"protrider"
        #         ,'file_sa': '/home/stefan/gagneurlab/s/project/protrider/loipf/data/prok_version_P20200317_paper_samples/py_outrider/prok_batches.csv',
        #           'covariates': ["PROTEOMICS_BATCH","gender","INSTRUMENT"], "seed":5,
        #           'X_is_outlier': '/home/stefan/Desktop/X_is_outlier.csv', "float_type":"float32",
        #           "output":'/home/stefan/gagneurlab/s/project/protrider/loipf/results/20200713_min_sample_size/trained_obj/protrider/tmt_0_1/xrds_output_pytest2/'
        #           }


        args.update(sample_args)
        print_func.print_dict(args)
        Full_run(args_input=args)






if __name__ == '__main__':
    main()








