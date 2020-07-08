import sys

## necessary for running script in shell -> TODO better fix !!
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)
# print(dir_path)

import parser.input_parser as input_parser
import utilis.print_func as print_func
from full_run import Full_run



def main():
    # print(sys.argv)
    args = input_parser.parse_args(sys.argv[1:])

    print(args)
    if args['file_meas'] is not None:    # if new tables are given, otherwise execute example
        Full_run(args_input=args)
        # print_func.print_dict(args)
        pass

    else:
        # folder_path = '/home/stefan/gagneurlab/home/Documents/mofa_outrider/01_example_blood_subset/'
        # folder_path = '/home/stefan/gagneurlab/home/Documents/mofa_outrider/06_sample_blood_outlier_z3/'
        folder_path = '/home/stefan/Desktop/py_outrider/gene_subset_02/'


        # args2 = { "file_meas" : folder_path+"counts_raw.csv", "encod_dim": 3, 'verbose':True, 'num_cpus':6,
        args2 = { "file_meas" : folder_path+"counts_raw.csv", "encod_dim": None, 'verbose':True, 'num_cpus':6,
                  # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 2, "profile": "outrider"
                  # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 1, "profile": "pca"
                  'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 2, "profile": "protrider"
                ,'file_sa': folder_path+ 'sa_file_artificially.csv', 'cov_used': ["batch", "oneh"]
                  }

        # folder_path = '/home/stefan/gagneurlab/s/project/protrider/loipf/data/prok_version_P20200317_with_outliers/'
        # args2 = { "file_meas" : folder_path+"X_raw_out.csv", "encod_dim": 25, 'verbose':True, 'num_cpus':6,
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 2, "profile": "outrider"
        #           # 'X_is_outlier': folder_path+"trueCorruptions.csv", "max_iter": 1, "profile": "pca"
        #           'X_is_outlier': folder_path+"X_out_pos.csv", "max_iter": 10, "profile": "protrider"
        #         ,'file_sa': folder_path+ 'prok_batches.csv', 'cov_used': ["PROTEOMICS_BATCH","gender","INSTRUMENT"]
        #           }


        # folder_path = '/home/stefan/gagneurlab/s/project/protrider/loipf/py_outrider_tests/'
        # args2 = { "file_meas" : folder_path+"X_raw.csv", "encod_dim": 28, 'verbose':True, 'num_cpus':6,
        #           'X_is_outlier': folder_path+"X_is_outlier.csv", "max_iter": 15, "profile": "pca" #"protrider_cov1" #"protrider"
        #         ,'file_sa': folder_path+ 'sample_anno.csv', 'cov_used': ["sex", "date_processed_y_m_d"]
        #           }




        args.update(args2)
        Full_run(args_input=args)

        # print_func.print_dict(args)


        # folder_path = '/home/stefan/gagneurlab/home/Documents/mofa_outrider/04_sample_blood_outlier_2/'
        # folder_path = '/home/stefan/gagneurlab/home/Documents/mofa_outrider/06_sample_blood_outlier_z3/'
        # # folder_path = '/home/stefan/gagneurlab/home/Documents/jupyter_protrider/outrider_sample/'
        #
        # args = {'filepath': folder_path + 'counts_raw.csv', 'dataset': 'genes_neg_bin',
        # # args = {'filepath': folder_path + 'counts_vst.csv', 'dataset': 'genes_gaus',
        #           'output': folder_path, 'q': 25, 'num_cpus': 6, 'max_iter': 0, 'float_type': 'float64', 'verbose': True}
        # args.update(args)



        # Prec_rec_test(ae_class=Ae_pca, ds_class=get_dataset_class(args['dataset']), file_path=args['filepath'], encoding_dim=args['q'], output_path=args['output'],
        #                      num_cpus=args['num_cpus'], max_iter=0, float_type=args['float_type'], verbose=args['verbose'])
        # Prec_rec_test(ae_class=Ae_adam, ds_class=get_dataset_class(args['dataset']), file_path=args['filepath'], encoding_dim=args['q'], output_path=args['output'],
        #                      num_cpus=args['num_cpus'], max_iter=300, float_type=args['float_type'], verbose=args['verbose'])
        # Prec_rec_test(ae_class=Ae_bfgs, ds_class=get_dataset_class(args['dataset']), file_path=args['filepath'], encoding_dim=args['q'], output_path=args['output'],
        #                      num_cpus=args['num_cpus'], max_iter=0, float_type=args['float_type'], verbose=args['verbose'])

        # Comparison_outrider(ae_class=Ae_bfgs, ds_class=get_dataset_class(args['dataset']), file_path=args['filepath'], encoding_dim=args['q'], output_path=args['output'],
        #                      num_cpus=args['num_cpus'], max_iter=args['max_iter'], float_type=args['float_type'], verbose=args['verbose'])

        # Full_run(ae_class=Ae_bfgs, ds_class=get_dataset_class(args['dataset']), file_path=args['filepath'], encoding_dim=args['q'], output_path=args['output'],
        # # Full_run(ae_class=Ae_bfgs, ds_class=get_dataset_class(args['dataset']), file_path=args['filepath'], encoding_dim=args['q'], output_path=args['output'],
        #                  num_cpus=args['num_cpus'], max_iter=args['max_iter'], float_type=args['float_type'], verbose=args['verbose'] )








if __name__ == '__main__':
    main()








