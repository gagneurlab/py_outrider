
from defaults import check_parser




class Full_run():

    def __init__(self, args_input):

        args_mod = check_parser.Check_parser(args_input).args_mod
        # xrds = create dataset (args_mod

        # check parser
        # create data set - insert noise, outlier, inject
        # run model
        # calculate statistics
        # plot



        None


    def create_dataset(self, args_input):
        None











# import os
#
# from utilis_methods import data_handling as dh
#
#
# ### performs outrider python run on folder with counts.csv table from OUTRIDER
#
# class Full_run():
#
#     def __init__(self, ae_class, ds_class, file_path, encoding_dim, output_path, num_cpus, max_iter, float_type,
#                  verbose):
#         self.output_path = dh.path(output_path, 'results_xarr')
#         os.makedirs(self.output_path, exist_ok=True)
#         print(f'results directory = {self.output_path}')
#
#         if not os.listdir(self.output_path):
#             self.exp = ds_class(data_file_path=file_path, encoding_dim=encoding_dim,
#                                 num_cpus=num_cpus, float_type=float_type, verbose=verbose)
#             self.ae = ae_class(self.exp)
#             self.ae.run_autoencoder(max_iter=max_iter)
#             xrds = dh.get_xarr_obj(self.ae)
#             xrds.to_zarr(self.output_path)
#         else:
#             print('ERROR: SELECT AN EMPTY FOLDER OTHERWISE OUTPUT XARRAY CAN NOT BE SAFED')
#             ### TODO fix otherwise
#











