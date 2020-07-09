import pandas as pd
import numpy as np
import xarray as xr

from profiles.profile_outrider import Profile_outrider
from profiles.profile_protrider import Profile_protrider
from profiles.profile_pca import Profile_pca



class Create_xarray():

    def __init__(self, args_input):

        X_file = self.read_data_file(args_input["file_meas"])

        ### create dict to transform to xarray object
        xrds_dict = {
            "X":(("sample","meas"), X_file.values),
            "X_na":(("sample","meas"), np.isfinite(X_file).values )
        }

        xrds_coords = {
            "sample": X_file.index,
            "meas": X_file.columns
        }

        if args_input["file_sa"] is not None:
            sample_anno = self.get_sample_anno(args_input["file_sa"], X_file)
            xrds_dict["sample_anno"] = (("sample","sample_anno_col"), sample_anno.values.astype(str))
            xrds_coords["sample_anno_col"] = sample_anno.columns

            if args_input["cov_used"] is not None:
                cov_sample = self.get_covariates(sample_anno, args_input["cov_used"])
                xrds_dict["cov_sample"] = (("sample", "cov_used"), cov_sample.values)
                xrds_coords["cov_sample_col"] = cov_sample.columns

        if args_input["X_is_outlier"] is not None:
            X_is_outlier =  self.get_X_is_outlier(args_input["X_is_outlier"], X_file)
            X_is_outlier[~xrds_dict["X_na"][1]] = np.nan   # force same nan values as in X
            xrds_dict["X_is_outlier"] = (("sample", "meas"), X_is_outlier.values)

        self.xrds = xr.Dataset(xrds_dict, coords = xrds_coords)

        ### add additional metadata
        for add_attr in ['encod_dim','num_cpus','output','output_list', 'float_type',
                         'max_iter','verbose','seed','output_plots'] :
            self.xrds.attrs[add_attr] = args_input[add_attr]

        self.xrds["par_sample"] = (("sample"), np.repeat(1, len(self.xrds.coords["sample"])))
        # self.xrds.attrs['float_type'] = self.get_float_type(args_input['float_type'])
        self.xrds.attrs['profile'] = self.get_profile(args_input['profile'])

        ### preprocess xrds
        self.xrds.attrs['profile'].prepro.prepro_xrds(self.xrds)








    def read_data_file(self, file_path):
        if file_path is None:
            print(f'file_path specified is None')
            return None
        else:
            data_file = pd.read_csv(file_path, sep=',', header=0, index_col=0).fillna(np.nan)
            return data_file


    def get_sample_anno(self, sa_file_path, X_file):
        sample_anno = self.read_data_file(sa_file_path)   # TODO fix with no index col ?

        ### find sample_id column
        sample_col_found = None
        for col in sample_anno:
            if set(X_file.index).issubset(sample_anno[col]):
                sample_col_found = col

        if sample_col_found is None:
            raise ValueError("file_meas sample names not found in file_sa or not complete")
        elif len(sample_anno[sample_col_found]) != len(set(sample_anno[sample_col_found])):
            raise ValueError(f"duplicates found in file_sa sample_id column: {sample_col_found}")
        else:
            sample_anno.rename(columns = {sample_col_found:'sample_id'}, inplace = True)
            sample_anno.set_index(sample_anno["sample_id"], inplace=True)

        ### sort according to X_file and remove unnecessary
        sample_anno = sample_anno.reindex(X_file.index)
        return sample_anno


    def get_covariates(self, sample_anno, cov_used):

        # TODO HANDLE NAN CASES IN COVARIATES

        if not set(cov_used).issubset(sample_anno.columns):
            print("INFO: not all covariates could be found in file_sa")

        cov_used = [x for x in cov_used if x in sample_anno.columns]
        cov_sample = sample_anno[cov_used].copy()

        ### transform each cov column to the respective 0|1 code
        for c in cov_sample:
            col = cov_sample[c].astype('category')
            if len(col.cat.categories) == 1:
                cov_sample.drop(c, axis=1, inplace=True, errors='ignore')
            elif len(col.cat.categories) == 2:
                only_01 = [True if x in [0,1] else False for x in col.cat.categories]
                if all(only_01) is True:
                    # print(f"only_01: {c}")
                    pass
                else:
                    # print(f"2 cat: {c}")
                    oneh = pd.get_dummies(cov_sample[c])
                    cov_sample[c] = oneh.iloc[:,0]
            else:
                # print(f">2 cat: {c}")
                oneh = pd.get_dummies(cov_sample[c])
                oneh.columns = [c + "_" + str(x) for x in oneh.columns]
                cov_sample.drop(c, axis=1, inplace=True, errors='ignore')
                cov_sample = pd.concat([cov_sample, oneh], axis=1)
        return cov_sample


    def get_X_is_outlier(self, X_is_outlier_path, X_file):
        X_is_outlier_file = self.read_data_file(X_is_outlier_path)
        if X_file.shape != X_is_outlier_file.shape:
            raise ValueError(f"different shapes of X [{X_file.shape}] and X_is_outlier [{X_is_outlier_file.shape}]")
        if not all(X_is_outlier_file.index == X_file.index):
            raise ValueError(f"different rownames of X and X_is_outlier")
        if not all(X_is_outlier_file.columns == X_file.columns):
            raise ValueError(f"different columns of X and X_is_outlier")
        return X_is_outlier_file


    # def get_float_type(self, float_type):
    #     if float_type == 'float32':
    #         return np.float32
    #     elif float_type== 'float64':
    #         return np.float64
    #     else:
    #         print(f"INFO: float_type {float_type} not found, using float64")
    #         return np.float64


    def get_profile(self, profile):
        if profile.lower()=="outrider":
            return Profile_outrider()
        elif profile.lower()=="protrider":
            return Profile_protrider()
        elif profile.lower()=="pca":
            return Profile_pca()
        else:
            print('create own profile here out of parameter')








