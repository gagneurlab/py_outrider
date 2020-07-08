from pathlib import Path
import shutil
import os

class Check_parser():

    def __init__(self, args_input):

        ### modify arguments and transform to path obj
        self.args_mod = args_input.copy()
        self.args_mod['file_meas'] = self.check_file_meas(args_input['file_meas'])
        self.args_mod['file_sa'] = self.check_file_sa(args_input['file_sa'])
        self.args_mod['encod_dim'] = self.check_encod_dim(args_input['encod_dim'])
        self.args_mod['output'] = self.check_output(args_input['file_meas'], args_input['output'])
        self.args_mod['X_is_outlier'] = self.check_X_is_outlier(args_input['X_is_outlier'])

### TODO CHECK COV USED IS LIST




    def get_path(self, file_path, name):
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f'{name} does not exists: {file_path}')
        else:
            return path


    def check_file_meas(self, file_meas):
        if file_meas is None:
            raise ValueError("file_meas path not specified")
        else:
            return self.get_path(file_meas, "file_meas")


    def check_file_sa(self, file_sa):
        if file_sa is None:
            return None
        else:
            return self.get_path(file_sa, "file_sa")


    def check_encod_dim(self, encod_dim):
        # if encod_dim < 2:
        #     raise ValueError(f'encod_dim must be >1: {encod_dim}')
        # else:
        #     return encod_dim
        return encod_dim


    def check_output(self, file_meas, output):
        if output is None:
            output = Path(file_meas).parent
        else:
            output = Path(output)

        if output.exists() and not output.is_file():
            if not os.listdir(output):  # if empty
                return str(output.resolve())
            else:
                output = output / "outrider_result.zarr"
                if output.exists():
                    shutil.rmtree(output)
                return str(output.resolve())
        else:
            output.mkdir(parents=True, exist_ok=True)
            return str(output.resolve())



    def check_X_is_outlier(self, X_is_outlier):
        if X_is_outlier is None:
            return None
        else:
            return self.get_path(X_is_outlier, "X_is_outlier")





