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
    args = input_parser.parse_args(sys.argv[1:])

    print_func.print_dict(args)
    Full_run(args_input=args)






if __name__ == '__main__':
    main()








