'''
This file refers the mlp course

Author: Yan Gao
email: gaoy4477@gmail.com
'''

import argparse

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True 
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False 
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(description='Feature extraction')

    parser.add_argument('--subsampling_rate', nargs="?", type=float, default=0.1, 
                        help='Subsampling rate for each slice')
    parser.add_argument('--begin_time', nargs="?", type=int, default=5,
                        help='Sample begin from this timestamp')
    parser.add_argument('--end_time', nargs="?", type=int, default=5,
                        help='Sample end at this timestamp')
    parser.add_argument('--begin_slice', nargs="?", type=int, default=600, 
                        help='Sample begin from this timestamp')
    parser.add_argument('--end_slice', nargs="?", type=int, default=699, 
                        help='Sample end at this timestamp')
    parser.add_argument('--file_name_3D', nargs="?", type=str, default="training_data_3D",
                        help='File name of saved 3D feature')
    parser.add_argument('--file_name_4D', nargs="?", type=str, default="training_data_4D",
                        help='File name of saved 4D feature')
    parser.add_argument('--size', nargs="?", type=int, default=3,
                        help='Type of different size of the area')

    args = parser.parse_args()
    print(args)
    return args
