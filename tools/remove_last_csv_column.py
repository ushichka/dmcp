import argparse
import numpy as np

parser = argparse.ArgumentParser(description='show csv as heatmap')
parser.add_argument('--csv')
parser.add_argument('--sep', default=';')
parser.add_argument('--out', default='last_col_removed.csv')
parser.add_argument('--type', default="int")
args = parser.parse_args()

arr = np.genfromtxt(args.csv,delimiter=args.sep, dtype='str')

arr_removed = arr[:,0:-1].astype(args.type)

np.savetxt(args.out, arr_removed, delimiter=args.sep, fmt='%i')