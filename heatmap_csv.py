import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='show csv as heatmap')
parser.add_argument('--csv')
parser.add_argument('--sep', default=',')
args = parser.parse_args()

arr = np.loadtxt(args.csv,
                 delimiter=args.sep, dtype=str).astype("float32")
plt.imshow(arr)
plt.show()