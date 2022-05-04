import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='show csv as heatmap')
parser.add_argument('--csv')
parser.add_argument('--sep', default=',')
args = parser.parse_args()

def isNumeric(s):
   s = s.strip()
   try:
      s = float(s)
      return True
   except:
      return False

arr = np.genfromtxt(args.csv,delimiter=args.sep, dtype='str')
#t = np.where(canBeNumber(arr),arr, 0)
arr.flat = [ float(e) if isNumeric(e) else np.nan for e in arr.flat]

arr = arr.astype(np.float32)

plt.imshow(arr)
plt.show()