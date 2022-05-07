import argparse
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
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

#from  scipy import ndimage as si
#from skimage import util
#from skimage.restoration import estimate_sigma, denoise_tv_chambolle, denoise_wavelet
#def inverseContrastImage(I):
#   sigma = estimate_sigma(I, channel_axis=-1, average_sigmas=True)
#   I = denoise_tv_chambolle(I, weight=500,channel_axis=-1)
#   iCI = util.invert(si.sobel(I))/I
#   return iCI

arr = np.genfromtxt(args.csv,delimiter=args.sep, dtype='str')
#t = np.where(canBeNumber(arr),arr, 0)
arr.flat = [ float(e) if isNumeric(e) else np.nan for e in arr.flat]

arr = arr.astype(np.float32)

plt.imshow(arr,cmap=cc.cm.gouldian)
plt.show()