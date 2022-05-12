import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='execute dmcp on ushichka')
    parser.add_argument("--imK")
    parser.add_argument("--imP")
    parser.add_argument("--dmK")
    parser.add_argument("--dmP")
    parser.add_argument("--dmIm")
    parser.add_argument("--cps")
    parser.add_argument("--out")


    args = parser.parse_args()

    # read images and set origin to bottom left
    imK = np.loadtxt(args.imK, delimiter=",")
    imP = np.loadtxt(args.imP, delimiter=",")

    dmK = np.loadtxt(args.dmK, delimiter=",")
    dmP = np.loadtxt(args.dmP, delimiter=",")
    dmIm = np.loadtxt(args.dmIm, delimiter=",")[-1:0:-1,:]

    cps = np.loadtxt(args.cps, delimiter=",")
