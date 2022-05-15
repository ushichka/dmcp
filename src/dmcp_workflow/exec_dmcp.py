import argparse
import numpy as np
from src.pycv.dmcp import dmcp, dm_to_world

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='execute dmcp on ushichka')
    parser.add_argument("--imK")
    parser.add_argument("--imP")
    parser.add_argument("--cps")
    parser.add_argument("--out")


    args = parser.parse_args()

    # read images and set origin to bottom left
    imK = np.loadtxt(args.imK, delimiter=",")
    imP = np.loadtxt(args.imP, delimiter=",")

    cps = np.loadtxt(args.cps, delimiter=",")

    A = dmcp(imK, imP, cps[:,:2], cps[:,:2:])

    np.savetxt(args.out, A, delimiter=',') 
    print(f"transformation saved to {args.out}")

