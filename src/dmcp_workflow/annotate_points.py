import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import colorcet as cc

def annotate(image_im, image_dm):
    """origin bottom left """
    def draw(ax: plt.Axes, im: np.ndarray, points):
        #print(im)
        with plt.ion():
            ax.imshow(im, origin="lower", cmap=cc.cm.get("gouldian_r"))
            if len(points) != 0:
                points = np.array(points)
                ax.scatter(points[:,0], points[:,1],c="cyan", marker="x")
                for i in range(points.shape[0]):
                    ax.annotate(str(i+1),points[i,:],bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1),xytext=(points[i,0]+15, points[i,1]+15))
            ax.figure.canvas.draw()

    plt.figure()
    im_ax = plt.gca()
    points_im = []
    draw(im_ax, image_im, points_im)

    plt.figure()
    dm_ax = plt.gca()
    points_dm = []
    draw(dm_ax, image_dm, points_dm)

    # interactivity



    def on_click(event, ax, im, points):
        #print("onclick")
        if event.button is MouseButton.LEFT:
            #print("  left button")
            if event.inaxes:
                x, y = (event.xdata, event.ydata)
                #print("  in axes")
                #print('  data coords %f %f' % (x, y))
                imval = im[round(y),round(x)]
                #print(imval)
                if not np.isnan(imval):
                    points.append([x, y])
                    draw(ax, im,points)
                else:
                    print("cannot select, value is nan")
                

    im_ax.figure.canvas.mpl_connect('button_press_event', lambda event: on_click(event, im_ax, image_im, points_im))
    dm_ax.figure.canvas.mpl_connect('button_press_event', lambda event: on_click(event, dm_ax, image_dm, points_dm))

    plt.show(block=True)


    # create cps from annotated points
    if len(points_im) != len(points_dm):
        print("you must annotate the same amount of points in each image", file=sys.stderr)
        exit(1)
    cps = np.hstack((np.array(points_im), np.array(points_dm)))
    return cps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='execute dmcp on ushichka')
    parser.add_argument("--im")
    parser.add_argument("--dm")
    parser.add_argument("--out")


    args = parser.parse_args()

    path_im = args.im
    path_dm = args.dm
    path_cps = args.out

    # read images and set origin to bottom left
    image_im = np.loadtxt(path_im, delimiter=",")[-1:0:-1,:]
    image_dm = np.loadtxt(path_dm, delimiter=",")[-1:0:-1,:]

    cps = annotate(image_im, image_dm)

    np.savetxt(path_cps,cps,delimiter=",", fmt="%06.1f")
    print(f"annotations saved to {path_cps}")