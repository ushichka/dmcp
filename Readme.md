## Programming languages to install 
- python (Recommend using a virtual environment with <=3.9)
- julia (Tested with Julia 1.7.2)
  - ALL OS installation: **Remember to add Julia to your system's path**
  - Windows users may want to restart their systems just to ensure the Path is updated (do this also if Python later throws an error saying the ```julia``` module is not found)

## Setup Dependencies
 -  python: python -m ensurepip, python -m pip install X
    -  numpy
    -  scipy
    -  pyvista (depends on vtk, pip issue on python 3.10. 3.9 works)
    -  julia (allows cross-talk between python and julia)

## Run these commands after installing the required packages

### Python 
Open up Python in your virtual environment and run these two commands.
imageio.plugins.freeimage.download()
```
>>> import julia
>>> julia.install()
```

### Julia
* Open a command prompt/bash window
* Move to the ```dmcp``` directory
```
>>> ] activate .
>>> instantiate
```
* Press on Backspace to return to the Julia coding space and exit with ```exit()```

### Example

#### Step 1: Create a depth map 
We'll use the ```capture_depth``` module to create a depth map. The depth map is a 2D projection of 
a 3D object where pixels indicate depth from the camera.

* To start you need the volume's mesh as a .ply file
* Open a command prompt/bash window and move to the ```dmcp``` directory
* Type the following command
 ```bash
 python -m capture_depth --mesh <meshfile_path_here> --outIm <path_imagefilepathhere>.csv --outK <path_camera_intrinsics>.csv --outP <path_projectionmat>.csv
 ```
    * This will open up a PyVista 3D visualisation window.
	* **Make the interface FULLSCREEN (important as this is the image that will be save)**
	* Navigate using the interface till you have a view that broadly matches that of your experimental image (it's useful having the experimental image on another window for comparison).
    * Once you're happy with the match, you can close the window. An output depth-map image will be saved along with the camera intrinsics and projection matrix as csv files. 

#### Step 2: Annotating matching points between depth map and your camera image
* Open up/stay in the ```dmcp``` directory
* Run the annotation routine with the command below. This will open a window with the two images. The depth map is on top, and your image is at the bottom.
(Especially the first time you run this command it may take a few seconds to a minute to compile. This is normal)
```
julia --project=. annotate_points.jl --dm data/dmIm.csv  --im demo/imIm.csv --out data/cps.csv
```
* Click on at least 8 points in both views which correspond to each other. You need to do it in the correct order ('1' in the depth map, and then the same point in the camera image). You **cannot** correct a mouse click as of now. Re-type the command if you made a mistake. 
* When you're done, close the plot window. The output file will be in the folder.

**NOTE** images should be stored so that the origin is on the top left, with x-axis pointing to the right and y-axis pointing downwards (most programs default to this)

#### Step 3: Run the camera-mesh alignment
* Stay in the ```dmcp``` folder and run the alignment. Type the following into your command prompt/bash window
```
julia --project=. exec_dmcp.jl --imK demo/imK.csv --imP demo/imP.csv --dmK data/dmK.csv --dmP data/dmP.csv --dmIm data/dmIm.csv --cps data/cps.csv --out data/transform.csv
```
* The final output is a 'transform' csv file with a 4x4 matrix in it. Multiply this transform matrix to any 3D points obtained from camera 3d tracking to bring the camera points into the mesh coordinate system


## Usage
 1. python -m capture_depth --mesh [LIDARFILE] --out [OUTFILE]
 2. julia --project=. annotate_points.jl -dm [DM.csv] -im [IM.csv] (may take some time to start, especially first start)
 3. julia --project=. exec_dmcp.jl [OPTIONS]

  
## Interaction
Explained [here-FIXLINK](https://makie.juliaplots.org/v0.15.2/examples/layoutables/axis/)

BUGS:
* Last column removal in line 39 of ```annotate_points.jl``` leads to issues in plotting
* Julia csv sep file

TODO:
* Make all 3 steps follow naturally using a common Python script
* FeatureRequest: add hold s and click in ```annotate_points.jl```
* Implement correspondence annotation between the 3D mesh view and the experimental image (not 2D depth map and experimental image)
* Update the Julia plotting interface help link
* add requirements file 
* use last calibration round instead of 1st

## Ushichka-Interface
We included the python module ```pyushichka``` for easy experimentation with the ushichka dataset.
You can read calibration data (intrinsics _K_ and projection matrix _P_) for each camera _(0,1,2)_.

Example:
```python
from pyushichka import loadCalibration
cam = 1 # second camera
K, P = loadCalibration(cam, "data/ushichka/2018-08-18") # 2018-08-18 is the recording of a specific night

```
## Experiment Script
/home/julian/venvs/dmcp/bin/python /home/julian/code/uni/dmcp/exec_experiment.py --dir /tmp/experiments --mesh demo/lidar_roi.ply --recording ~/data/ushichka/2018-08-18 --cam 0 [--step 1]

### python annotation tool
```
python -m compute_reprojection_error --cps C:\data\experiments_demo\cps.csv --dm C:\data\experiments_demo\dmIm.csv --dmK C:\data\experiments_demo\dmK.csv --dmP C:\data\experiments_demo\dmP.csv  --im C:\data\experiments_demo\imIm.csv  --imP C:\data\experiments_demo\imP.csv --transform C:\data\experiments_demo\transform.csv --outErrs C:\data\experiments_demo\reprErrs.csv --outScatter C:\data\experiments_demo\reprScatter.png --outBar C:\data\experiments_demo\reprBar.png 
```