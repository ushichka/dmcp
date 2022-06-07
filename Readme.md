# DMCP Repository

## Programming languages to install 
- python (Recommend using a virtual environment with <=3.9)

## Setup Dependencies
 -  python: python -m ensurepip, python -m pip install -r requirements.txt
for the annotation script to work a sensible matplotlib backend should additionally be installed. TKAgg is recommended for linux, QT5Agg for windows computers.

The python code should then be installed as module ```src```.

For imageio to read pfm files, the respective plugin might be needed to be downloaded separately.

## Directory Structure
 - demo: contains example data from ushichka dataset.
 - notebooks: contains visualizations for ushichka and middlebury experiment done in the paper. Also some scripts that may help debugging.
 - **src**: contains main code
   - **dmcpworkflow**: contains code to quickly execute experiments
   - pycv: contains computer vision related methods, including _dmcp_
   - pyushichka: contains code used to read the ushichka dataset
   - **experiment**: contains Experiment class that allows for quick experiment execution and visualization
 - tools: contains standalone python cli tools e.g. for visualizing csv single-channel

## Experiments
### Executing a DMCP Experiment

```bash
export MESH=[MESHPATH] # alternative to providing mesh as cli-option
python src/experiment.py [path to experiment] --exec # view result using --repr and --pose
```
Example using data from _demo_ directory:
Download lidar_roi.ply from `10.5281/zenodo.6620671`.
```bash
python src/experiment.py demo --mesh lidar_roi.ply --exec # generate depth map, annotate points, compute SCA
python src/experiment.py demo --mesh lidar_roi.ply --repr # compute and show reprojection errors
python src/experiment.py demo --mesh lidar_roi.ply --pose # show estimated pose in lidar scene
```

Assumes csv files have comma delimiter.
Writes result into the experiment directory.
transform.csv contains the transformation to transform points from _native_ to _world_ space.

### Experiments from the paper
The ushichka experiment can be performed using `python src/dmcpworkflow/middlebury_experiment.py`. The middlebury experiment can be performed using `python src/dmcpworkflow/middlebury_experiment.py`. Usage is shown by appending the `--help` flag to the previous commands.

The python scripts load data from the ushichka / middlebury dataset and format it to fit the requirements for the experiment script to run as described above. The execution of the experiment.py script directly is not necessary.

## Ushichka-Interface
We included the python module ```pyushichka``` for easy experimentation with the ushichka dataset.
You can read calibration data (intrinsics _K_ and projection matrix _P_) for each camera _(0,1,2)_.

Example:
```python
from pyushichka import loadCalibration
cam = 1 # second camera
K, P = loadCalibration(cam, "data/ushichka/2018-08-18") # 2018-08-18 is the recording of a specific night
```
