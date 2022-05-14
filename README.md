## Setup
To start working on this repo, cd in to it and exec ``` pip install -e . ```
Make sure to use **python 3.9**. Python 3.10 seems to have problems with vtk required by pyvista at the moment.

For ipython use 
```
%load_ext autoreload
%autoreload 2
```
at the beginning to automatically reload changes in src without reinstalling.

## annotation
for notebook make sure to use interactive backend such as TKAgg using ```matplotlib.use("TKAgg")```