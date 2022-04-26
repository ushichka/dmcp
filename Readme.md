## Usage
 - python dmcp_python.py LIDAR_PATH
 - julia --project=. annotate_points.jl (may take some time to start, especially first start)

## Setup Dependencies
 -  python: python -m ensurepip, python -m pip install X
    -  numpy
    -  scipy
    -  pyvista (depends on vtk, pip issue on python 3.10. 3.9 works)
    -  julia
       -  in python (once): import julia\n julia.install()
 - julia
   - in julia REPL: ]activate . and ]instantiate 
  
## Interaction
Explained [here](https://makie.juliaplots.org/v0.15.2/examples/layoutables/axis/)

## known issues
 - annotated points sometimes "jump", however they are stored correctly and will jump back to correct position eventually. It likely is an issue in the Makie / Observable library
