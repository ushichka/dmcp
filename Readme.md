## Usage
 1. python -m capture_depth --mesh [LIDARFILE] --out [OUTFILE]
 2. julia --project=. annotate_points.jl -dm [DM.csv] -im [IM.csv] (may take some time to start, especially first start)
 3. julia --project=. exec_dmcp.jl [OPTIONS]
### Example
```bash
python -m capture_depth --mesh demo/lidar_roi.ply --outIm data/dmIm.csv --outK data/dmK.csv --outP data/dmP.csv
julia --project=. annotate_points.jl --dm data/dmIm.csv  --im demo/imIm.csv --out data/cps.csv
julia --project=. exec_dmcp.jl --imK demo/imK.csv --imP demo/imP.csv --dmK data/dmK.csv --dmP data/dmP.csv --dmIm data/dmIm.csv --cps data/cps.csv --out data/transform.csv
```

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
