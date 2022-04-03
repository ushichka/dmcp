## Setup Dependencies
 -  python
    -  numpy
    -  scipy
    -  pyvista
    -  julia
       -  in python (once): import julia\n julia.install()
 - julia
   - in julia REPL: ]instantiate 
  
## Interaction
Explained [here](https://makie.juliaplots.org/v0.15.2/examples/layoutables/axis/)

## known issues
 - annotated points sometimes "jump", however they are stored correctly and will jump back to correct position eventually. It likely is an issue in the Makie / Observable library