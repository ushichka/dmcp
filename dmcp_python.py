from julia import Main as jl
import scipy.io
import numpy as np

# demo data from thesis
mat = scipy.io.loadmat("demo/dmcp_inputs_demo.mat")

# call julia implementation
jl.include("src/dmcp_alg.jl")
A = jl.exec_dmcp(mat["K_th"].astype(np.float64), mat["P_th"].astype(np.float64), mat["I_dm"].astype(np.float32), mat["K_dm"].astype(np.float64), mat["P_dm"].astype(np.float64), mat["cps"].astype(np.float64))

print(A)