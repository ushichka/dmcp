#%%

import julia

def dmcp_alg(Kim, Pim, Idm, Kdm, Pdm, cps):
    j = julia.Julia()
    j.include("src/dmcp_alg.jl")
    jl_dmcp_alg = j.eval("exec_dmcp")
    P, A = jl_dmcp_alg(Kim, Pim, Idm, Kdm, Pdm, cps)
    return A

def dmcp_alg_debug(Kim, Pim, Idm, Kdm, Pdm, cps):
    j = julia.Julia()
    j.include("src/dmcp_alg.jl")
    jl_dmcp_alg = j.eval("exec_dmcp")
    P, A = jl_dmcp_alg(Kim, Pim, Idm, Kdm, Pdm, cps)
    return P, A