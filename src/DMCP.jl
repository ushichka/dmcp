module dmcp

#export find_transformation_dmcp
include("dmcp_alg.jl")

function find_transformation_dmcp(Kim::Matrix{Cdouble}, Pim::Matrix{Cdouble}, Idm::Matrix{Cdouble}, Kdm::Matrix{Cdouble}, Pdm::Matrix{Cdouble}, cps::Matrix{Cdouble})::Matrix{Cdouble}
    return exec_dmcp(Kim, Pim, Idm, Kdm, Pdm, cps)
end

end