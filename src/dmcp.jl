using MAT
using ImageProjectiveGeometry

sel_dm_point(row) = row[3], row[4]

# where artificial camera K saw the point
# inv(K) represents the line from camera center to p
# p lies on on this line at distance given in depth_map
# TODO: does inv(K) results in normalized koordinates? (length 1) and make sure it is compatible with depth map units 
point_in_depth_map_to_camera_space(px ::Float64, py::Float64, Kdm::Array{Float64,2}, Idm::Array{Float32,2}) = Idm[round(Int,py),round(Int,px)] * inv(Kdm) * [px, py, 1]

# extrinsic matrix: world -> camera
extract_camera_extrinsic_matrix(K,P) = inv(K) * P
extract_camera_pose_matrix(K,P) = extract_camera_extrinsic_matrix(K,P) |> makehomogeneous |> inv |> makeinhomogeneous
point_in_camera_space_to_world_space(px ::Float64, py::Float64, pz::Float64, P, K) = (extract_camera_pose_matrix(K,P) |> makehomogeneous) * [px, py, pz, 1] |> makeinhomogeneous

# cps: a matrix with 4 columns and n rows. Each row represents a correspondence. The columns are organized with pointImgX pointImgY pointDmX pointDmY
# K: camera intrinsic matrix
function dmcp_alg(Idm, Kdm, cps) 

    # -- preprocessing --
    
    # bring cps from depth map to camera space (3D)
    bring_to_camera_space(px, py) = point_in_depth_map_to_camera_space(px,py, Kdm, Idm)
    pdm_camera = cps |> eachrow .|> sel_dm_point .|> p->bring_to_camera_space(p[1], p[2])

    # bring cps from camera space to world space
    bring_to_world_space(px,py,pz) = point_in_camera_space_to_world_space(px,py,pz,P,K)
    pdm_lidar = pdm_camera .|> bring_to_world_space

    # -- single pose calibration -- 

    # -- calculate transformation --

end



function demo()
    mat = matread("demo/dmcp_inputs_demo.mat")
    
    Ith = mat["I_th"]
    Kth = mat["K_th"]
    Pth = mat["P_th"]

    Idm = mat["I_dm"]
    Kdm = mat["K_dm"]
    Pdm = mat["P_dm"]

    cps = mat["cps"]

    return (Ith = Ith, Kth = Kth, Pth = Pth, Idm = Idm, Kdm = Kdm, Pdm = Pdm, cps = cps)
end

di = demo_input = demo()

# create input correspondences
dmcp_alg(di.Idm, di.Kdm, di.cps)
