# (c) Julian Jandeleit 2022
using MAT # to read demo
using LinearAlgebra
la = LinearAlgebra

include("absoluteOrientationQuaternionHorn.jl")



## -- helper methods for single pose calibration --

# cps matrix with each row a correspondence. Expects rows [point2D point3D] = [u v X Y Z]
# returns calibrated camera pinhole projection matrix P
function estimate_projection_matrix_dlt(cps)

    # build knows system matrix 2k x 12 A that determines coefficients
    rows = []
    for row in eachrow(cps)

        # 2D Point
        u = row[1]
        v = row[2]

        # 3D Point
        X = row[3]
        Y = row[4]
        Z = row[5]

        push!(rows, [-X -Y -Z -1 0 0 0 0 u * X u * Y u * Z u])
        push!(rows, [0 0 0 0 -X -Y -Z -1 v * X v * Y v * Z v])
    end
    A = vcat(rows...)

    # solve homogeneous linear system Ax = 0. x Represent the coefficients of the camera matrix
    # solve in least squares sense regarding reprojection error using SVD 
    U, S, V = la.svd(A)
    P = reshape(V[:, 12], 4, 3) |> la.transpose

    return P
end

# as in estimate_projection_matrix_dlt, with its result P
function reprojection_error(P, cps)
    world_pts = cps |> eachrow .|> row -> [row[3], row[4], row[5]]
    reprojected_points = []
    for pt in world_pts
        push!(reprojected_points, P * [pt; 1] |> p -> p / p[end] |> p -> p[1:end-1])
    end

    errors = []
    for i in 1:length(reprojected_points)
        p_cps = cps[i, 1:2]
        p_rep = reprojected_points[i]

        err = abs.(p_cps - p_rep) |> sum
        push!(errors, err)
    end

    return errors
end

sel_im_point(row) = row[1], row[2]
sel_dm_point(row) = row[3], row[4]

# where artificial camera K saw the point
# inv(K) represents the line from camera center to p
# p lies on on this line at distance given in depth_map
# TODO: does inv(K) results in normalized koordinates? (length 1) and make sure it is compatible with depth map units 
point_in_depth_map_to_camera_space(px::Float64, py::Float64, Kdm::Array{Float64,2}, Idm::Array{Float32,2}) = Idm[round(Int, py), round(Int, px)] * inv(Kdm) * [px, py, 1]

# extrinsic matrix: world -> camera
extract_camera_extrinsic_matrix(K, P) = inv(K) * P
extract_camera_pose_matrix(K, P) = extract_camera_extrinsic_matrix(K, P) |> x -> [x; 0 0 0 1] |> inv
point_in_camera_space_to_world_space(px::Float64, py::Float64, pz::Float64, P, K) = (extract_camera_pose_matrix(K, P) |> x -> x[1:3, 1:4]) * [px, py, pz, 1]



# -- helper methods for transformation estimation --

# estimates how much space 2 is larger than space 1 for camera matrices P1 and P2
function estimate_scaling(K1, P1, K2, P2)
    E1 = [inv(K1) * P1; 0 0 0 1]
    E2 = [inv(K2) * P2; 0 0 0 1]

    # ref: https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813
    sv_1 = [norm(E1[1:3, 1]) norm(E1[1:3, 2]) norm(E1[1:3, 3])] # vector of each norm of column in rotation matrix
    sv_2 = [norm(E2[1:3, 1]) norm(E2[1:3, 2]) norm(E2[1:3, 3])]

    scale_factor = norm(sv_2) / norm(sv_1)
    return scale_factor
end

function point_in_world_space_to_camera_space(K, P, point, scale_factor)
    E = [inv(K) * P; 0 0 0 1]
    pc = E * [point; 1]
    pc = pc / pc[end]
    return pc[1:end-1] * scale_factor
end

# cps: a matrix with 4 columns and n rows. Each row represents a correspondence. The columns are organized with pointImgX pointImgY pointDmX pointDmY
# K: camera intrinsic matrix
# Pim in calibration space
function exec_dmcp(Kim, Pim, Idm, Kdm, Pdm, cps)

    # -- single pose calibration -- 

    # bring cps from depth map to camera space dm (3D)
    bring_to_camera_space_dm(px, py) = point_in_depth_map_to_camera_space(px, py, Kdm, Idm)
    pdm_camera = cps |> eachrow .|> sel_dm_point .|> p -> bring_to_camera_space_dm(p[2], p[1]) # TODO: why does it need to be switched? Does it still work for old setting? Why no crash before?

    # bring cps from camera space to world space
    bring_to_world_space(px, py, pz) = point_in_camera_space_to_world_space(px, py, pz, Pdm, Kdm)
    pdm_world = pdm_camera .|> p -> bring_to_world_space(p[1], p[2], p[3])

    # estimate pinhole projection matrix from image to world space correspondences
    # P is the projection matrix of the camera that observed input Image, in world space
    cps_mat_img_world = hcat(first.(cps |> eachrow .|> sel_im_point), last.(cps |> eachrow .|> sel_im_point), hcat(pdm_world...)')
    P = estimate_projection_matrix_dlt(cps_mat_img_world)
    repr_err = reprojection_error(P, cps_mat_img_world)


    # -- calculate transformation --

    # estimate scaling between camera calibration space and world space
    scale_factor = estimate_scaling(Kim, P, Kim, Pim)

    # bring cps from world to camera_space (of the camera that took the image)
    bring_to_camera_space_est(p) = point_in_world_space_to_camera_space(Kim, P, p, scale_factor)
    pdm_camera_est = pdm_world .|> bring_to_camera_space_est

    # bring cps from camera space to calibration space
    # this step works because P and Pim describe the same camera and thus have the same camera space. how ever their "world space differs" (calibration vs world space)
    bring_to_calibration_space(px, py, pz) = point_in_camera_space_to_world_space(px, py, pz, Pim, Kim)
    pdm_calib = pdm_camera_est .|> p -> bring_to_calibration_space(p[1], p[2], p[3])

    # solve absolute orientation problem between cps in world and calibration space
    # we want the transformation from calibration space to world space
    s, R, T = absoluteOrientationQuaternion(hcat(pdm_calib...), hcat(pdm_world...), false)

    # build affine transform A from individual parts
    A = [[s * R T]; 0 0 0 1]

    return A
    #return pdm_camera, pdm_world, P, repr_err, scale_factor, cps_mat_img_world, pdm_camera_est, pdm_calib
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

    return (Ith=Ith, Kth=Kth, Pth=Pth, Idm=Idm, Kdm=Kdm, Pdm=Pdm, cps=cps)
end

#di = demo_input = demo()

# create input correspondences
#t1, t2, t3, t4, t5, t6, t7, t8 = exec_dmcp(di.Kth, di.Pth, di.Idm, di.Kdm, di.Pdm, di.cps)
#A = exec_dmcp(di.Kth, di.Pth, di.Idm, di.Kdm, di.Pdm, di.cps);

#t3_pose = inv([inv(di.Kth) * t3; 0 0 0 1])