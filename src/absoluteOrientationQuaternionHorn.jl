# (c) Julian Jandeleit 2022
# implementation analogous to Matlab implementation by Dr. Christian Wengert, Dr. Gerald Bianchi ETH Zurich, Computer Vision Laboratory, Switzerland
using Statistics
using LinearAlgebra
la = LinearAlgebra

function absoluteOrientationQuaternion(A, B, doScale)
    # test size of point sets
    c1, r1 = size(A)
    c2, r2 = size(B)

    # Number of points
    Na = r1;

    # Compute centroids
    Ca = mean(A |> eachcol)
    Cb = mean(B |> eachcol)

    # remove (substract) centroid
    An = A |> eachcol .|> col-> col- Ca
    An = hcat(An...)
    Bn = B |> eachcol .|> col-> col - Cb
    Bn = hcat(Bn...)

    # compute quaternions
    M = zeros(4,4)
    for i in 1:Na
        #Shortcuts
        a = [0;An[:,i]];
        b = [0;Bn[:,i]];    
        #Crossproducts
        Ma = [  a[1] -a[2] -a[3] -a[4] ; 
                a[2]  a[1]  a[4] -a[3] ; 
                a[3] -a[4]  a[1]  a[2] ; 
                a[4]  a[3] -a[2]  a[1]  ];
        Mb = [  b[1] -b[2] -b[3] -b[4] ; 
                b[2]  b[1] -b[4]  b[3] ; 
                b[3]  b[4]  b[1] -b[2] ; 
                b[4] -b[3]  b[2]  b[1]  ];
        #Add up
        M = M + Ma'*Mb;
    end

    # compute eigenvalues
    E = la.eigvecs(M)

    # compute rotation matrix
    e = E[:,4]
    M1 = [  e[1] -e[2] -e[3] -e[4] ; 
            e[2]  e[1]  e[4] -e[3] ; 
            e[3] -e[4]  e[1]  e[2] ; 
            e[4]  e[3] -e[2]  e[1]  ];
    M2 = [  e[1] -e[2] -e[3] -e[4] ; 
            e[2]  e[1] -e[4]  e[3] ; 
            e[3]  e[4]  e[1] -e[2] ; 
            e[4] -e[3]  e[2]  e[1]  ];

    R = M1'*M2

    #Retrieve the 3x3 rotation matrix
    R = R[2:4,2:4];

    # compute scale factor if necessary
    if doScale
        a = 0
        b = 0
        for i=1:Na
            a = a + Bn[:,i]'*R*An[:,i];
            b = b + Bn[:,i]'*Bn[:,i];
            s = b/a
        end 
    else
        s = 1
    end

    # final Translation vector
    T = Cb - s*R*Ca

    return s, R, T

end
