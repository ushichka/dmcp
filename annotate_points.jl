using GLMakie
using CSV
using DelimitedFiles

## read input
dm = readdlm("data/dm.csv", ',')[end:-1:1, 1:end]'

## start drawing
fig = Figure()
ax1 = Axis(fig[1, 1:2])
ax2 = Axis(fig[2, 1:2])

pts = Matrix{Float64}(undef, 0, 2)
#push!(pts, [0, 1])
#push!(pts, [4, 2])
#pts = hcat(pts...)' |> x -> convert(Matrix{Float64}, x)

ptsO = Observable(pts)

sdm = heatmap!(ax1, dm)
s1 = scatter!(ax1, ptsO)

on(events(fig).mousebutton, priority=0) do event
    test = event
    if event.button == Mouse.left
        if event.action == Mouse.press
            if mouseover(ax1.scene, s1) # depth map

            else # image

            end
            mpos = mouseposition(ax1.scene)
            println("mouse presssed at $mpos")
            # do something
            #ptsO[] = rand(Float64, (3, 2)) * 500
            ptsO[] = vcat(ptsO[], [mpos[1] mpos[2]])
        else
            # do something else when the mouse button is released
        end
    end
    # Do not consume the event
    return Consume(false)
end

fig