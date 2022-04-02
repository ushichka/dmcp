using GLMakie
using CSV
using DelimitedFiles

## read input
dm = readdlm("export/dm.csv", ',')[end:-1:1,1:end]'

## start drawing
fig = Figure()
ax1 = Axis(fig[1, 1:2])
ax2 = Axis(fig[2, 1:2])

pts = []
push!(pts,[0,1])
push!(pts,[4,2])


pts = hcat(pts...)' |> x->convert(Matrix{Float64}, x)

ptsO = Observable(pts)

sdm =heatmap!(ax1, dm)
s1 = scatter!(ax1,ptsO)

on(events(fig).mousebutton, priority = 0) do event
    test = event
    if event.button == Mouse.left
        if event.action == Mouse.press
            println("over $(mouseover(ax1.scene,s1))") # needs to actually be over plot (e.g. image^^)
            println("mouse presssed at $(mouseposition(ax1.scene))")
            # do something
            ptsO[] = rand(Float64, (3,2)) * 500
        else
            # do something else when the mouse button is released
        end
    end
    # Do not consume the event
    return Consume(false)
end

fig