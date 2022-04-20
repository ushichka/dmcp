using GLMakie
using CSV
using DelimitedFiles

## read input
dm = readdlm("data/dm.csv", ',')[end:-1:1, 1:end]'
im = readdlm("data/im.csv", ';')[end:-1:1, 1:end-1]' # thermal images have one column to many...

## start drawing
fig = Figure()
ax1 = Axis(fig[1, 1:2], yrectzoom=false, xrectzoom=false, aspect=dm |> size |> s -> s[1] / s[2])
ax2 = Axis(fig[2, 1:2], yrectzoom=false, xrectzoom=false, aspect=im |> size |> s -> s[1] / s[2])

ptsDM = Matrix{Float64}(undef, 0, 2)
ptsIM = Matrix{Float64}(undef, 0, 2)

# plot on top
ptsDMO = Observable(ptsDM)
hmapDM = heatmap!(ax1, dm)
scatDM = scatter!(ax1, ptsDMO, marker=:xcross, glowwidth=15, color=:firebrick, markersize=25)
# plot on bottom
#ptsU = Observable
ptsIMO = Observable(ptsIM)
hmapIM = heatmap!(ax2, im)
scatIM = scatter!(ax2, ptsIMO, marker=:xcross, glowwidth=15, color=:firebrick, markersize=25)

function setLabel(ax, ptsO, mpos)
    text!(ax, size(ptsO[])[1] |> string, position=(mpos[1] + 10, mpos[2] + 10), space=:data, textsize=25, glowwidth=15, color=:ivory, font="Julia Mono")
end

on(events(fig).mousebutton, priority=0) do event
    if event.button == Mouse.left
        if event.action == Mouse.press
            if mouseover(ax1.scene, hmapDM) # depth map
                mpos = mouseposition(ax1.scene)
                ptsDMO[] = vcat(ptsDMO[], [mpos[1] mpos[2]])
                setLabel(ax1, ptsDMO, mpos)
            elseif mouseover(ax2.scene, hmapIM) # image
                mpos = mouseposition(ax2.scene)
                ptsIMO[] = vcat(ptsIMO[], [mpos[1] mpos[2]])
                setLabel(ax2, ptsIMO, mpos)
            else # clicked somewhere else
            end
        else
            # do something else when the mouse button is released
        end
    end
    # Do not consume the event
    return Consume(false)
end

println("press ctrl+shift+left_mouse to reset window")
println("hold s and click to insert point")

fig

## write annotated results
# concat to matrix rows: imx imy dmx dmy
# cps = hcat(ptsIMO[]..., ptsDMO[]...)
# writedlm( "data/cps.csv",  A, ',')