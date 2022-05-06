using ArgParse

function parse_commandline()
    
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--dm", "-d"
            arg_type = String
            required = true
        "--dmsep"
            arg_type = Char
            default = ','
        "--im", "-i"
            arg_type = String
            required = true
        "--imsep"
            arg_type = Char
            default = ','
        "--out", "-o"
            arg_type = String
            default = "cps.csv"
        
    end

    return parse_args(ARGS, s)
end

parsed_args = parse_commandline()

println("loading libraries")
using GLMakie
using CSV
using DelimitedFiles

## read input
println("reading input")
dm = readdlm(parsed_args["dm"], parsed_args["dmsep"], Float32)[end:-1:1, 1:end]'
im = readdlm(parsed_args["im"], parsed_args["imsep"], Float32)[end:-1:1, 1:end]' # thermal images have one column to many...

dm = convert(Matrix{Float32}, dm)
im = convert(Matrix{Float32}, im)

## start drawing
println("preparing visualization")
fig = Figure()
ax1 = Axis(fig[1, 1:2], yrectzoom=false, xrectzoom=false, aspect=dm |> size |> s -> s[1] / s[2])
ax2 = Axis(fig[2, 1:2], yrectzoom=false, xrectzoom=false, aspect=im |> size |> s -> s[1] / s[2])

ptsDM = Matrix{Float64}(undef, 0, 2)
ptsIM = Matrix{Float64}(undef, 0, 2)

# plot on top
ptsDMO = Observable(ptsDM)
hmapDM = heatmap!(ax1, dm)
#scatDM = scatter!(ax1, ptsDMO, marker=:xcross, glowwidth=15, color=:firebrick, markersize=25)
# plot on bottom
#ptsU = Observable
ptsIMO = Observable(ptsIM)
hmapIM = heatmap!(ax2, im)
#scatIM = scatter!(ax2, ptsIMO, marker=:xcross, glowwidth=15, color=:firebrick, markersize=25)

function addpoint(ax, point, points) 
    scatter!(ax, point, marker=:xcross, glowwidth=15, color=:firebrick, markersize=25)
    setLabel(ax, point, point, size(points)[1]+1 |> string)
    return vcat(points, point)
end

function setLabel(ax, point, mpos, label)
    text!(ax, label, position=(mpos[1] + 10, mpos[2] + 10), space=:data, textsize=50, glowwidth=15, color=:ivory, font="Julia Mono")
end

on(events(fig).mousebutton, priority=0) do event
    if event.button == Mouse.left
        if event.action == Mouse.press
            if mouseover(ax1.scene, hmapDM) # depth map
                mpos = mouseposition(ax1.scene)
                point = [mpos[1] mpos[2]]
                points = ptsDMO[]
                ptsDMO[] = addpoint(ax1, point, points)                
            elseif mouseover(ax2.scene, hmapIM) # image
                mpos = mouseposition(ax2.scene)
                point = [mpos[1] mpos[2]]
                points = ptsIMO[]
                ptsIMO[] = addpoint(ax2, point, points)     
            else # clicked somewhere else
            end
        else
            # do something else when the mouse button is released
        end
    end
    # Do not consume the event
    return Consume(false)
end
println("executing...")
println("press ctrl+shift+left_mouse to reset window")
println("hold s and click to insert point")

while true
    gl_screen = fig |> display
    wait(gl_screen)
    if size(ptsDMO[]) == size(ptsIMO[])
        println("writing cps to $(parsed_args["out"])")
        cps = hcat(ptsDMO[], ptsIMO[])
        writedlm(parsed_args["out"], cps, ',')    
        exit(0)
    else
        println("cps must have same length")
        exit(1)
    end
end

println(ptsDMO[])
println(ptsIMO[])

## write annotated results
# concat to matrix rows: imx imy dmx dmy
# cps = hcat(ptsIMO[]..., ptsDMO[]...)
# writedlm( "data/cps.csv",  A, ',')