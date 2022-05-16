include("src/dmcp_alg.jl")
using ArgParse

function parse_commandline()

    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--imK"
            arg_type = String
            required = true
        "--imP"
            arg_type = String
            required = true
        
        "--dmK"
            arg_type = String
            required = true
        "--dmP"
            arg_type = String
            required = true
        "--dmIm"
            arg_type = String
            required = true

        "--cps"
            arg_type = String
            required = true
        
        "--out", "-o"
            arg_type = String
            default = "transform.csv"
        
        "--outPdlt"
            arg_type = String
            default = "Pdlt.csv"

    end

    return parse_args(ARGS, s)
end

parsed_args = parse_commandline()

using DelimitedFiles

## reading input
#println("reading input files")
imK = readdlm(parsed_args["imK"], ',')
imP = readdlm(parsed_args["imP"], ',')

dmK = readdlm(parsed_args["dmK"], ',')
dmP = readdlm(parsed_args["dmP"], ',')
dmIm = readdlm(parsed_args["dmIm"], ',')[end:-1:1, 1:end]

cps = readdlm(parsed_args["cps"], ',')

# convert types
dmIm = convert(Matrix{Float32}, dmIm)
cps = convert(Matrix{Float64}, cps)

#println("executing algorithm")
Pdlt, transform = exec_dmcp(imK, imP, dmIm, dmK, dmP, cps)

writedlm(parsed_args["outPdlt"], Pdlt, ',')
writedlm(parsed_args["out"], transform, ',') 
println("transformation saved to $(parsed_args["out"])")