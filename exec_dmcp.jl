using ArgParse

function parse_commandline()
    Idm, Kdm, Pdm
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
        
    end

    return parse_args(ARGS, s)
end

parsed_args = parse_commandline()

using DelimitedFiles

## reading input
println("reading input files")
imK = readdlm(parsed_args["imK"], ',')
imP = readdlm(parsed_args["imP"], ',')

dmK = readdlm(parsed_args["dmK"], ',')
dmP = readdlm(parsed_args["dmP"], ',')
dmIm = readdlm(parsed_args["dmIm"], ',')[end:-1:1, 1:end]'

cps = readdlm(parsed_args["cps"], ',')

println("executing algorithm")
transform = exec_dmcp(imK, imP, dmIm, dmk, dmP, cps)

println("writing to $(parse_args["out"])")
writedlm(parsed_args["out"], transform, ',') 