using PackageCompiler

create_library(".", "build";
                lib_name="libDMCP",
                force=true)

exit()