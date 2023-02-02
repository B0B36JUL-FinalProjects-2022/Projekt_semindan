using Pkg
Pkg.activate(".")
using Titanic
using TOML
deps = TOML.parsefile("Project.toml")["deps"]
for d in deps
    if !isdefined(Titanic, Symbol(d))
        println("$d is not being used")
    end
end
