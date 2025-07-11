using Plots;

include("gaussian_process.jl")
include("kernel_functions.jl")
include("kernels.jl")
include("mean_functions.jl")
include("objective_functions.jl")

function main()
    BUDGET = 20
    x = range(-10, 10, length = 100)
    y = range(-10, 10, length = 100)
    X = repeat(x, inner = length(y))
    Y = repeat(y, outer = length(x))
    opt, f = cross_in_tray(X, Y)
    XY = hcat(X,Y) # Creates a mesh

end

main()