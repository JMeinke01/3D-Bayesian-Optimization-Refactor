using Plots;

include("kernel_functions.jl")
include("kernels.jl")
include("mean_functions.jl")
include("objective_functions.jl")
include("sampling.jl")
include("gaussian_process.jl")

function main()
    BUDGET = 20
    x = range(-10, 10, length = 100)
    y = range(-10, 10, length = 100)
    X = repeat(x, inner = length(y))
    Y = repeat(y, outer = length(x))
    opt, f = cross_in_tray(X, Y)
    XY = hcat(X,Y) # Creates a mesh
    num_init_samples = 10 # Initial samples
    σ = 1e-6 # Noise variable
    𝒟 = rand_sample(XY, num_init_samples, f, σ)
    θ = (1.0, 0.5) # Hyperparameters in the form of (σ, ℓ) or (σ, ℓ, p)
    κ = squared_exponential(θ)
    rbf = RadialBasisFunction(κ, θ)

end

main()