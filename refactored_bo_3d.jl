using Plots;

include("kernel_functions.jl")
include("kernels.jl")
include("mean_functions.jl")
include("objective_functions.jl")
include("sampling.jl")
include("gaussian_process.jl")

function main()
    BUDGET = 200
    x = range(-10, 10, length = 100)
    y = range(-10, 10, length = 100)
    X = repeat(x, inner = length(y))
    Y = repeat(y, outer = length(x))
    opt, f = cross_in_tray(X, Y)
    display(plot(x, y, f, st=:surface))
    XY = hcat(X,Y) # Creates a mesh
    num_init_samples = 10 # Initial samples
    σ = 1e-6 # Noise variable
    𝒟 = rand_sample(XY, num_init_samples, f, σ)
    θ = (3.0, 1.5) # Hyperparameters in the form of (σ, ℓ) or (σ, ℓ, p)
    κ, θ = squared_exponential(θ)
    rbf = RadialBasisFunction(κ, θ)
    min = 0;
    for i in 1 : size(XY)[1]
        if f(XY[i, 1], XY[i, 2]) < min
            min = f(XY[i, 1], XY[i, 2])
        end
    end
    println(min)
    # We will use the value of 1e-6 for jitter in our covariance matrix calculations
    Κ_ss = eval_KXX(rbf, XY, 1e-6)
    Κ_xx = eval_kxx(rbf, 𝒟[:, 1:2], BUDGET, 1e-6)
    Κ_xs = eval_KxX(rbf, 𝒟[:, 1:2], XY, BUDGET)

    μ_pri = mean_zero()

    GP = GaussianProcess(μ_pri, κ, Κ_ss, Κ_xx, Κ_xs)
    
    for i in num_init_samples : BUDGET
        # μ_post, std = predict_f(GP, 𝒟, XY, num_init_samples)
        exp_imp, μ_post = expected_improvement(GP, 𝒟, XY)
        # println(size(exp_imp), " ", size(μ_post))
        𝒟 = best_sampling_point(exp_imp, XY, 𝒟, f, σ)
        if i != BUDGET
            GP.Κ_xx = update_KXX(rbf, GP.Κ_xx, i, 𝒟[:, 1:2], 1e-6)
            GP.Κ_xs = update_kxX(rbf, GP.Κ_xs, i, 𝒟[:, 1:2], XY)
        end
        # Z = reshape(μ_post, length(x), length(y))
        Z = reshape(μ_post, length(x), length(y))
        display(plot(x, y, Z, st =:surface))
        
    end
    println(minimum(𝒟[:, 3]))
end

main()