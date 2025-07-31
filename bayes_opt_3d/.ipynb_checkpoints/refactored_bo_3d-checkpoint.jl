using Plots;
using Measures;

include("kernel_functions.jl")
include("kernels.jl")
include("mean_functions.jl")
include("objective_functions.jl")
include("sampling.jl")
include("gaussian_process.jl")
include("hyper_parameter_opt.jl")

function main()
    BUDGET = 200
    x = range(-10, 10, length = 100)
    y = range(-10, 10, length = 100)
    X = repeat(x, inner = length(y))
    Y = repeat(y, outer = length(x))
    opt, f = cross_in_tray(X, Y)
    # display(plot(x, y, f, st=:surface, xlabel = "x", ylabel = "y", title = "Cross-in-Tray Function"))
    XY = hcat(X,Y) # Creates a mesh
    num_init_samples = 10 # Initial samples
    σ = 1e-6 # Noise variable
    𝒟 = rand_sample(XY, num_init_samples, f, σ)
    θ = [10.0, 1.0] # Hyperparameters in the form of (σ, ℓ) or (σ, ℓ, p)
    rbf = squared_exponential(θ)
    # temp = linear_periodic(θ)
    # temp2 = ∇linear_periodic()

    # min = 0;
    # for i in 1 : size(XY)[1]
    #     if f(XY[i, 1], XY[i, 2]) < min
    #         min = f(XY[i, 1], XY[i, 2])
    #     end
    # end
    # println(min)

    # We will use the value of 1e-6 for jitter in our covariance matrix calculations
    Κ_ss = eval_KXX(rbf, XY, 1e-6)
    Κ_xx = eval_kxx(rbf, 𝒟[:, 1:2], BUDGET, 1e-6)
    Κ_xs = eval_KxX(rbf, 𝒟[:, 1:2], XY, BUDGET)

    μ_pri = mean_zero()
    GP = GaussianProcess(μ_pri, rbf.kernel, Κ_ss, Κ_xx, Κ_xs)
    
    for i in num_init_samples : BUDGET
        # println(i)
        exp_imp, μ_post = expected_improvement(GP, 𝒟, XY)
        𝒟 = best_sampling_point(exp_imp, XY, 𝒟, f, σ)
        if i != BUDGET
            GP.Κ_xx = update_KXX!(rbf, GP.Κ_xx, i, 𝒟[:, 1:2], 1e-6)
            GP.Κ_xs = update_kxX!(rbf, GP.Κ_xs, i, 𝒟[:, 1:2], XY)
        end
        # if i % 25 == 0 && (i != num_init_samples && i < BUDGET) 
        #     println("in")
        # Κ_xx_v = @view GP.Κ_xx[1 : size(𝒟, 1), 1 : size(𝒟, 1)]
        # θ_opt = optimize_hypers(rbf.hyperparameters, [0.01, 0.01], [20.0, 20.0], 1e-6, 
        #     rbf, 𝒟[:, 1:2], 𝒟[:, 3], Κ_xx_v)
        # rbf.hyperparameters = θ_opt
        # rebuild_KXX!(rbf, GP.Κ_ss, XY, 1e-6)
        # rebuild_kxx!(rbf, GP.Κ_xx, 𝒟[:, 1:2], 1e-6)
        # rebuild_kxX!(rbf, GP.Κ_xs, 𝒟[:, 1:2], XY)
        # end
        # Z = reshape(μ_post, length(x), length(y))
        # display(plot(x, y, Z, ylabel = "y", xlabel = "x", st =:surface, title = "Untuned Hyperparameter Surrogate"))
    
        # println(𝒟[i, 3], " ", i)
    end
    println(findmin(𝒟[:, 3]))
end

main()


#=
Scatter plot code:

Before loop:
perc_err_vals = abs.((𝒟[:, 3] .- min) ./ min) .* 100
    scatter_pl = scatter([1:10], perc_err_vals, xlabel = "Sample #", ylabel = "Percent Error", legend = false, 
    title = "Cross-in-Tray Function Percent Error Data", color=:black, markersize=:8, markerstrokewidth=:2.1)
    scatter_min = scatter([1:10], [minimum(perc_err_vals)], xlabel = "Sample #", ylabel = "Percent Error", legend = false, 
    title = "Cross-in-Tray Function Minimum Percent Error", color=:black, markersize=:8, markerstrokewidth=:2.1)

Inside loop:
curr_z = 𝒟[i, 3]
        p_err = abs((curr_z - min) / min) * 100
        push!(perc_err_vals, p_err)
        scatter!(scatter_pl, [i + 1], [p_err], color =:red, marker=:xcross, markersize=:8, markerstrokewidth=:2.1)
        scatter!(scatter_min, [i + 1], [minimum(perc_err_vals)], color =:red, marker=:xcross, markersize=:8, markerstrokewidth=:2.1)

Outside Loop:
        plot(scatter_pl, scatter_min, layout = (1, 2), size = (1300, 600), margin = 10mm)
=#