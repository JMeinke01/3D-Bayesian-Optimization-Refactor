using LinearAlgebra
using Distributions

# includes("mean_functions")
# includes("kernels.jl")

abstract type AbstractGaussianProcess end

mutable struct GaussianProcess{T1, T2, T3} <: AbstractGaussianProcess
    μ::T1
    kernel::T2
    Κ_ss::T3
    Κ_xx::T3
    Κ_xs::T3
end

function GaussianProcess(μ::T1, kernel::T2, Κ_ss::T3, 
    Κ_xx::T3, Κ_xs::T3) where {T1 <: MeanFunction, T2 <: AbstractKernel, 
    T3 <: AbstractMatrix{<: Real}}
    return GaussianProcess{T1, T2, T3}(μ, kernel, Κ_ss, Κ_xx, Κ_xs)
end

function predict_f(GP::AbstractGaussianProcess, X::AbstractMatrix{<:Real}, 
    X_star::AbstractMatrix{<:Real}, prev::Int)
    Κ_ss = GP.Κ_ss
    Κ_xx = GP.Κ_xx
    Κ_xs = GP.Κ_xs
    μ = GP.μ
    Κ_xx_v = @view Κ_xx[1:prev, 1:prev]
    Κ_xs_v = @view Κ_xs[1:prev, 1:size(Κ_xs, 2)]
    y = X[:, 3]
    L = cholesky(Κ_xx_v)
    α = calculate_α(L, y, μ, X)
    μ_post = calculate_μ_post(α, Κ_xs_v, μ, X_star)
    Α = calculate_Α(L, Κ_xs_v)
    σ_post = calculate_σ_post(Α, Κ_ss)
    return μ_post, sqrt.(σ_post)
end


function calculate_α(L::LinearAlgebra.Cholesky, y::AbstractVector{<:Real}, 
    μ::MeanFunction, X::AbstractMatrix{<:Real}) 
    return L \ (y - μ(X))
end

function calculate_μ_post(α::AbstractVector{<:Real}, Κ_xs::AbstractMatrix{<:Real},
    μ::MeanFunction, X_star::AbstractMatrix{<:Real})
    return μ(X_star) + dot(Κ_xs, α)
end

function calculate_Α(L::LinearAlgebra.Cholesky, Κ_xs::AbstractMatrix{<:Real})
    return L \ Κ_xs
end

function calculate_σ_post(Α::AbstractMatrix{<:Real}, Κ_ss::AbstractMatrix{<:Real})
    Σ =  Κ_ss - dot(Α, Α)
    return diag(Σ)
end

function expected_improvement(GP::AbstractGaussianProcess, X::AbstractMatrix{<:Real}, 
    X_star::AbstractMatrix{<:Real}, prev::Int; ζ = 0.1)
    μ, σ = predict_f(GP, X, X_star, prev)
    f_opt = minimum(X)
    imp = @. (f_opt - μ - ζ)
    z = imp ./ σ
    return imp .* cdf.(Ref(Normal()), z) .+ σ .* pdf.(Ref(Normal()), z), μ
end