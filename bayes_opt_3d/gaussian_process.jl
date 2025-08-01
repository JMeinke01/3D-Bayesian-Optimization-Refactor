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
    Κ_xx::T3, Κ_xs::T3) where {T1 <: Function, T2 <: AbstractKernel, 
    T3 <: AbstractMatrix{<: Real}}
    return GaussianProcess{T1, T2, T3}(μ, kernel, Κ_ss, Κ_xx, Κ_xs)
end

@views function predict_f(GP::AbstractGaussianProcess, X::AbstractMatrix{<:Real}, 
    X_star::AbstractMatrix{<:Real})
    Κ_ss = GP.Κ_ss
    μ = GP.μ
    len = size(X)[1]
    Κ_xx_v = GP.Κ_xx[1:len, 1:len]
    Κ_xs_v = GP.Κ_xs[1:len, 1:size(X_star)[1]]
    y = X[:, 3]
    Kc = cholesky(Κ_xx_v)
    μ_post = calculate_μ_post(Kc, y, Κ_xs_v, μ, X, X_star)
    σ_post = calculate_σ_post(Kc, Κ_xs_v, Κ_ss)
    @inbounds begin
        @simd for i in length(σ_post)
            σ_post[i] = sqrt(σ_post[i])
        end
    end
    return μ_post, σ_post
end


function calculate_μ_post(Kc::LinearAlgebra.Cholesky, y::AbstractVector{<:Real}, Κ_xs::AbstractMatrix{<:Real},
    μ::Function, X::AbstractMatrix{<:Real}, X_star::AbstractMatrix{<:Real})
    α = Kc \ (y - μ(X))
    return μ(X_star) + Κ_xs' * α
end


@views function calculate_σ_post(Kc::LinearAlgebra.Cholesky, Κ_xs::AbstractMatrix{<:Real}, 
    Κ_ss::AbstractMatrix{<:Real})
    Α = Kc.L \ Κ_xs
    Α_diag = zeros(length(diag(Κ_ss)))
    σ = zeros(size(Κ_ss, 1))
    @inbounds begin
        @simd for i in 1 : length(diag(Κ_ss))
            Α_diag[i] = dot(Α[:, i], Α[:, i])
        end
        @simd for i in 1 : length(σ)
            σ[i] = Κ_ss[i, i] - Α_diag[i]
        end
    end
    return σ
end

function expected_improvement(GP::AbstractGaussianProcess, X::AbstractMatrix{<:Real}, 
    X_star::AbstractMatrix{<:Real}; ζ = 0.1)
    μ, σ = predict_f(GP, X, X_star)
    f_opt = minimum(X)
    imp = zeros(length(μ))
    z = zeros(length(μ))
    @inbounds begin
        @simd for i in length(μ)
            imp[i] = f_opt - μ[i] - ζ
            z[i] = imp[i] / σ[i]
        end 
    end
    return imp .* cdf.(Ref(Normal()), z) .+ σ .* pdf.(Ref(Normal()), z), μ
end

