abstract type AbstractKernel end
function eval_kernel(k::AbstractKernel) end


mutable struct RadialBasisFunction{T1, T2} <: AbstractKernel
    kernel::T1
    hyperparameters::T2
end

function RadialBasisFunction(
    kernel::T1,
    hyperparameters::T2) where {T1 <: Function, T2 <: AbstractVector{<:Real}}
    return RadialBasisFunction{T1, T2}(kernel, hyperparameters)
end

function get_noise_variance(kernel::AbstractKernel)
    return kernel.hyperparameters[1]
end

function eval_k(rbf::RadialBasisFunction, x::AbstractArray{<:Real}, 
    y::AbstractArray{<:Real})
    return rbf.kernel(x, y, rbf.hyperparameters)
end

(rbf::RadialBasisFunction)(x, y) = eval_k(rbf, x, y)    

mutable struct Periodic{T1, T2} <: AbstractKernel
    kernel::T1
    hyperparameters::T2
end

function Periodic(kernel::T1, 
    hyperparameters::T2) where {T1 <: Function, T2 <: AbstractVector{<:Real}}
    return Periodic{T1, T2}(kernel, hyperparameters)
end

# Creates the covariance matrix of the observed data which can be expanded, k_xx
@views function eval_kxx(kernel::AbstractKernel, X::AbstractMatrix{<:Real}, d::Int, σ::Real)
    cov = zeros(d, d)
    N = size(X)[1]
    @inbounds begin
        for i in 1 : N
            @simd for j in i : N
                cov[i, j] = eval_k(kernel, X[i, :], X[j, :])
                cov[j, i] = cov[i, j]
                if i == j
                    cov[i, j] += σ
                end
            end
        end
    end
    return cov
end

#  Calculates and creates the cross covariance matrix, KxX
@views function eval_KxX(kernel::AbstractKernel, X::AbstractMatrix{<:Real}, 
    X_star::AbstractMatrix{<:Real}, d::Int)
    m = size(X)[1]
    N = size(X_star)[1]
    cov = zeros(d, N)
    @inbounds begin
        for i in 1 : N
            @simd for j in 1 : m
                cov[j, i] = eval_k(kernel, X[j, :], X_star[i, :])
            end        
        end
    end
    return cov
end

# Calculates and creates the training set covariance matrix, KXX
@views function eval_KXX(kernel::AbstractKernel, X::AbstractMatrix{<:Real}, σ::Real) 
    d = size(X)[1]
    cov = zeros(d, d)
    k0 = eval_k(kernel, X[1, :], X[1, :]) + σ
    @inbounds begin
        for i in 1 : d
            cov[i,i] = k0
            @simd for j in i+1 : d
                cov[i,j] = eval_k(kernel, X[i, :], X[j, :])
                cov[j, i] = cov[i, j]
            end
        end
    end
    return cov
end

# Evaluates the kernel at given points
function eval_k(k::Periodic, 
    x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) 
    return k.kernel(x, y, k.hyperparameters)
end

# Updates the KxX covariance matrix to accommodate the new observation
@views function update_kxX!(kernel::AbstractKernel, Κ::AbstractMatrix{<:Real}, prev::Int, 
    X::AbstractMatrix{<:Real}, X_star::AbstractMatrix{<:Real})
    len = size(X_star)[1]
    @inbounds begin
        @simd for i in 1 : len
            Κ[prev + 1, i] = eval_k(kernel, X[prev + 1, :], X_star[i, :])
        end
    end
    return Κ
end

# Updates the KxX covariance matrix to accomodate the new observation
@views function update_KXX!(kernel::AbstractKernel, Κ::AbstractMatrix{<:Real}, 
    prev::Int, X::AbstractMatrix{<:Real}, σ) 
    @inbounds begin
        @simd for i in 1 : prev + 1
            Κ[prev + 1, i] = eval_k(kernel, X[prev + 1, :], X[i, :])
            Κ[i, prev + 1] = Κ[prev + 1, i]
        end
    end
    Κ[prev + 1, prev + 1] += σ
    return Κ
end


#=
The functions below modify the various covariance matrices in place enabling minimal
memory re-allocation/waste when optimizing the hyperparameters and re-evaluating all matrices
=#

function rebuild_KXX!(kernel::AbstractKernel, Κ::AbstractMatrix{<:Real}, 
    X::AbstractMatrix{<:Real}, σ::Real)
    N = size(X, 1)
    for i in 1 : N
        for j in i : N
            Κ[i,j] = eval_k(kernel, X[i, :], X[j, :])
            Κ[j, i] = Κ[i, j]
            if i == j
                Κ[i, j] += σ
            end
        end
    end
end

function rebuild_kxx!(kernel::AbstractKernel, Κ::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real}, σ::Real)
    N = size(X, 1)
    for i in 1 : N
        for j in i : N
            Κ[i,j] = eval_k(kernel, X[i, :], X[j, :])
            Κ[j, i] = Κ[i, j]
            if i == j
                Κ[i, j] += σ
            end
        end
    end
end

function rebuild_kxX!(kernel::AbstractKernel, Κ::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real}, X_star::AbstractMatrix{<:Real})
    m = size(X)[1]
    N = size(X_star)[1]
    for i in 1 : N
        for j in 1 : m
            Κ[j, i] = eval_k(kernel, X[j, :], X_star[i, :])
        end        
    end
end