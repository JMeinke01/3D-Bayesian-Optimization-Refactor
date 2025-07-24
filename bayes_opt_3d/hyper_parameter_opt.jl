using Optim;
using ForwardDiff;
using LinearAlgebra;

# Calculates the negative log likelihood to be used
function negative_log_likelihood(Kc::LinearAlgebra.Cholesky, Κ_xx::AbstractMatrix{<:Real}, 
    y::AbstractVector{<:Real})
    α = 0.5 * dot(y, Kc \ y) 
    β = log(det(Κ_xx))
    c = size(Κ_xx)[1]/2 * log(π/2)
    return α + β + c
end

function nll(kernel::AbstractKernel, X::AbstractMatrix{<:Real}, 
    y::AbstractVector{<:Real}, σ::Real, Κ_xx::AbstractMatrix{<:Real})
    # rebuild_kxx!(kernel, Κ_xx, X, σ)
    Κ_xx = eval_kxx(kernel, X, size(X, 1), σ)
    Kc = cholesky(Κ_xx)
    α = 0.5 * dot(y, Kc \ y)
    β = log(det(Κ_xx))
    c = size(Κ_xx)[1] / 2 * log(π/2)
    return α + β + c
end

# Calculates the gradient of the negative log likelihood
function ∇_nll(Kc::LinearAlgebra.Cholesky, y::AbstractVector{<:Real}, 
    grad::AbstractVector{T}, σ) where T <: Function
    grad_info = Vector{Float64}(0, 3)
    for i in 1 : length(grad)
        grad_matrix = grad_cov_matrix(X, grad[i], σ)
        α = Kc \ y
        Α = -0.5 * a' * grad_matrix * α
        prod = Kc \ grad_matrix
        β = tr(prod)
        grad_info[i] = Α + β        
    end
    return grad_info
end

# Creates a covariance matrix based on the new gradient equation
function grad_cov_matrix(X::AbstractMatrix{<:Real}, grad::Function, σ::Real)
    n = length(X)
    new_cov = zeros(n, n)
    for i in 1 : n
        for j in i : n
            cov[i, j] = grad(X[i, :], X[j, :])
            cov[j, i] = cov[i, j]
            if i == j
                cov[i, j] += σ
            end
        end
    end
    return new_cov
end

# Optimizes the hyperparameters and returns new hyperparameters
function optimize_hypers(
    θ_init::AbstractVector{<:Real}, 
    lbs::AbstractVector{<:Real}, 
    ubs::AbstractVector{<:Real}, 
    σ::Real, 
    kernel::AbstractKernel, 
    X::AbstractMatrix{<:Real}, 
    y::AbstractVector{<:Real},
    Κ_xx::AbstractMatrix{<:Real})
    nll_clos = θ -> begin
        kernel.hyperparameters = θ
        θ = nll(kernel, X, y, σ, Κ_xx)
    end
    res = optimize(nll_clos, lbs, ubs, θ_init, Fminbox(BFGS()))
    println(Optim.minimizer(res))
    return Optim.minimizer(res)    
end
