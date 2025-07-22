using Optim;
using ForwardDiff;
using LinearAlgebra;

# Calculates the negative log likelihood to be used
function negative_log_likelihood(Kc::LinearAlgebra.Cholesky, Κ_xx::AbstractMatrix{<:Real}, 
    y::AbstractVector{<:Real})
    α = 0.5 .* dot(y, Kc \ y) 
    β = log(det(Κ_xx))
    c = size(Κ_xx)[1]/2 * log(π/2)
    return α + β + c
end

# Calculates the gradient of the negative log likelihood
function ∇_nll(Kc::LinearAlgebra.Cholesky, y::AbstractVector{<:Real}, 
    grad::AbstractVector{T}, σ) where T <: Function
    Κ_inv = inv(Kc)
    grad_info = Vector{Float64}(0, 3)
    for i in 1:len(grad)
        grad_matrix = grad_cov_matrix(X, grad[i], σ)
        α = y' * Κ_inv
        Α = -0.5 * α * grad_matrix * α'
        prod = Κ_inv * grad_matrix
        β = tr(prod)
        grad_info[i] = Α + β        
    end
    return grad_info
end

# Creates a covariance matrix based on the new gradient equation
function grad_cov_matrix(X::AbstractMatrix{<:Real}, grad::Function, σ::Real)
    new_cov = zeros(len(X), len(X))
    for i in 1 : len(X)
        for j in i : len(X)
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
function optimize_hypers(nll::Real, θ::AbstractVector{<:Real}, 
    lbs::AbstractVector{<:Real}, ubs::AbstractVector{<:Real})
    res = optimize(nll, grad, θ, BFGS(); autodiff =:forward)
    return res
end
