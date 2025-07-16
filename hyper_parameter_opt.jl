function negative_log_likelihood(Kc::LinearAlgebra.Cholesky, Κ_xx::AbstractMatrix{<:Real}, 
    y::AbstractVector{<:Real})
    α = 0.5 .* y' * Kc \ y 
    β = log(det(Κ_xx))
    c = size(Κ_xx)[1]/2 * log(π/2)
    return α + β + c
end

