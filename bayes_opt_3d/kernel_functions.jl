#=
    If θ is a vector of length 2, then the hyperparameters are θ[1] = σ
    and θ[2] = ℓ. If θ is length 3, then the first two entries represent
    the same hyperparameter, but the third is the periodicity 
    hyperparameter
=#

# Constructs the matern32 kernel function
function matern32_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = sqrt(3) * dist / θ[2]
    return  θ[1]^2 * (1 + frac) * exp(-frac)
end

function ∂M32_∂σ(M32::RadialBasisFunction)
θ = M32.hyperparameters
κ = M32.kernel
return 2 / θ[1] * κ
end

function ∂M32_∂ℓ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = sqrt(3) * dist / θ[2]
    return  θ[1]^2 * (1 + frac) * exp(-frac) * frac / θ[2] + exp(-frac) * (-frac / θ[2])

end

function ∇matern32(M32::RadialBasisFunction)
    ∂σ = ∂M32_∂σ(M32)
    return (∂σ, ∂M32_∂ℓ)
end

# Given hyperparameters, creates a new Matern32 kernel
function matern32(θ::AbstractVector{<:Real})
    return RadialBasisFunction(matern32_κ, θ)
end

# Constructs the matern52 kernel function
function matern52_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real}) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(5) * dist / θ[2]
        return  θ[1]^2 * (1 + frac + (frac^2) / 3) * exp(-frac)
end

function ∂M52_∂σ(M52::RadialBasisFunction)
    θ = M52.hyperparameters
    κ = M52.kernel
    return 2 \ θ[1] * κ
end

function ∂M52_∂ℓ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = sqrt(5) * dist / θ[2]
    term = θ[1]^2 * exp(-frac)
    return term * (1 + frac + (frac^2) / 3) * frac / θ[2] + ((-frac / θ[2]) - 2 * (frac^2) / (3 * θ[2]))  
end

# Given hyperparameters, creates a new Matern52 kernel
function matern52(θ::AbstractVector{<:Real})
    return RadialBasisFunction(matern52_κ, θ)
end

function ∇matern52(M52::RadialBasisFunction)
    ∂σ = ∂M52_∂σ(M52)
    return (∂σ, ∂M52_∂ℓ)
end
# Constucts the square exponential kernel function
function squared_exponential_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real}) #representation of our kernel function
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = diff' * diff
    return  θ[1]^2 * exp(-dist / (2 * θ[2]^2))
end

# Computes the partial derivative of the SE kernel function with respect to σ
function ∂SE_∂σ(SE::RadialBasisFunction)
    θ = SE.hyperparameters
    κ = SE.kernel
    return 2 \ θ[1] * κ
end

# Computes the partial derivative of the SE kernel function with respect to ℓ
function ∂SE_∂ℓ(SE::RadialBasisFunction)
    θ = SE.hyperparameters
    κ = SE.kernel
    return κ * log(κ) * -2 / θ[2]
end

# Given hyperparameters, creates a new squared exponential kernel
function squared_exponential(θ::AbstractVector{<:Real})
    return RadialBasisFunction(squared_exponential_κ, θ)
end

# Constructs the gradient function(s) of the squared exponential kernel with respect to θ
function ∇squared_exponential(SE::RadialBasisFunction) 
    ∂σ = ∂SE_∂σ(SE)
    ∂ℓ = ∂SE_∂ℓ(SE)
    return (∂σ, ∂ℓ)
end

# Constructs the periodic kernel function
function periodic_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    return θ[1]^2 * exp(-2*sin(pi*dist / θ[3])^2 / θ[2]^2)            
end

function ∂p_∂ℓ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    return 4 * sin(π * dist / θ[3])^2 / ℓ^3
end


function ∂p_∂σ(P::Periodic)
    θ = P.hyperparameters
    κ = P.kernel
    return 2 / θ[1] * κ
end


function ∂p_∂p(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = π * dist / θ[3]
    return (-2 / θ[2]) * 2cos(frac) * sin(frac) * (-frac / θ[3])
end


function linear_periodic(θ::AbstractVector{<:Real})
    return Periodic(periodic_κ + squared_exponential_κ, θ)
end


function ∇linear_periodic(L::RadialBasisFunction, P::Periodic)
    ∇SE = ∇squared_exponential(L)
    ∇P = ∇(P)
    return (∇SE[1] + ∇P[1], ∇SE[2] + ∇P[2], ∇P[3])
end


function ∇periodic(P::Periodic)
    κ = P.kernel
    return(∂p_∂σ, κ * ∂p_∂ℓ, κ * ∂p_∂p)
end

# Given hyperparameters, creates a new periodic kernel
function periodic(θ::AbstractVector{<:Real})
    return Periodic(periodic_κ, θ)
end
