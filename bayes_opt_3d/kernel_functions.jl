#=
    If θ is a vector of length 2, then the hyperparameters are θ[1] = σ
    and θ[2] = ℓ. If θ is length 3, then the first two entries represent
    the same hyperparameter, but the third is the periodicity 
    hyperparameter
=#

# Let k_1 and k_2 be positive definite kernels. Then k = k_1 + k_2 is also
# a positive definite kernel. 

# Constructs the matern32 kernel function
function matern32_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = sqrt(3) * dist / θ[2]
    return  θ[1]^2 * (1 + frac) * exp(-frac)
end

# Returns the partial derivative of the matern32 kernel function with respect to σ
function ∂M32_∂σ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = sqrt(3) * dist / θ[2]
    return  2 * θ[1] * (1 + frac) * exp(-frac)
end

# Returns the partial derivative of the matern32 kernel function with respect to ℓ
function ∂M32_∂ℓ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = sqrt(3) * dist / θ[2]
    return  θ[1]^2 * (1 + frac) * exp(-frac) * frac / θ[2] + exp(-frac) * (-frac / θ[2])

end

# Returns the gradient of the matern32 kernel function
function ∇matern32()
    return (∂M32_∂σ, ∂M32_∂ℓ)
end

# Given hyperparameters, creates a new Matern32 kernel
function matern32(θ::AbstractVector{<:Real})
    return RadialBasisFunction(matern32_κ, θ)
end

# Constructs the matern52 kernel function
function matern52_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real}) 
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(5) * dist / θ[2]
        return  θ[1]^2 * (1 + frac + (frac^2) / 3) * exp(-frac)
end

# Returns the partial derivative of the matern52 kernel function with respect to σ
function ∂M52_∂σ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(5) * dist / θ[2]
        return  2 * θ[1] * (1 + frac + (frac^2) / 3) * exp(-frac)
end

# Returns the partial derivative of the matern52 kernel function with respect to ℓ
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

# Returns the gradient information of the matern52 kernel function
function ∇matern52()
    return (∂M52_∂σ, ∂M52_∂ℓ)
end

# Constucts the square exponential kernel function
function squared_exponential_κ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = diff' * diff
    return  θ[1]^2 * exp(-dist / (2 * θ[2]^2))
end

# Computes the partial derivative of the SE kernel function with respect to σ
function ∂SE_∂σ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = diff' * diff
    return  2 * θ[1] * exp(-dist / (2 * θ[2]^2))
end

# Computes the partial derivative of the SE kernel function with respect to ℓ
function ∂SE_∂ℓ(SE::RadialBasisFunction)
    σ, ℓ = SE.hyperparameters
    return (x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) -> begin
        diff = x - x_other
        dist = diff' * diff
        return (σ^2 * exp(-dist / (2 * ℓ^2))) * dist / ℓ^3
    end
end

# Given hyperparameters, creates a new squared exponential kernel
function squared_exponential(θ::AbstractVector{<:Real})
    return RadialBasisFunction(squared_exponential_κ, θ)
end

# Constructs the gradient function(s) of the squared exponential kernel with respect to θ
function ∇squared_exponential() 
    return (∂SE_∂σ, ∂SE_∂ℓ)
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
    return θ[1]^2 * exp(-2*sin(pi*dist / θ[3])^2 / θ[2]^2) * 4 * sin(π * dist / θ[3])^2 / ℓ^3
end


function ∂p_∂σ(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    return 2 * θ[1] * exp(-2*sin(pi*dist / θ[3])^2 / θ[2]^2)   
end


function ∂p_∂p(x::Vector{Float64}, x_other::Vector{Float64}, 
    θ::AbstractVector{<:Real})
    @assert length(x) == length(x_other)
    diff = x - x_other
    dist = sqrt(diff' * diff)
    frac = π * dist / θ[3]
    return θ[1]^2 * exp(-2*sin(pi*dist / θ[3])^2 / θ[2]^2) * (-2 / θ[2]) * 2cos(frac) * sin(frac) * (-frac / θ[3])
end


# Returns the gradient information of the periodic kernel
function ∇periodic()
    return(∂p_∂σ, ∂p_∂ℓ, ∂p_∂p)
end

# Given hyperparameters, creates a new periodic kernel
function periodic(θ::AbstractVector{<:Real})
    return Periodic(periodic_κ, θ)
end

# Creates a linear periodic kernel
function linear_periodic(θ::AbstractVector{<:Real})
    lin_per = (x, y, θ) -> squared_exponential_κ(x, y, θ[1:2]) + periodic_κ(x, y, θ[3:5])
    return Periodic(lin_per, θ)
end

# Returns the gradient infromation of the linear periodic kernel
function ∇linear_periodic()
    ∇SE = ∇squared_exponential()
    ∇P = ∇periodic()
    ∂σ = (x, y, θ) -> ∇SE[1](x, y, θ[1:2]) + ∇P[1](x, y, θ[3:5])
    ∂ℓ = (x, y, θ) -> ∇SE[2](x, y, θ[1:2]) + ∇P[2](x, y, θ[3:5])
    return (∂σ, ∂ℓ, ∇P[3])
end