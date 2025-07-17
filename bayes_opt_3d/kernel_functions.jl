# Given hyperparameters, creates a new Matern32 kernel
function matern32(θ)
    # σ = θ[1] ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}, θ) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(3) * dist / θ[2]
        return  θ[1]^2 * (1 + frac) * exp(-frac)
    end
    return κ, θ
end

# Given hyperparameters, creates a new Matern52 kernel
function matern52(θ)
    # σ = θ[1] ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}, θ) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(5) * dist / θ[2]
        return  θ[1]^2 * (1 + frac + (frac^2) / 3) * exp(-frac)
    end
    return κ, θ
end


# function matern52_κ(x::Vector{Float64}, x_other::Vector{Float64}, θ) #representation of our kernel function
#         @assert length(x) == length(x_other)
#         diff = x - x_other
#         dist = sqrt(diff' * diff)
#         frac = sqrt(5) * dist / θ[2]
#         return  θ[1]^2 * (1 + frac + (frac^2) / 3) * exp(-frac)
# end

# # Given hyperparameters, creates a new Matern52 kernel
# function Matern52(θ)
#     return RadialBasisFunction(matern52_κ, θ)
# end


function squared_exponential_κ(x::Vector{Float64}, x_other::Vector{Float64}, θ) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = diff' * diff
        return  θ[1]^2 * exp(-dist /(2 * θ[2]^2))
    end

function squared_exponential(θ)
    return RadialBasisFunction(squared_exponential_κ, θ)
end

# Given hyperparameters, creates a new squared exponential kernel
# function squared_exponential(θ) 
#     # σ = θ[1] ℓ = θ[2]
#     function κ(x::Vector{Float64}, x_other::Vector{Float64}, θ) #representation of our kernel function
#         @assert length(x) == length(x_other)
#         diff = x - x_other
#         dist = diff' * diff
#         return  θ[1]^2 * exp(-dist /(2 * θ[2]^2))
#     end
#     return κ, θ
# end

# Given hyperparameters, creates a new periodic kernel
function periodic(θ)
    # σ = θ[1] ℓ = θ[2] p = θ[3]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}, θ)
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        return θ[1]^2 * exp(-2*sin(pi*dist / θ[3])^2 / θ[2]^2)            
    end
    return κ, θ
end