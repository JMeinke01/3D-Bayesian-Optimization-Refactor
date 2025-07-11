# Given hyperparameters, creates a new Matern32 kernel
function matern32(θ)
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(3) * dist / ℓ
        return  σ^2 * (1 + frac) * exp(-frac)
    end
    return κ
end

# Given hyperparameters, creates a new Matern52 kernel
function matern52(θ)
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        frac = sqrt(5) * dist / ℓ
        return  σ^2 * (1 + frac + (frac^2) / 3) * exp(-frac)
    end
    return κ
end

# Given hyperparameters, creates a new squared exponential kernel
function squared_exponential(θ) 
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = diff' * diff
        return  σ^2 * exp(-dist /(2 * ℓ^2))
    end
    return κ
end

# Given hyperparameters, creates a new periodic kernel
function periodic(θ)
    σ = θ[1]
    ℓ = θ[2]
    p = θ[3]
    function κ(x::Vector{Float64}, x_other::Vector{Float64})
        @assert length(x) == length(x_other)
        diff = x - x_other
        dist = sqrt(diff' * diff)
        return σ^2 * exp(-2*sin(pi*dist / p)^2 / ℓ^2)            
    end
    return κ
end