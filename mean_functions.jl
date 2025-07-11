function mean_zero()
    function m(X::Matrix{<:Real})
        len = size(X, 1)
        return zeros(len, 1)
    end
    return m
end

function mean_constant(β::Real)
    function m(X::Matrix{Real})
        len = size(X,1)
        return fill(len, β)
    end
    return m
end

function mean_linear(β::Vector{<:Real})
    function m(X::Matrix{<:Real})
        @assert size(X, 1) == length(β)
        return dot(X, β)
    end
    return m
end