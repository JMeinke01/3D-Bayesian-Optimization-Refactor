function rand_sample(X_star::AbstractMatrix{<:Real}, num_samples::Int, 
    f_obj::Function, σ::Real)
    X = zeros(num_samples, 3)
    seen = Set{Tuple{<:Real, <:Real}}()
    d = size(X_star, 1)
    for i in 1 : num_samples
        randVal = rand(1:d)
        samp = tuple(X[randVal, :]...)
        while samp in seen
            randVal = rand(1 : rows)
            samp = tuple(X[randVal, :]...)
        end
        push!(seen, samp)
        samp_pt = f_obj(samp[1], samp[2])
        pt = [samp[1], samp[2], samp_pt + σ]
        X[i, :] = pt
    end
    return X
end

function best_sampling_point(acq_funq::AbstractVector{<:Real}, X_star::AbstractVector{<:Real},
    X::AbstractVector{<:Real}, f_obj::Function, σ::Real)
    samp_pt = findmax(acq_func)
    xy = X_star[(samp_pt[2])[1], :]
    z = f_obj(xy[1], xy[2])
    val_vec = [xy[1], xy[2], z + σ]
    X = vcat(X, val_vec')
    println(z + σ)
    return X
end