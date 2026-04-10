const DEFAULT_MAX_SEGMENTS::Int = 25


"""
    AllocFreeWeights{S <: Real, T <: Number, V <: AbstractVector{T}} <: AbstractWeights{S, T, V}

Non-mutable weights used in the `sample` function in this package in order to avoid allocations.
"""
struct AllocFreeWeights{S <: Real, T <: Number, V <: AbstractVector{T}} <: AbstractWeights{S, T, V}
    values::V
    sum::S

    function AllocFreeWeights(ws::V, sum::S) where {S <: Real, T <: Number, V <: AbstractVector{T}}
        return new{S, T, V}(ws, sum)
    end
end

function AllocFreeWeights(ws::V) where {T <: Number, V <: AbstractVector{T}}
    return AllocFreeWeights(ws, sum(ws))
end


struct Objective{T <: Number, F <: Function, G <: Function}
    "The (log-concave) function defining the density `f(x) -> y`"
    f::F
    "Its gradient `grad(x) -> y'`"
    grad::G

    @doc """
        Objective(f::Function, grad::Function, ::Type{T} = Float64) where {T <: Number}

    Create an `Objective` directly defined by its function `f` and custom gradient `grad`.
    The parameter `T` represents the type of the expected input and is utilized when preparing gradients using autodiff.
    By default `T = Float64`.


    !!! warning
        Observe that `f` should be in its
        log-concave form and that no checks are performed in order to verify this.
    """
    function Objective(f::Function, grad::Function, ::Type{T} = Float64) where {T <: Number}
        return new{T, typeof(f), typeof(grad)}(f, grad)
    end
end

"""
    Objective(f::Function, init = one(Float64); adbackend = AutoForwardDiff())

Create an `Objective` for a function automatically generating its gradient.
Gradients are calculated using `DifferentiationInterface.jl` using the backend of choice
with `ForwardDiff.jl` being the default. In order to prepare the gradient an initial value is
required. By default this is `one(Float64)`. If a gradient for a different type is
desired, it should be specified through this initial value.

If a custom/manual gradient is available it may instead be provided.

!!! warning
    Observe that `f` should be in its
    log-concave form and that no checks are performed in order to verify this.

# Example

```julia
# Create an objective using `Mooncake.jl` autodiff and Float32 as its type.
ARS.Objective(somefun, one(Float32); adbackend = AutoMooncake(; config=nothing))
# Create an objective using the defaults `AutoDiff.jl` autodiff and Float64 as its type.
ARS.Objective(somefun)
```
"""
function Objective(f::Function, init = one(Float64); adbackend = AutoForwardDiff())
    gradprep = prepare_gradient(f, adbackend, init)
    obj = Objective(
        f,
        x -> gradient(f, gradprep, adbackend, x),
        typeof(init)
    )
    return obj
end

abstract type AbstractHull end

"""
    struct UpperHull{T} <: AbstractHull
           intercepts::Vector{T}
           slopes::Vector{T}
           intersections::Vector{T}
           abscissae::Vector{T}
           segment_weights::Vector{T}
           domain::Tuple{T, T}
    end

Upper hull of the enveloping function.
"""
struct UpperHull{T} <: AbstractHull
    intercepts::Vector{T}
    slopes::Vector{T}
    intersections::Vector{T}
    abscissae::Vector{T}
    segment_weights::Vector{T}
    weights_cumsum::Vector{T}
    domain::Tuple{T, T}
end

Base.broadcastable(h::UpperHull) = Ref(h)

slopes(h::UpperHull) = h.slopes
intercepts(h::UpperHull) = h.intercepts
intersections(h::UpperHull) = h.intersections
abscissae(h::UpperHull) = h.abscissae
line(h::UpperHull, i::Integer) = (h.slopes[i], h.intercepts[i])
lineinds(h::UpperHull) = 1:(length(slopes(h)))
n_lines(h::UpperHull) = length(slopes(h))
segment_weights(h::UpperHull) = h.segment_weights
weights_cumsum(h::UpperHull) = h.weights_cumsum

"""
    eval_hull(h::UpperHull, x)

Eval hull `h` at `x`.
"""
function eval_hull(h::UpperHull{T}, x::T) where {T}
    ints = intersections(h)
    i = searchsortedfirst(ints, x) - 1
    i = i == length(ints) + 1 ? i - 1 : i
    i = iszero(i) ? i + 1 : i
    sl, int = line(h, i)
    return sl * x + int
end


"""
    intersection(slope1::T, intercept1::T, slope2::T, intercept2::T) where {T}

Returns the intersection abscissa between 2 lines as defined by their slopes and intercepts. Returns NaN if the lines are paralell.
"""
function intersection(slope1::T, intercept1::T, slope2::T, intercept2::T) where {T}
    return (intercept2 - intercept1) / (slope1 - slope2)
end

"""
    calc_intersects!(out::AbstractVector{T}, slopes::AbstractVector{T}, intercepts::AbstractVector{T}) where {T}

Calculate intersections of a series of lines defined by their `slopes` and `intercepts`, storing the result in `out`.
"""
function calc_intersects!(out::AbstractVector{T}, slopes::AbstractVector{T}, intercepts::AbstractVector{T}) where {T}
    for i in eachindex(out)
        out[i] = intersection(slopes[i], intercepts[i], slopes[i + 1], intercepts[i + 1])
    end
    return nothing
end

"""
    calc_intersects(slopes::AbstractVector{T}, intercepts::AbstractVector{T}) where {T}

Calculate intersections of a series of lines defined by their `slopes` and `intercepts`.
"""
function calc_intersects(slopes::AbstractVector{T}, intercepts::AbstractVector{T}) where {T}
    # return [
    #     intersection(slopes[i], intercepts[i], slopes[i + 1], intercepts[i + 1])
    #         for i in 1:(length(slopes) - 1)
    # ]

    out = Vector{T}(undef, length(slopes) - 1)
    calc_intersects!(out, slopes, intercepts)
    return out
end


"""
    calc_slopes_and_intercepts(obj::Objective, abscissae::AbstractVector{T}) where {T}

Calculates slopes and intercepts of the lines tangential to the objective function at `abscissae`.
"""
function calc_slopes_and_intercepts(obj::Objective, abscissae::AbstractVector{T}) where {T}
    issorted(abscissae) ||
        throw(ArgumentError("`abscissae` should be sorted in ascending order."))
    slopes = [obj.grad(abscissae[i]) for i in eachindex(abscissae)]
    intercepts = [
        slopes[i] * (-abscissae[i]) + obj.f(abscissae[i])
            for i in eachindex(abscissae)
    ]
    return slopes, intercepts
end


"""
    exp_integral_line(slope, intercept, x1, x2)

Calculate the integral of `exp(slope * x + intercept)` between `x1` and `x2`.
"""
function exp_integral_line(slope, intercept, x1, x2)
    return if !iszero(slope)
        (exp(slope * x2 + intercept) - exp(slope * x1 + intercept)) / slope
    else
        (x2 - x1) * exp(intercept)
    end
end

"""
    UpperHull(obj::Objective, abscissae::Vector{T}, domain::Tuple{T, T}) where {T}

Create an `UpperHull` based on the objective function, initial abscissae and its domain.
"""
function UpperHull(obj::Objective{T}, abscissae::Vector{T}, domain::Tuple{T, T}) where {T}
    slopes, intercepts = calc_slopes_and_intercepts(obj, abscissae)


    intersects = [domain[1]; calc_intersects(slopes, intercepts); domain[2]]

    wgts = Vector{T}(undef, length(slopes))
    for i in 1:length(wgts)
        wgts[i] = exp_integral_line(slopes[i], intercepts[i], intersects[i], intersects[i + 1])
    end
    wgts_cumsum = cumsum(wgts)
    # Sizehinting
    sizehint!(slopes, 50)
    sizehint!(intercepts, 50)
    sizehint!(intersects, 50)
    sizehint!(wgts, 50)
    sizehint!(wgts_cumsum, 50)
    sizehint!(abscissae, 50)
    return UpperHull(intercepts, slopes, intersects, abscissae, wgts, wgts_cumsum, domain)
end

# Calculate the inverse CDF of a segment, handling a slope of zero.
"""
    inv_cdf_seg(slope, intercept, r, w, intersect, intersect2)

Calculate the inverse cumulative density of a hull segment at `r` based on its...

- `slope`
- `intercept`
- `w` (weight)
- `intersect`, `intersect2`, its intersections with previous/next segments
"""
@inline function inv_cdf_seg(slope, intercept, r, w, intersect, intersect2)
    return if !iszero(slope)
        log(exp(-intercept) * r * w * slope + exp(slope * intersect)) / slope
    else
        r * (intersect2 - intersect) + intersect
    end
end

# Draw a single sample from `h`.
function sample_hull(rng::AbstractRNG, h::UpperHull{T}) where {T}
    s = weights_cumsum(h)
    u = rand(rng, T) * s[end]
    k = length(s)

    ind = 1
    while u > s[ind] && ind < k
        ind += 1
    end

    sl, int = line(h, ind)
    return inv_cdf_seg(sl, int, rand(rng), segment_weights(h)[ind], intersections(h)[ind], intersections(h)[ind + 1])
end
sample_hull(h::UpperHull) = sample_hull(default_rng(), h)


# Draw `n` samples from `h`, storing the result in `out`.
function sample_hull!(rng::AbstractRNG, out::AbstractVector{T}, h::UpperHull{T}, n::Integer) where {T}
    seg_wgts = segment_weights(h)
    inds = sample(rng, 1:n_lines(h), AllocFreeWeights(seg_wgts, sum(seg_wgts)), n)
    rands = rand(rng, n)
    for i in eachindex(inds, out, rands)
        ind = inds[i]
        sl, int = line(h, ind)
        out[i] = inv_cdf_seg(sl, int, rands[i], segment_weights(h)[ind], intersections(h)[ind], intersections(h)[ind + 1])
    end
    return
end
sample_hull!(out::AbstractVector{T}, h::UpperHull{T}, n::Integer) where {T} = sample_hull!(default_rng(), out, h, n)

function sample_hull!(rng, out::AbstractVector{T}, h::UpperHull) where {T}
    return sample_hull!(rng, out, h, length(out))
end
sample_hull!(out::AbstractVector{T}, h::UpperHull) where {T} = sample_hull!(default_rng(), out, h)

# Draw `n` samples from `h`.
function sample_hull(rng::AbstractRNG, h::UpperHull{T}, n::Integer) where {T}
    out = Vector{T}(undef, n)
    sample_hull!(rng, out, h, n)
    return out
end
sample_hull(h::UpperHull, n::Integer) = sample_hull(default_rng(), h, n)

struct LowerHull{T}
    intercepts::Vector{T}
    slopes::Vector{T}
    intersections::Vector{T}
end


Base.broadcastable(h::LowerHull) = Ref(h)


intercepts(h::LowerHull) = h.intercepts
slopes(h::LowerHull) = h.slopes
intersections(h::LowerHull) = h.intersections
line(h::LowerHull, i) = (h.slopes[i], h.intercepts[i])

function calc_lower_slopes_and_intercepts!(
        slout::AbstractVector{T}, intout::AbstractVector{T},
        intersections::AbstractVector{T}, f::Function
    ) where {T}

    for i in eachindex(slout, intout)
        slout[i] = (f(intersections[i + 1]) - f(intersections[i])) /
            (intersections[i + 1] - intersections[i])
        intout[i] = slout[i] * (-intersections[i]) + f(intersections[i])
    end
    return nothing
end


function LowerHull(upper::UpperHull{T}, obj::Objective{T}) where {T}
    intersections = abscissae(upper) # NOTE: This makes them alias each other!
    n_segs = length(intersections) - 1
    sl = Vector{T}(undef, n_segs)
    int = Vector{T}(undef, n_segs)

    calc_lower_slopes_and_intercepts!(sl, int, intersections, obj.f)

    # Sizehinting
    sizehint!(sl, 50)
    sizehint!(int, 50)

    return LowerHull(int, sl, intersections)
end

function eval_hull(h::LowerHull, x)
    i = searchsortedfirst(intersections(h), x)
    if isone(i) || i == lastindex(intersections(h)) + 1
        return -Inf
    end
    sl, int = line(h, i - 1)
    return sl * x + int
end


"""
    ARSampler{T, F, G}

Adaptive rejection sampler containing the objective function, its gradient and the
upper/lower hull of the piecewise linear envelope.
"""
struct ARSampler{T, F, G}
    objective::Objective{T, F, G}
    upper_hull::UpperHull{T}
    lower_hull::LowerHull{T}
end

"""
    ARSampler(
        obj::Objective{T, F, G},
        domain::Tuple{T, T},
        initial_points::Vector{T}
    ) where {T <: AbstractFloat, F <: Function, G <: Function}

    ARSampler(
        obj::Objective{T, F, G},
        domain::Tuple{T, T},
        search_range::Tuple{T, T} = (-10.0, 10.0),
        search_step = 0.1
    ) where {T <: AbstractFloat, F <: Function, G <: Function}

Initialize an adaptive rejection sampler over an objective function from `obj` ([`AdaptiveRejectionSampling.Objective`](@ref)).
`initial_points` should be a vector defining the abscissae of the initial segments of the sampler.
 At least 2 of the points should be on opposite sides of the objective function's maximum.

If no `initial_points` an attempt to find suitable ones will be made by searching for
the first negative/positive slope of `obj`. The default search range is `-10:10` with a
step of `0.1`.

!!! warning
    As it currently stands, finding inital points will fail for distributions bounded
    to the left/right of their maximum. In these cases the initial point(s) need to
    be provided manually (however, only a single initial point has to be provided).
"""
function ARSampler end

function ARSampler(
        obj::Objective{T, F, G},
        domain::Tuple{T, T},
        initial_points::Vector{T}
    ) where {T <: AbstractFloat, F <: Function, G <: Function}

    u = UpperHull(obj, initial_points, domain)
    l = LowerHull(u, obj)

    return ARSampler{T, F, G}(obj, u, l)
end


function ARSampler(obj::Objective{T, F, G}, domain::Tuple{<:Number, <:Number}, search_range::Tuple{<:Number, <:Number} = (-10.0, 10.0); search_step = 0.5) where {T <: AbstractFloat, F <: Function, G <: Function}
    domain_T = T.(domain)
    search_range_T = T.(search_range)
    lbs = T(max(domain_T[1], search_range_T[1]))
    ubs = T(min(domain_T[2], search_range_T[2]))
    search_step_T = T(search_step)
    initial_points = find_initial_points(obj, lbs, ubs, search_step_T)

    return ARSampler(obj, domain_T, initial_points)
end

support(s::ARSampler) = s.upper_hull.domain
abscissae(s::ARSampler) = abscissae(s.upper_hull)
slopes(s::ARSampler) = slopes(s.upper_hull)
intercepts(s::ARSampler) = intercepts(s.upper_hull)

"""
    find_initial_points(obj, lbs, ubs, search_step)

Attempts to find suitable initial points for abscissae based on the objective functions by looking for
the first points with positive and negative slopes respectively. Simply searches between the bounds specified
by `lbs` and `ubs` with step size `search_step`.
"""
function find_initial_points(obj, lbs, ubs, search_step)
    grad = obj.grad
    r = lbs:search_step:ubs
    g = (grad(x) for x in r)
    l = findfirst(>(0), g)
    u = findfirst(<(0), g)
    isnothing(l) && throw(
        ErrorException(
            lazy"""
            Could not find the left initial segment.
            Search range: $(lbs) -- $(ubs)
            Search step: $(search_step)
            """
        )
    )
    isnothing(u) && throw(
        ErrorException(
            lazy"""
            Could not find the right initial segment.
            Search range: $(lbs) -- $(ubs)
            Search step: $(search_step)
            """
        )
    )
    return [r[l], r[u]]
end


function n_segments(s::ARSampler)
    return n_lines(s.upper_hull)
end

# Adds a segment with abscissa at `x` to `s`
"""
    add_segment!(s::ARSampler{T}, x::T) where {T <: AbstractFloat}

Modifies the hulls of `s`, adding a segment with abscissa at `x`.
"""
function add_segment!(s::ARSampler{T}, x::T) where {T <: AbstractFloat}

    # Calculate slope, intercept and index of new segment
    new_slope = s.objective.grad(x)
    new_intercept = (new_slope * -x) + s.objective.f(x)
    new_ind = searchsortedfirst(abscissae(s.upper_hull), x)

    # Insert new slope and intercept
    all_intersections = intersections(s.upper_hull)
    all_slopes = slopes(s.upper_hull)
    all_intercepts = intercepts(s.upper_hull)
    insert!(all_slopes, new_ind, new_slope)
    # simple_insert!(all_slopes, new_ind, new_slope)
    insert!(all_intercepts, new_ind, new_intercept)
    # simple_insert!(all_intercepts, new_ind, new_intercept)

    # Insert new abscissa
    insert!(abscissae(s.upper_hull), new_ind, x)
    # simple_insert!(abscissae(s.upper_hull), new_ind, x)

    # TODO: Only calculate the intersections and weights that actually change.
    # Recalculate intersection points for segments, given the new segment
    push!(all_intersections, all_intersections[end]) # Extend intercepts by one
    calc_intersects!(@view(all_intersections[(begin + 1):(end - 1)]), all_slopes, all_intercepts)

    # Recalculate segment weights
    all_weights = segment_weights(s.upper_hull)
    push!(all_weights, zero(T))

    for i in eachindex(all_weights)
        all_weights[i] = exp_integral_line(all_slopes[i], all_intercepts[i], all_intersections[i], all_intersections[i + 1])
    end

    # Recalculate cumsum of weights
    w_cumsum = weights_cumsum(s.upper_hull)
    push!(w_cumsum, zero(T))
    cumsum!(w_cumsum, all_weights)


    #= Lower hull time =#
    # We don't need to add anything to the lower intersections as it is the same vector used for abscissae
    # in the upper hull

    all_inters_lower = intersections(s.lower_hull)
    lowerslopes = slopes(s.lower_hull)
    lowerintercepts = intercepts(s.lower_hull)
    push!(lowerslopes, zero(T))
    push!(lowerintercepts, zero(T))
    calc_lower_slopes_and_intercepts!(
        lowerslopes,
        lowerintercepts,
        all_inters_lower,
        s.objective.f
    )

    return nothing
end

function __sample_single!(rng::AbstractRNG, s::ARSampler{T}, add_segments::Bool, max_segments::Integer) where {T <: AbstractFloat}
end

# TODO: Make this prepopulate `out` using `sample_hull!`?
# NOTE: The above might be inefficient initially if most of the samples get rejected anyway?
"""
    __sample!(rng::AbstractRNG, out::AbstractVector{T}, s::ARSampler{T}, add_segments::Bool, max_segments) where {T<:AbstractFloat}


All other `__sample!` methods call this one.
"""
function __sample!(rng::AbstractRNG, out::AbstractVector{T}, s::ARSampler{T}, add_segments::Bool, max_segments) where {T <: AbstractFloat}
    n = length(out)
    n_accepted = 0
    while n_accepted < n
        x = sample_hull(rng, s.upper_hull)
        up = eval_hull(s.upper_hull, x)
        lo = eval_hull(s.lower_hull, x)
        w = rand(rng)
        # Squeeze test
        if w <= exp(lo - up)
            n_accepted += 1
            out[n_accepted] = x
        elseif w <= exp(s.objective.f(x) - up)
            # Accept sample i
            n_accepted += 1
            out[n_accepted] = x
            if add_segments && n_segments(s) < max_segments
                add_segment!(s, x)
            end
        elseif add_segments && n_segments(s) < max_segments
            add_segment!(s, x)
        end
    end
    return nothing
end
__sample!(out::Vector{T}, s::ARSampler{T}, add_segments::Bool, max_segments) where {T} = __sample!(default_rng(), out, s, add_segments, max_segments)

# Draw `n` samples, returning a newly allocated vector.
function __sample!(rng::AbstractRNG, s::ARSampler{T}, n::Integer, add_segments::Bool, max_segments) where {T <: AbstractFloat}
    out = Vector{T}(undef, n)
    __sample!(rng, out, s, add_segments, max_segments)
    return out
end
__sample!(s::ARSampler{T}, n::Integer, add_segments::Bool, max_segments) where {T <: AbstractFloat} = __sample!(default_rng(), s, n, add_segments, max_segments)


"""
    sample!([rng=default_rng()], s::ARSampler, n::Integer; add_segments::Bool=true, max_segments::Integer = DEFAULT_MAX_SEGMENTS)
    sample!([rng=default_rng()], v::AbstractVector, s::ARSampler; add_segments::Bool=true, max_segments::Integer = DEFAULT_MAX_SEGMENTS)

Draw samples from `s`. If supplied, a vector `v` will be filled with samples.
Otherwise the number of samples is specified with `n`.
"""
function sample! end

function sample!(rng::AbstractRNG, s::ARSampler{T}, n::Integer; add_segments::Bool = true, max_segments = DEFAULT_MAX_SEGMENTS) where {T <: AbstractFloat}
    return __sample!(rng, s, n, add_segments, max_segments)
end

function sample!(s::ARSampler{T}, n::Integer; add_segments::Bool = true, max_segments = DEFAULT_MAX_SEGMENTS) where {T <: AbstractFloat}
    sample!(default_rng(), s, n, add_segments = add_segments, max_segments = max_segments)
end

function sample!(rng::AbstractRNG, v::AbstractVector{T}, s::ARSampler{T}; add_segments::Bool = true, max_segments = DEFAULT_MAX_SEGMENTS) where {T}
    __sample!(rng, v, s, add_segments, max_segments)
    return nothing
end

function sample!(v::AbstractVector{T}, s::ARSampler{T}; add_segments::Bool = true, max_segments = DEFAULT_MAX_SEGMENTS) where {T}
    sample!(default_rng(), v, s, add_segments = add_segments, max_segments = max_segments)
end
