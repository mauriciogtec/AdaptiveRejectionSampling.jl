
"""
A log conconcave function is majorized with a piecewise envelop, which on the original scale is piecewise
exponential. As the resulting extremely precise envelop adapts, the rejection rate dramatically decreases.
"""
module AdaptiveRejectionSampling
# ------------------------------
using Random # Random stdlib
# ------------------------------
import ForwardDiff: derivative 
import StatsBase: sample, weights
# ------------------------------
export Line, Objective, Envelop, RejectionSampler # Structures/classes
export run_sampler!, sample_envelop, eval_envelop, add_segment! # Methods
# ------------------------------

"""
    Line(slope::Float64, intercept::Float64)
Basic ensamble-unit for an envelop.
"""
mutable struct Line
    slope::Float64
    intercept::Float64
end

"""
    intersection(l1::Line, l2::Line)
Finds the horizontal coordinate of the intersection between lines
"""
function intersection(l1::Line, l2::Line)
    @assert l1.slope != l2.slope "slopes should be different"
    -(l2.intercept - l1.intercept) / (l2.slope - l1.slope)
end

"""
    exp_integral(l::Line, x1::Float64, x2::Float64)
Computes the integral
    ``LaTeX \\int_{x_1} ^ {x_2} \\exp\\{ax + b\\} dx. ``
The resulting value is the weight assigned to the segment [x1, x2] in the envelop
"""
@inline function exp_integral(l::Line, x1::Float64, x2::Float64)
    a, b = l.slope, l.intercept
    v1, v2 = a*x1, a*x2
    if v1 > 25.0 || v2 > 25.0 || a == 0.0 || b > 25.0
        @warn "exp_integral: numerical instability, truncating, check for under/overflow, consider truncating logf"
        v1 = min(v1, 25.0)
        v2 = min(v2, 25.0)
        b = min(b, 25.0)
        if a == 0
            a = (v2 - v1) * 1e-6
        end
    end
    exp(b) * (exp(v2) - exp(v1)) / a
end

"""
    Envelop(lines::Vector{Line}, support::Tuple{Float64, Float64})
A piecewise linear function with k segments defined by the lines `L_1, ..., L_k` and cutpoints
`c_1, ..., c_k+1` with `c1 = support[1]` and `c2 = support[2]`. A line L_k is active in the segment
[c_k, c_k+1], and it's assigned a weight w_k based on [exp_integral](@exp_integral). The weighted integral
over c_1 to c_k+1 is one, so that the envelop is interpreted as a density.
"""
mutable struct Envelop
    lines::Vector{Line}
    cutpoints::AbstractVector{Float64}
    weights::AbstractVector{Float64}
    size::Int

    Envelop(lines::Vector{Line}, support::Tuple{Float64,Float64}) = begin
        @assert issorted([l.slope for l in lines], rev=true) "line slopes must be decreasing"
        intersections = [intersection(lines[i], lines[i+1]) for i in 1:(length(lines)-1)]
        cutpoints = [support[1]; intersections; support[2]]
        @assert issorted(cutpoints) "cutpoints must be ordered"
        @assert length(unique(cutpoints)) == length(cutpoints) "cutpoints can't have duplicates"
        weights = [exp_integral(l, cutpoints[i], cutpoints[i+1]) for (i, l) in enumerate(lines)]
        @assert Inf ∉ weights "Overflow in assigning weights"
        new(lines, cutpoints, weights, length(lines))
    end
end


"""
    add_segment!(e::Envelop, l::Line)
Adds a new line segment to an envelop based on the value of its slope (slopes must be decreasing
always in the envelop). The cutpoints are automatically determined by intersecting the line with
the adjacent lines.
"""
function add_segment!(e::Envelop, l::Line)
    # Find the position in sorted array with binary search
    pos = searchsortedfirst([-line.slope for line in e.lines], -l.slope)
    # Find the new cutpoints
    if pos == 1
        new_cut = intersection(l, e.lines[pos])
        # Insert in second position, first one is the support bound
        insert!(e.cutpoints, pos + 1, new_cut)
    elseif pos == e.size + 1
        new_cut = intersection(l, e.lines[pos-1])
        insert!(e.cutpoints, pos, new_cut)
    else
        new_cut1 = intersection(l, e.lines[pos-1])
        new_cut2 = intersection(l, e.lines[pos])
        splice!(e.cutpoints, pos, [new_cut1, new_cut2])
        @assert issorted(e.cutpoints) "incompatible line: resulting intersection points aren't sorted"
    end
    # Insert the new line
    insert!(e.lines, pos, l)
    e.size += 1
    # Recompute weights (this could be done more efficiently in the future by updating the neccesary ones only)
    e.weights = [exp_integral(line, e.cutpoints[i], e.cutpoints[i+1]) for (i, line) in enumerate(e.lines)]
end

"""
    sample_envelop(rng::AbstractRNG, e::Envelop)
    sample_envelop(e::Envelop)
Samples an element from the density defined by the envelop `e` with it's exponential weights.
See [`Envelop`](@Envelop) for details.
"""
function sample_envelop(rng::AbstractRNG, e::Envelop)
    # Randomly select lines based on envelop weights
    i = sample(rng, 1:e.size, weights(e.weights))
    a, b = e.lines[i].slope, e.lines[i].intercept
    # Use the inverse CDF method for sampling
    log(exp(-b) * rand(rng) * e.weights[i] * a + exp(a * e.cutpoints[i])) / a
end

function sample_envelop(e::Envelop)
    sample_envelop(Random.GLOBAL_RNG, e)
end


"""
    eval_envelop(e::Envelop, x::Float64)
Eval point a point `x` in the piecewise linear function defined by `e`. Necessary for evaluating
the density assigned to the point `x`.
"""
function eval_envelop(e::Envelop, x::Float64)
    # searchsortedfirst is the proper method for and ordered list
    pos = searchsortedfirst(e.cutpoints, x)
    if pos == 1 || pos == length(e.cutpoints) + 1
        return 0.0
    else
        a, b = e.lines[pos-1].slope, e.lines[pos-1].intercept
        return exp(a * x + b)
    end
end

# --------------------------------

"""
    Objective(logf::Function, support:)
    Objective(logf::Function, grad::Function)
Convenient structure to store the objective function to be sampled. It must receive the logarithm of
f and not f directly. It uses automatic differentiation by default, but the user can provide the derivative optionally.
"""
struct Objective
    logf::Function
    grad::Function
    Objective(logf::Function) = begin
        # Automatic differentiation
        grad(x) = derivative(logf, x)
        new(logf, grad)
    end
    Objective(logf::Function, grad::Function) = new(logf, grad)
end

"""
RejectionSampler(f::Function, support::Tuple{Float64, Float64}, init::Tuple{Float64, Float64})
    RejectionSampler(f::Function, support::Tuple{Float64, Float64}[ ,δ::Float64])
An adaptive rejection sampler to obtain iid samples from a logconcave function supported in 
`support = (support[1], support[2])`. f can either be the either probability density of the 
function to be sampled, or its logarithm. For the latter, use the keyword argument `log=true`. 
The functions can be unnormalized in the sense that the probability density can be specified up to a constant. 
The adaptive rejection samplings algorithm requires two initial points `init = init[1], init[2]`
such that (d/dx)logp(init[1]) > 0 and (d/dx)logp(init[2]) < 0. These points can be provided directly
(typically, any point left and right of the mode will do). It is also possibe to specify and search
range and delta for a greedy search of the initial points. 
The `support` must be of the form `(-Inf, Inf), (-Inf, a), (b, Inf), (a,b)`, and it represent the
interval in which f has positive value, and zero elsewhere.

The alternative constructor uses a search_range, a value δ for the distance between points in the search,
and min/max slope values in absolute terms.

## Keyword arguments
- `max_segments::Int = 10` : max size of envelop, the rejection-rate is usually slow with a small number of segments
- `max_failed_factor::Float64 = 0.001`: level at which throw an error if one single sample has a rejection rate
    exceeding this value
- `logdensity::Bool = false`: indicator fo whether `f` is the log of the probability density, up to a normalization constant.
- `search_range::Tuple{Float64,Float64} = (-10.0, 10.0)`: range in which to search for initial points
- `min_slope::Float64 = 1e-6`: minimum slope in absolute value of logf at the initial/found points
- `max_slope::Float64 = Inf: maximum slope in absolute value of logf at the initial/found points
"""
struct RejectionSampler
    objective::Objective
    envelop::Envelop
    max_segments::Int
    max_failed_rate::Float64
    # Constructor when initial points are provided
    RejectionSampler(
        f::Function,
        support::Tuple{Float64,Float64},
        init::Tuple{Float64,Float64};
        max_segments::Int=25,
        logdensity::Bool=false,
        max_failed_rate::Float64=0.001,
    ) = begin
        @assert support[1] < support[2] "invalid support, not an interval"
        if logdensity
            objective = Objective(f)
        else
            objective = Objective(x -> log(f(x)))
        end
        x1, x2 = init
        @assert x1 < x2 "cutpoints must be ordered"
        a1, a2 = objective.grad(x1), objective.grad(x2)
        @assert 0.0 < a1 "logf must have positive slope at initial cutpoint 1"
        @assert a2 < 0.0 "logf must have negative slope at initial cutpoint 2"
        b1, b2 = objective.logf(x1) - a1 * x1, objective.logf(x2) - a2 * x2
        line1, line2 = Line(a1, b1), Line(a2, b2)
        envelop = Envelop([line1, line2], support)
        new(objective, envelop, max_segments, max_failed_rate)
    end
""
    RejectionSampler(
        f::Function,
        support::Tuple{Float64,Float64},
        δ::Float64=0.5;
        search_range::Tuple{Float64,Float64}=(-10.0, 10.0),
        min_slope::Float64=1e-6,
        max_slope::Float64=10.0,
        logdensity::Bool=false,
        kwargs...
    ) = begin
        if logdensity
            logf = f
        else
            logf = (x -> log(f(x)))
        end
        grad(x) = derivative(logf, x)
        grid_lims = max(search_range[1], support[1]), min(search_range[2], support[2])
        grid = grid_lims[1]:δ:grid_lims[2]
        grads = grad.(grid)
        i1 = findfirst(min_slope .< grads .< max_slope)
        i2 = findlast(min_slope .< - grads .< max_slope)
        @assert (i1 !== nothing) && (i2 !== nothing) "couldn't find initial points, please provide them or change `search_range`"
        @assert i1 < i2 "function is not logconcave, first index with grad>0 must be smaller than first index with grad<0"
        x1, x2 = grid[i1], grid[i2]
        @info "initial points found at $(x1), $(x2) with grads $(grads[i1]), $(grads[i2])"
        RejectionSampler(f, support, (x1, x2); logdensity=logdensity, kwargs...)
    end
end

"""
    run_sampler!(rng::AbstractRNG, sampler::RejectionSampler, n::Int)
    run_sampler!(sampler::RejectionSampler, n::Int)
It draws `n` iid samples of the objective function of `sampler`, and at each iteration it adapts the envelop
of `sampler` by adding new segments to its envelop.
"""
function run_sampler!(rng::AbstractRNG, s::RejectionSampler, n::Int)
    i = 0
    failed, max_failed = 0, trunc(Int, n / s.max_failed_rate)
    out = zeros(n)
    while i < n
        candidate = sample_envelop(rng, s.envelop)
        acceptance_ratio = exp(s.objective.logf(candidate)) / eval_envelop(s.envelop, candidate)
        if rand(rng) < acceptance_ratio
            i += 1
            out[i] = candidate
        else
            if length(s.envelop.lines) <= s.max_segments
                a = s.objective.grad(candidate)
                b = s.objective.logf(candidate) - a * candidate
                add_segment!(s.envelop, Line(a, b))
            end
            failed += 1
            @assert failed < max_failed "max_failed_factor reached"
        end
    end
    out
end

function run_sampler!(s::RejectionSampler, n::Int)
    run_sampler!(Random.GLOBAL_RNG, s, n)
end
end # module AdaptiveRejectionSampling
