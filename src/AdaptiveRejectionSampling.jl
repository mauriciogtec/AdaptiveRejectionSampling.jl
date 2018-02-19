
"""
A log conconcave function is majorized with a piecewise envelop, which on the original scale is piecewise exponential. As the resulting extremely precise envelop adapts, the rejection rate dramatically decreases.
"""
module AdaptiveRejectionSampling
# ------------------------------
using ForwardDiff # For automatic differentiation, no user nor approximate derivatives
using StatsBase # To include the basic sample from array function
# ------------------------------
export Line, Objective, Envelop, RejectionSampler # Structures/classes
export run_sampler!, eval_envelop # Methods
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
     - (l2.intercept - l1.intercept) / (l2.slope - l1.slope)
end

"""
    exp_integral(l::Line, x1::Float64, x2::Float64)
Computes the integral
    ``LaTeX \int_{x_1} ^ {x_2} \exp\{ax + b\} dx. ``
The resulting value is the weight assigned to the segment [x1, x2] in the envelop
"""
function exp_integral(l::Line, x1::Float64, x2::Float64)
    a, b = l.slope, l.intercept
    exp(b) * (exp(a * x2) - exp(a * x1)) / a
end

"""
    Envelop(lines::Vector{Line}, support::Tuple{Float64, Float64})
A piecewise linear function with k segments defined by the lines `L_1, ..., L_k` and cutpoints `c_1, ..., c_k+1` with `c1 = support[1]` and `c2 = support[2]`. A line L_k is active in the segment [c_k, c_k+1], and it's assigned a weight w_k based on [exp_integral](@ref). The weighted integral over c_1 to c_k+1 is one, so that the envelop is interpreted as a density.
"""
mutable struct Envelop
    lines::Vector{Line}
    cutpoints::Vector{Float64}
    weights::Vector{Float64}
    size::Int

    Envelop(lines::Vector{Line}, support::Tuple{Float64, Float64}) = begin
        @assert issorted([l.slope for l in lines], rev = true) "line slopes must be decreasing"
        intersections = [intersection(lines[i], lines[i + 1]) for i in 1:(length(lines) - 1)]
        cutpoints = [support[1]; intersections; cutpoints[2]]
        @assert issorted(cutpoints) "cutpoints must be ordered"
        @assert length(unique(cutpoints)) == length(cutpoints) "cutpoints can't have duplicates"
        weights = [exp_integral(l, cutpoints[i], cutpoints[i + 1]) for (i, l) in enumerate(lines)]
        @assert Inf ∉ weights "Overflow in assigning weights"
        new(lines, cutpoints, weights, length(lines))
    end
end


"""
    add_line_segment!(e::Envelop, l::Line)
Adds a new line segment to an envelop based on the value of its slope (slopes must be decreasing always in the envelop). The cutpoints are automatically determined by intersecting the line with the adjacent lines.
"""
function add_line_segment!(e::Envelop, l::Line)
    # Find the position in sorted array with binary search
    pos = searchsortedfirst([-line.slope for line in e.lines], -l.slope)
    # Find the new cutpoints
    if pos == 1
        new_cut = intersection(l, e.lines[pos])
        # Insert in second position, first one is the support bound
        insert!(e.cutpoints, pos + 1, new_cut)
    elseif pos == e.size + 1
        new_cut = intersection(l, e.lines[pos - 1])
        insert!(e.cutpoints, pos, new_cut)
    else
        new_cut1 = intersection(l, e.lines[pos - 1])
        new_cut2 = intersection(l, e.lines[pos])
        splice!(e.cutpoints, pos, [cut1, cut2])
        @assert issorted(e.cutpoints)  "incompatible line: resulting intersection points aren't sorted"
    end
    # Insert the new line
    insert!(e.lines, pos, l)
    # Recompute weights (this could be done more efficiently in the future by updating the neccesary ones only)
    e.weights = [exp_integral(line, e.cutpoints[i], e.cutpoints[i + 1]) for (i, line) in enumerate(e.lines)]
end

"""
    sample(p::Envelop, n::Int)
Samples `n` elements iid from the density defined by the envelop `e` with it's exponential weights. See [`Envelop`](@ref) for details.
"""
function sample(e::Envelop, n::Int)
    # Randomly select lines based on envelop weights
    line_num = sample(1:e.size, weights(e.weights), n)
    a = [l.slope for l in e.lines]
    b = [l.intercept for l in e.lines]
    # Generate random uniforms
    u_list = rand(n)
    # Use the inverse CDF method for sampling
    [log(exp(-b[i])*u*e.weights[i]*a[i] + exp(a[i]*e.cutpoints[i]))/a[i] for (i, u) in zip(line_num, u_list)]
end

"""
    eval_envelop(e::Envelop, x::Float64)
Eval point a point `x` in the piecewise linear function defined by `e`. Necessary for evaluating the density assigned to the point `x`.
"""
function eval_envelop(e::Envelop, x::Float64)
    # searchsortedfirst is the proper method for and ordered list
    pos = searchsortedfirst(e.cutpoints, x)
    @assert 1 < pos < length(e.cutpoints) + 2 "x is outside the specified density support"
    a, b = e.lines[pos - 1].slope, e.lines[pos - 1].intercept
    exp(a * x + b)
end

# --------------------------------

"""
    Objective(logf::Function, support:)
    Objective(logf::Function, grad::Function)
Convenient structure to store the objective function to be sampled. It must receive the logarithm of f and not f directly. It uses automatic differentiation by default, but the user can provide the derivative optionally.
"""
struct Objective
    logf::Function
    grad::Function
    Objective(logf::Function) = begin
        # Automatic differentiation
        grad(x) = ForwardDiff.derivative(logf, x)
        new(logf, grad)
    end
    Objective(logf::Function, grad::Function) = new(logf, grad)
end

"""
    RejectionSampler(f::Function, support::Tuple{Float64, Float64}[ ,δ::Float64])
    RejectionSampler(f::Function, support::Tuple{Float64, Float64}, init::Tuple{Float64, Float64})
An adaptive rejection sampler to obtain iid samples from a logconcave function `f`, supported in the domain `support` = (support[1], support[2]). To create the object, two initial points `init = init[1], init[2]` such that `loff'(init[1]) > 0` and `logf'(init[2]) < 0` are necessary. If they are not provided, the constructor will perform a greedy search based on `δ`.

 The argument `support` must be of the form `(-Inf, Inf), (-Inf, a), (b, Inf), (a,b)`, and it represent the interval in which f has positive value, and zero elsewhere.

## Keyword arguments
- `max_segments::Int = 10` : max size of envelop, the rejection-rate is usually slow with a small number of segments
- `max_failed_factor::Float64 = 0.001`: level at which throw an error if one single sample has a rejection rate exceeding this value
"""
mutable struct RejectionSampler
    objective::Objective
    envelop::Envelop

    RejectionSampler(
            f::Function,
            support::Tuple{Float64, Float64},
            init::Tuple{Float64, Float64};
            max_segments::Int = 10,
            max_failed_factor::Float64 = 0.001
    ) = begin
        logf(x) = log(f(x))
        objective = Objective(logf)
        x1, x2 = init
        @assert x1 < x2 "cutpoints must be ordered"
        a1, a2 = objective.grad(x1), objective.grad(x2)
        @assert a1 >= 0 "logf must have positive slope at initial cutpoint 1"
        @assert a2 <= 0 "logf must have negative slope at initial cutpoint 2"
        b1, b2 = objective.logf(x1) - a1 * x1, objective.logf(x2) - a2 * x2
        line1, line2 = Line(a1, b1), Line(a2, b2)
        envelop = Envelop([line1, line2], support)
        new(objective, envelop)
    end

    RejectionSampler(
            f::Function,
            support::Tuple{Float64, Float64},
            δ::Float64 = 0.5;
            max_search_steps::Int = 100,
            kwargs...
    ) = begin
        logf(x) = log(f(x))
        grad(x) = ForwardDiff.derivative(logf, x)
        x1, x2 = -δ, δ
        i, j = 0, 0
        while (grad(x1) <= 0 || grad(x2) >= 0)  && i < max_attempts
            if grad(x1) <= 0
                x1 -= δ
            elsif grad(x2) >= 0
                x2 -= δ
            end
            i += 1
        end
        while (grad(x1) <= 0 || grad(x2) >= 0)  && j < max_attempts
            if grad(x1) <= 0
                x1 += δ
            elsif grad(x2) >= 0
                x2 += δ
            end
            j += 1
        end
        @assert i != max_attempts && j != max_attempts "couldn't find initial points, please provide them or verify that f is logconcave"
        RejectionSampler(f, (x1, x2); kwargs...)
    end
end

"""

"""
function run_sampler!(sampler::RejectionSampler, n::Int)
    i = 0
    failed, max_failed = 0, trunc(Int, n / max_failed_factor)
    out = zeros(n)
    while i < n
        candidate = get_samples(sampler.envelop, 1)[1]
        acceptance_ratio = exp(sampler.objective.logf(candidate)) / eval_envelop(sampler.envelop, candidate)
        if rand() < acceptance_ratio
            i += 1
            out[i] = candidate
        else
            if length(sampler.envelop.lines) <= max_segments
                a = sampler.objective.grad(candidate)
                b = sampler.objective.logf(candidate) - a * candidate
                add_line_segment!(sampler.envelop, Line(a, b))
            end
            failed += 1
            @assert failed < max_failed "max_failed_factor reached"
        end
    end
    out
end

end #
