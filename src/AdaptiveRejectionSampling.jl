module AdaptiveRejectionSampling
# ------------------------------
using ForwardDiff
using StatsBase
# ------------------------------
export Line, Objective, Envelop, RejectionSampler
export run_sampler!, eval_envelop

# ------------------------------
"""
Line
"""
mutable struct Line
    slope::Float64
    intercept::Float64
end

"""
Find the intersection between lines
"""
function intersection(l1::Line, l2::Line)
    @assert l1.slope != l2.slope "slopes should be different"
     - (l2.intercept - l1.intercept) / (l2.slope - l1.slope)
end

"""
Eval line
"""
function exp_integral(l::Line, x1::Float64, x2::Float64)
    a, b = l.slope, l.intercept
    exp(b) * (exp(a * x2) - exp(a * x1)) / a
end

"""
Ordered Partition
"""
mutable struct Envelop
    lines::Vector{Line}
    cutpoints::Vector{Float64}
    weights::Vector{Float64}
    Envelop(lines::Vector{Line}) = begin
        @assert issorted([l.slope for l in lines], rev = true) "line slopes must be decreasing"
        cutpoints = [intersection(lines[i], lines[i + 1]) for i in 1:(length(lines) - 1)]
        @assert issorted(cutpoints) "resultin cutpoints aren't ordered"
        @assert length(unique(cutpoints)) == length(cutpoints) "cutpoints can't contain duplicates"
        int_lims = [-Inf; cutpoints; Inf]
        weights = [exp_integral(l, int_lims[i], int_lims[i + 1]) for (i, l) in enumerate(lines)]
        @assert Inf ∉ weights "Numerical error integrating: overflow"
        new(lines, cutpoints, weights)
    end
end


"""
Adds a new line segment to an ordered partition based on its slope
"""
function add_line_segment!(p::Envelop, l::Line)
    # Find the position in sorted array with binary search
    pos = searchsortedfirst([-line.slope for line in p.lines], -l.slope)
    # Insert line segment
    if pos == 1
        cut = intersection(l, p.lines[pos])
        insert!(p.cutpoints, pos, cut)
    elseif pos == length(p.lines) + 1
        cut = intersection(l, p.lines[pos - 1])
        insert!(p.cutpoints, pos - 1, cut)
    else
        cut1 = intersection(l, p.lines[pos - 1])
        cut2 = intersection(l, p.lines[pos])
        splice!(p.cutpoints, pos - 1, [cut1, cut2])
        @assert issorted(p.cutpoints)  "resulting intersection points aren't sorted"
    end
    insert!(p.lines, pos, l)
    int_lims = [-Inf; p.cutpoints; Inf]
    p.weights = [exp_integral(line, int_lims[i], int_lims[i + 1]) for (i, line) in enumerate(p.lines)];
    nothing
end

"""
Sample from an envelop
"""
function get_samples(p::Envelop, n::Int64)
    line_num = sample(1:length(p.lines), weights(p.weights), n)
    a = [l.slope for l in p.lines]
    b = [l.intercept for l in p.lines]
    lims = [-Inf; p.cutpoints]
    un = rand(n)
    [log(exp(-b[i]) * u * p.weights[i] * a[i] + exp(a[i] * lims[i])) / a[i] for (i, u) in zip(line_num, un)]
end

"""
Eval point on ordered partition
"""
function eval_envelop(p::Envelop, x::Float64)
    pos = searchsortedfirst(p.cutpoints, x)
    a, b = p.lines[pos].slope, p.lines[pos].intercept
    exp(a * x + b)
end

# --------------------------------

"""
Objective function to sample
"""
struct Objective
    logf::Function
    grad::Function
    Objective(logf::Function) = begin
        grad(x) = ForwardDiff.derivative(logf, x)
        new(logf, grad)
    end
end

"""
Rejection Sampler
"""
mutable struct RejectionSampler
    objective::Objective
    envelop::Envelop
    RejectionSampler(f::Function, x1::Float64, x2::Float64)  = begin #x1 and x2 should be auto
        logf(x) = log(f(x))
        objective = Objective(logf)
        a1, a2 = objective.grad(x1), objective.grad(x2)
        @assert a1 >= 0 "logf must have positive slope at x1"
        @assert a2 <= 0 "logf must have negative slope at x2"
        b1, b2 = objective.logf(x1) - a1 * x1, objective.logf(x2) - a2 * x2
        line1, line2 = Line(a1, b1), Line(a2, b2)
        envelop = Envelop([line1, line2])
        new(objective, envelop)
    end
    RejectionSampler(f::Function; δ = 0.5, max_attempts = 100)  = begin #x1 and x2 should be auto
        logf(x) = log(f(x))
        grad(x) = ForwardDiff.derivative(logf, x)
        x1, x2 = -δ, δ
        i, j = 0, 0
        while grad(x1) <= 0 && i < max_attempts
            x1 -= δ
            i += 1
        end
        while grad(x2) >= 0 && j < max_attempts
            x2 += δ
            j += 1
        end
        @assert i != max_attempts && j != max_attempts "couldn't find initial points, please provide them or verify that f is logconcave"
        RejectionSampler(f, x1, x2)
    end
end

function run_sampler!(sampler::RejectionSampler, n::Int64; max_segments::Int64 = 25, max_failed_factor::Float64 = 0.01)
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
