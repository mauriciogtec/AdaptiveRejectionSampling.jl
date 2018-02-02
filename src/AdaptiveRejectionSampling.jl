module AdaptiveRejectionSampling
using ForwardDiff
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
    x = - (l2.intercept - l1.intercept) / (l2.slope - l1.slope)
    # y = l1.slope * x + l1.intercept
    x
end

"""
Ordered Partition
"""
mutable struct OrderedPartition
    lines::Vector{Line}
    cutpoints::Float64
    size::Int32
    weights::Float64
    OrderedPartition(lines::Vector{Line}) = begin
        @assert issorted([l.slope for l in lines], rev = true) "line slopes must be decreasing"
        cutpoints = [intersection(lines[i], lines[i + 1]) for i in 1:(length(lines) - 1)]
        @assert issorted(cutpoints) "resultin cutpoints aren't ordered"
        @assert length(unique(cutpoints)) == length(cutpoints) "cutpoints can't contain duplicates"
        weights = [exp(l.slope[i])]
        new(lines, cutpoints, length(lines), weights)
    end
end


"""
Adds a new line segment to an ordered partition based on its slope
"""
function add_line_segment!(p::OrderedPartition, l::Line)
    # Find the position
    pos = 1
    while l.slope < p.lines[pos].slope || pos > length(p.lines)
        pos += 1
    end

    # Insert line segment
    if pos == 1 || pos == length(p.lines) + 1
        cut = intersection(l, p.lines[pos])
        insert!(p.cutpoints, pos, cut)
    else
        cut1 = intersection(l, p.lines[pos - 1])
        cut2 = intersection(l, p.lines[pos])
        splice!(p.cutpoints, pos - 1, [cut1, cut2])
        @assert issorted(p.cutpoints)  "resulting intersection points aren't sorted"
    end
    insert!(p.lines, pos, l)
end


# --------------------------------
export Line, OrderedPartition
export intersection, add_line_segment!
# --------------------------------
# """
# A Majorizer most contain information about the problem
# """
# mutable struct Envelop
#     line_segments::Array{LineSegment, 1}
#     logfun::Function
#     loggrad::Function
#     loghess::Function
# end
#
# """
# Eval gradient of Envelop
# """
# function initialize_envelop(fun::Function, x1::AbstractFloat, x2::AbstractFloat)
#     @assert x1 < x2
#     logfun(x) = log(fun(x))
#     loggrad(x) = ForwardDiff.derivative(logfun, x)
#     @assert loggrad(x1) > 0
#     @assert loggrad(x2) < 0
#     loghess(x) = ForwardDiff.derivative(loggrad, x)
#     m1, m2 = loggrad(x1), loggrad(x2)
#     b1, b2 = logfun(x2), logfun(x2)
#     l1, l2 = LineSegment(m1, b1, -Inf, Inf), LineSegment(m2, b2, -Inf, Inf)
#     intersectx, intersecty = intersection(l1, l2)
#     l1.upper = intersectx
#     l2.lower = intersectx
#     return Envelop([l1, l2], logfun, loggrad, loghess)
# end
#
# # ------------------------------------
# export Envelop, initialize_envelop
end # module
