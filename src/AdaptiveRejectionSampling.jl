module AdaptiveRejectionSampling
using ForwardDiff
# ------------------------------
import Base.indices

"""
A structure for LineSegment
"""
mutable struct LineSegment
    slope::AbstractFloat
    intercept::AbstractFloat
    lower::AbstractFloat
    upper::AbstractFloat
    LineSegment(m, b, lower, upper) =
        (lower < upper) ?
            new(m, b, lower, upper) : throw(ArgumentError("lower !< upper"))
end

"""
Line
"""
mutable struct Line
    slope::AbstractFloat
    intercept::AbstractFloat
end

"""
Ordered Partition
"""
mutable struct OrderedPartition
    lines::Array{Line, 1}
    cuts::Array{AbstractFloat, 1}
    size::Integer
    OrderedPartition(lines::Array{Line, 1}, cuts::Array{AbstractFloat, 1}) =
        issorted(cuts) && length(unique(cuts)) == length(cuts) ?
            new(lines, cuts) : throw(ArgumentError("cutpoints must be sorted"))
end

"""
Find the intersection between lines
"""
function intersection(l1::Union{LineSegment, Line}, l2::Union{LineSegment, Line})
    if l1.slope == l2.slope
        throw(ArgumentError("lines don't intersect, slopes must differ"))
    else
        x = - (l2.intercept - l1.intercept) / (l2.slope - l1.slope)
        y = l1.slope * x + l1.intercept
        return x, y
    end
end

# """
# Add a new line to a LineSegment array
# """
# function add_line_segment!(arr::Array{LineSegment}, newl::LineSegment)
#
# end

# --------------------------------
# export LineSegment, slope, intercept, intersection
export LineSegment, intersection
# --------------------------------
"""
A Majorizer most contain information about the problem
"""
mutable struct Envelop
    line_segments::Array{LineSegment, 1}
    logfun::Function
    loggrad::Function
    loghess::Function
end

"""
Eval gradient of Envelop
"""
function initialize_envelop(fun::Function, x1::AbstractFloat, x2::AbstractFloat)
    @assert x1 < x2
    logfun(x) = log(fun(x))
    loggrad(x) = ForwardDiff.derivative(logfun, x)
    @assert loggrad(x1) > 0
    @assert loggrad(x2) < 0
    loghess(x) = ForwardDiff.derivative(loggrad, x)
    m1, m2 = loggrad(x1), loggrad(x2)
    b1, b2 = logfun(x2), logfun(x2)
    l1, l2 = LineSegment(m1, b1, -Inf, Inf), LineSegment(m2, b2, -Inf, Inf)
    intersectx, intersecty = intersection(l1, l2)
    l1.upper = intersectx
    l2.lower = intersectx
    return Envelop([l1, l2], logfun, loggrad, loghess)
end

# ------------------------------------
export Envelop, initialize_envelop
end # module
