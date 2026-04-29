"""
A log conconcave function is majorized with a piecewise envelop, which on the original scale is piecewise
exponential. As the resulting extremely precise envelop adapts, the rejection rate dramatically decreases.
"""
module AdaptiveRejectionSampling

import SpecialFunctions: loggamma
using StatsBase
import Random: default_rng, AbstractRNG
using DifferentiationInterface
import ForwardDiff
using Compat

include("sampling.jl")
@compat public Objective, ARSampler, sample!

include("plot.jl")
@compat public hullplot, hullplot!
end # module AdaptiveRejectionSampling
