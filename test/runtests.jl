using AdaptiveRejectionSampling: ARSampler, Objective, sample!, AllocFreeWeights
using DifferentiationInterface
using ForwardDiff
using Mooncake
using Enzyme
using SpecialFunctions: loggamma
using Random
using Distributions
using Test
using JET
using Supposition
using Supposition: @check

include("wrappedallocs.jl")

normal(x, mu, sigma) = pdf(Normal(mu, sigma), x)
normal_log(x, mu, sigma) = logpdf(Normal(mu, sigma), x)
score_normal(x, mu, sigma) = -((x - mu) / sigma^2)

function test_density(x, k = 3, n = 20)
    alpha = exp(x)
    return x +
        x * (k - 3 / 2) +
        (-1 / (2 * alpha)) +
        loggamma(alpha) -
        loggamma(n + alpha)
end

@testset "AllocFreeWeights" begin
    v = rand(1000)

    @test @wrappedallocs(AllocFreeWeights(v)) == 0

    w = AllocFreeWeights(v)

    @test w.sum == sum(v)
    @test w.values == v
end

fn(x) = normal_log(x, 3.14, 3.0)
obj = Objective(fn)
x = 2.4

const f64gen = Data.Floats{Float64}(nans = false)
const f64genmu = Data.Floats{Float64}(nans = false, minimum = -1.0e8, maximum = 1.0e8)
const f64gensigma = Data.Floats{Float64}(nans = false, minimum = 0.0001)

@testset "Objective" begin
    @test obj isa Objective{Float64, <:Function, <:Function}
    @test obj.grad(x) ≈ score_normal(x, 3.14, 3.0)
    @check function objective_f64_grad(v = f64gen, mu = f64genmu, sigma = f64gensigma)
        o = Objective(x -> normal_log(x, mu, sigma))
        ≈(score_normal(v, mu, sigma), o.grad(v), rtol = 0.01, atol = 0.0001)
    end
    @check function objective_f64_f(v = f64gen, mu = f64genmu, sigma = f64gensigma)
        o = Objective(x -> normal_log(x, mu, sigma))
        o.f(v) ≈ normal_log(v, mu, sigma)
    end
    @test obj.grad(3.14) == 0.0
end


@testset "Truncated" begin
    f_gamma(x, α, β) = β^α * x^(α - 1) * exp(-β * x) / gamma(α)
    f_log_gamma(x, α, β) = α * log(β) + (α - 1) * log(x) - β * x - loggamma(α)
    obj = Objective(x -> f_log_gamma(x, 4, 2))
    lb = 4.0
    ub = 8.0
    sam = ARSampler(obj, (lb, ub), [lb, ub])
    v = sample!(sam, 10_000)
    @test all(lb .< v .< ub)

    obj = Objective(x -> f_log_gamma(x, 4, 2))
    lb = 9.5
    ub = 10.0
    sam = ARSampler(obj, (lb, ub), [(lb + ub) / 2])
    v = sample!(sam, 10_000)
    @test all(lb .< v .< ub)
end


const n_samples = 100_000

@testset "Autodiff backends" begin
    @testset "ForwardDiff" begin
        let
            sam = ARSampler(Objective(test_density; adbackend = AutoForwardDiff()), (-Inf, Inf))
            obj_big = Objective(test_density, one(BigFloat); adbackend = AutoForwardDiff())
            sam_big = ARSampler(obj_big, (-Inf, Inf))
            # Setup samples
            sam_2 = deepcopy(sam)
            Random.seed!(1)
            samples = sample!(sam, n_samples)
            Random.seed!(1)
            samples_2 = sample!(sam_2, n_samples)

            # Tests
            @test samples isa Vector{Float64}
            @test length(samples) == n_samples
            @test samples == samples_2 # Try to catch uninitialized memory

            # JET
            @test_opt Objective(test_density; adbackend = AutoForwardDiff())
            obj_jet = Objective(test_density; adbackend = AutoForwardDiff())
            @test_opt ARSampler(obj_jet, (-Inf, Inf))
            @test_opt sample!(sam, n_samples)


            # Test sampling BigFloat
            @test sample!(sam_big, 10_000) isa Vector{BigFloat}
        end
    end


    @testset "Mooncake" begin
        let
            sam = ARSampler(Objective(test_density; adbackend = AutoMooncake()), (-Inf, Inf))
            # Setup samples
            sam_2 = deepcopy(sam)
            Random.seed!(1)
            samples = sample!(sam, n_samples)
            Random.seed!(1)
            samples_2 = sample!(sam_2, n_samples)

            # Tests
            @test samples isa Vector{Float64}
            @test length(samples) == n_samples
            @test samples == samples_2 # Try to catch uninitialized memory

            # JET, errors from within Mooncake
            # @test_opt Objective(test_density; adbackend = AutoMooncake())
            # obj_jet = Objective(test_density; adbackend = AutoMooncake())
            # @test_opt ARSampler(obj_jet, (-Inf, Inf))
            # @test_opt sample!(sam, n_samples)
        end
    end


    @testset "Enzyme" begin
        let
            sam = ARSampler(Objective(test_density; adbackend = AutoEnzyme()), (-Inf, Inf))
            # Setup samples
            sam_2 = deepcopy(sam)
            Random.seed!(1)
            samples = sample!(sam, n_samples)
            Random.seed!(1)
            samples_2 = sample!(sam_2, n_samples)

            # Tests
            @test samples isa Vector{Float64}
            @test length(samples) == n_samples
            @test samples == samples_2 # Try to catch uninitialized memory

            # JET
            @test_opt Objective(test_density; adbackend = AutoEnzyme())
            obj_jet = Objective(test_density; adbackend = AutoEnzyme())
            @test_opt ARSampler(obj_jet, (-Inf, Inf))
            @test_opt sample!(sam, n_samples)
        end
    end
end


# import AdaptiveRejectionSampling
# ars = AdaptiveRejectionSampling

# using Test


# @testset "Line" begin
#     @test ars.Line(2.0, 3) isa ars.Line
#     @test ars.intersection(ars.Line(-1.0, 1.0), ars.Line(1.0, -1.0)) == 1.0
#     @test_throws AssertionError ars.intersection(ars.Line(-1.0, 1.0), ars.Line(-1.0, -1.0))
# end

# @testset "Envelop" begin
#     @test begin
#         l1 = ars.Line(1.0, 1.0)
#         l2 = ars.Line(-3.0, 2.0)
#         support = (-Inf, Inf)
#         ars.Envelop([l1, l2], support)
#     end isa ars.Envelop
#     @test_throws AssertionError begin
#         l1 = ars.Line(1.0, 1.0)
#         l2 = ars.Line(-3.0, 2.0)
#         ars.Envelop([l2, l1], (-Inf, Inf))
#     end
#     @test_throws AssertionError begin
#         l1 = ars.Line(1.0, 1.0)
#         l2 = ars.Line(-3.0, 2.0)
#         ars.Envelop([l2, l1], (1.0, -1.0))
#     end
#     @test begin
#         l1 = ars.Line(1.0, 1.0)
#         l2 = ars.Line(-3.0, 2.0)
#         l3 = ars.Line(-5.0, 5.0)
#         e = ars.Envelop([l1, l2, l3], (-Inf, Inf))
#         e.cutpoints == [-Inf, 0.25, 1.5, Inf]
#     end
#     @test begin
#         l1 = ars.Line(1.0, 1.0)
#         newline = ars.Line(0.0, 0.0)
#         l3 = ars.Line(-1.0, 1.0)
#         e = ars.Envelop([l1, l3], (-Inf, Inf))
#         ars.add_segment!(e, newline)
#         e.lines[2] == newline
#     end
# end

# @testset "Objective" begin
#     @test ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi)) isa ars.Objective
#     @test begin
#         objective = ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
#         objective.logf(0.0) ≈ 0.3989422804014327
#     end
#     @test begin
#         objective = ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
#         objective.grad(0.0) ≈ 0.0
#     end
#     @test begin
#         objective = ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
#         objective.logf(Inf) == 0.0 && objective.logf(Inf) == 0.0
#     end
# end

# @testset "RejectionSampler" begin
#     @test ars.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), (-Inf, Inf), (-1.0, 1.0)) isa ars.RejectionSampler
#     @test begin
#         sampler = ars.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), (-Inf, Inf), (-1.0, 1.0))
#         ars.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
#     end
#     @test begin
#         sampler = ars.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), (-Inf, Inf))
#         ars.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
#     end
#     @test_throws AssertionError begin
#         sampler = ars.RejectionSampler(x -> exp(-0.5 * x) / sqrt(2pi), (-Inf, Inf))
#         ars.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
#     end
#     @test begin
#         sampler = ars.RejectionSampler(x -> -abs(x), (-Inf, Inf), logdensity=true)
#         ars.run_sampler!(sampler, 5) isa Vector{T} where {T<:AbstractFloat}
#     end
# end
