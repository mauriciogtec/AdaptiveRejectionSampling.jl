import AdaptiveRejectionSampling
ars = AdaptiveRejectionSampling

using Test

@testset "Line" begin
    @test ars.Line(2.0, 3) isa ars.Line
    @test ars.intersection(ars.Line(-1.0, 1.0), ars.Line(1.0, -1.0)) == 1.0
    @test_throws AssertionError ars.intersection(ars.Line(-1.0, 1.0), ars.Line(-1.0, -1.0))
end

@testset "Envelop" begin
    @test begin
        l1 = ars.Line(1.0, 1.0)
        l2 = ars.Line(-3.0, 2.0)
        support = (-Inf, Inf)
        ars.Envelop([l1, l2], support)
    end isa ars.Envelop
    @test_throws AssertionError begin
        l1 = ars.Line(1.0, 1.0)
        l2 = ars.Line(-3.0, 2.0)
        ars.Envelop([l2, l1], (-Inf, Inf))
    end
    @test_throws AssertionError begin
        l1 = ars.Line(1.0, 1.0)
        l2 = ars.Line(-3.0, 2.0)
        ars.Envelop([l2, l1], (1.0, -1.0))
    end
    @test begin
        l1 = ars.Line(1.0, 1.0)
        l2 = ars.Line(-3.0, 2.0)
        l3 = ars.Line(-5.0, 5.0)
        e = ars.Envelop([l1, l2, l3], (-Inf, Inf))
        e.cutpoints == [-Inf, 0.25, 1.5, Inf]
    end
    @test begin
        l1 = ars.Line(1.0, 1.0)
        newline = ars.Line(0.0, 0.0)
        l3 = ars.Line(-1.0, 1.0)
        e = ars.Envelop([l1, l3], (-Inf, Inf))
        ars.add_segment!(e, newline)
        e.lines[2] == newline
    end
end

@testset "Objective" begin
    @test ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi)) isa ars.Objective
    @test begin
        objective = ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
        objective.logf(0.0) ≈ 0.3989422804014327
    end
    @test begin
        objective = ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
        objective.grad(0.0) ≈ 0.0
    end
    @test begin
        objective = ars.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
        objective.logf(Inf) == 0.0 && objective.logf(Inf) == 0.0
    end
end

@testset "RejectionSampler" begin
    @test ars.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), (-Inf, Inf), (-1.0, 1.0)) isa ars.RejectionSampler
    @test begin
        sampler = ars.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), (-Inf, Inf), (-1.0, 1.0))
        ars.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
    end
    @test begin
        sampler = ars.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), (-Inf, Inf))
        ars.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
    end
    @test_throws AssertionError begin
        sampler = ars.RejectionSampler(x -> exp(-0.5 * x) / sqrt(2pi), (-Inf, Inf))
        ars.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
    end
end
