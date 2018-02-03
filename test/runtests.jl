import AdaptiveRejectionSampling
arj = AdaptiveRejectionSampling

using Base.Test

@testset "Line" begin
    @test arj.Line(2.0, 3) isa arj.Line
    @test arj.intersection(arj.Line(-1.0, 1.0), arj.Line(1.0, -1.0)) == 1.0
    @test_throws AssertionError arj.intersection(arj.Line(-1.0, 1.0), arj.Line(-1.0, -1.0))
end

@testset "Envelop" begin
    @test begin
        l1 = arj.Line(1.0, 1.0)
        l2 = arj.Line(-3.0, 2.0)
        arj.Envelop([l1, l2])
    end isa arj.Envelop
    @test_throws AssertionError begin
        l1 = arj.Line(1.0, 1.0)
        l2 = arj.Line(-3.0, 2.0)
        arj.Envelop([l2, l1])
    end
    @test begin
        l1 = arj.Line(1.0, 1.0)
        l2 = arj.Line(-3.0, 2.0)
        l3 = arj.Line(-5.0, 5.0)
        p = arj.Envelop([l1, l2, l3])
        p.cutpoints == [0.25, 1.5]
    end
    @test begin
        l1 = arj.Line(1.0, 1.0)
        newline = arj.Line(0.0, 0.0)
        l3 = arj.Line(-1.0, 1.0)
        p = arj.Envelop([l1, l3])
        arj.add_line_segment!(p, newline)
        p.lines[2] == newline
    end
end

@testset "Objective" begin
    @test arj.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi)) isa arj.Objective
    @test begin
        objective = arj.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
        objective.logf(0.0) ≈ 0.3989422804014327
    end
    @test begin
        objective = arj.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
        objective.grad(0.0) ≈ 0.0
    end
    @test begin
        objective = arj.Objective(x -> exp(-0.5 * x^2) / sqrt(2pi))
        objective.logf(Inf) == 0.0 && objective.logf(Inf) == 0.0
    end
end

@testset "RejectionSampler" begin
    @test arj.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), -1.0, 1.0) isa arj.RejectionSampler
    @test begin
        sampler = arj.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi), -1.0, 1.0)
        arj.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
    end
    @test begin
        sampler = arj.RejectionSampler(x -> exp(-0.5 * x^2) / sqrt(2pi))
        arj.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
    end
    @test_throws AssertionError begin
        sampler = arj.RejectionSampler(x -> exp(-0.5 * x) / sqrt(2pi))
        arj.run_sampler!(sampler, 5) isa Vector{T} where T <: AbstractFloat
    end
end
