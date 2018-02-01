import AdaptiveRejectionSampling
m = AdaptiveRejectionSampling

using Base.Test

# # write your own tests here
# @testset "Lines, slopes and interceptions" begin
#     # @test m.LineSegment(1, 2, -1.0, 3) isa m.LineSegment
#     # @test_throws ArgumentError m.LineSegment(0, 0, 0.0 ,0)
#     # @test m.slope(m.LineSegment(0, 0, 1, 1)) == 1.0
#     # @test m.slope(m.LineSegment(0, 0, 0, 1)) == Inf
#     # @test m.intercept(m.LineSegment(0, 1, 1, 0)) == 1.0
# end

# write your own tests here
@testset "Lines and intersections" begin
    @test m.LineSegment(-1.0, 4.0, -Inf, 10.0) isa m.LineSegment
    @test_throws ArgumentError m.LineSegment(-1.0, 4.0, Inf, 10.0)
    @test m.intersection(m.LineSegment(1.0, .0, -Inf, Inf), m.LineSegment(-1.0, 1.0, -Inf, Inf)) == (0.5, 0.5)
    @test_throws ArgumentError m.intersection(m.LineSegment(0.0, 0.0, -Inf, Inf), m.LineSegment(0.0, 1.0, -Inf, Inf))
end
