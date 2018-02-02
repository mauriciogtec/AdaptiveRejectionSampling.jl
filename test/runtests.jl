using AdaptiveRejectionSampling

using Base.Test

@testset "Lines and intersection" begin
    @test Line(2.0, 3) isa Line
    @test intersection(Line(-1.0, 1.0), Line(1.0, -1.0)) == 1.0
    @test_throws AssertionError intersection(Line(-1.0, 1.0), Line(-1.0, -1.0))
end

@testset "Ordered partitions and add segments" begin
    @test begin
        l1 = Line(1.0, 1.0)
        l2 = Line(-3.0, 2.0)
        OrderedPartition([l1, l2])
    end isa OrderedPartition
    @test_throws AssertionError begin
        l1 = Line(1.0, 1.0)
        l2 = Line(-3.0, 2.0)
        OrderedPartition([l2, l1])
    end
    @test begin
        l1 = Line(1.0, 1.0)
        l2 = Line(-3.0, 2.0)
        l3 = Line(-5.0, 5.0)
        p = OrderedPartition([l1, l2, l3])
        p.cutpoints == [0.25, 1.5]
    end
    @test begin
        l1 = Line(1.0, 1.0)
        newline = Line(0.0, 0.0)
        l3 = Line(-1.0, 1.0)
        p = OrderedPartition([l1, l3])
        add_line_segment!(p, newline)
        p.lines[2] == newline
    end
end
