using Titanic
using Test

@testset "Titanic.jl" begin
    df = DataFrame(;
                   A=10:30,
                   B=50:70,
                   C=vcat([missing], collect(range(1, 18)), [missing, 2]),
                   D=rand(["a", "b"], 21))
    @testset "Read CSV" begin
        @test_throws ErrorException read_csv_data("train.csv")
    end

    # Write your tests here.
    @testset "Utils" begin
        @testset "Standartization" begin
            @test mean(standartize(df, :A)[!, :A]) == 0.0
            @test std(standartize(df, :A)[!, :A]) - 1.0 < 1e-9
        end
        @testset "Categorization" begin
            @test sort(unique(categorize(df, :D)[!, :D])) == [0, 1]
        end
    end
end
