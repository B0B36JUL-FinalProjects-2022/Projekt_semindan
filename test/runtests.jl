using Titanic
using Test
using DataFrames
using Statistics
using StatsBase

@testset "Titanic.jl" begin
    df = DataFrame(;
                   A = 10:30,
                   B = 50:70,
                   C = vcat([missing], collect(range(1, 18)), [missing, 2]),
                   D = rand(["a", "b"], 21),
                   E = rand(["c", "d"], 21),
                   F = rand(["eee", "fff", "ggg"], 21),
                   G = rand(["a", "b", "k"], 21),
                   L = vcat(zeros(10), ones(11)))
    @testset "Read CSV" begin @test_throws ErrorException read_csv_data("something.csv") end

    @testset "Utils" begin
        @testset "Replace" begin
            @test count(ismissing, replace_in_cols(df, :C, missing, 0)[!, :C]) == 0
            @test sort(unique(replace_in_cols(df, [:D, :G], ["a", "b"], ["A", "B"])[!, :D])) ==
                  ["A", "B"]
        end
        @testset "Apply" begin
            @test apply_to_cols(df, :A, median) == median(10:30)
            @test apply_to_cols(df, [:A, :B], median) == [median(10:30), median(50:70)]
            @test apply_to_cols(df, :B, arr -> map(x -> x + 1, arr)) == collect(50:70) .+ 1
            @test apply_to_cols(df, [:B], arr -> map(x -> x + 1, arr)) ==
                  [collect(50:70) .+ 1]
        end
        @testset "Standartization" begin
            @test_throws MethodError standartize(df)
            @test -1e-9 < std(standartize(df, :A)[!, :A]) - 1.0 < 1e-9
            @test mean(standartize(df, :A)[!, :A]) == 0.0
            @test -1e-9 < std(standartize(df, [:A, :B])[!, :A]) - 1.0 < 1e-9
            @test mean(standartize(df, [:A, :B])[!, :A]) == 0.0
            @test -1e-9 < std(standartize(df, [:A, :B])[!, :B]) - 1.0 < 1e-9
            @test mean(standartize(df, [:A, :B])[!, :B]) == 0.0
        end
        @testset "Categorization" begin
            @test sort(unique(categorize(df, :D)[!, :D])) == [0, 1]
            @test_throws TypeError categorize(df)
            @test sort(unique(categorize(df, [:E, :D])[!, :D])) == [0, 1]
            @test sort(unique(categorize(df, [:E, :D])[!, :E])) == [0, 1]
        end
        @testset "Onehot" begin
            @test length(names(to_onehot(df, :D; remove_original = true))) ==
                  length(names(df)) + length(unique(df[!, :D])) - 1
            @test length(names(to_onehot(df, [:D, :G]; remove_original = true))) ==
                  length(names(df)) +
                  length(unique(df[!, :D])) +
                  length(unique(df[!, :G])) - 1 - 1
        end
        @testset "Random Split" begin
            @test_throws ErrorException random_split(df, [0.2, 0.2])
            @test length(random_split(df, [0.2, 0.2, 0.6])) == 3
        end
    end

    @testset "Models" begin
        @testset "K_nn" begin
            @test_throws ErrorException K_nn(n = 0)
            @test K_nn().metric == K_nn(metric = Titanic.l2).metric
            knn = K_nn()
            model_fit!(knn, df[1:12, [:A]], df[1:12, :L])
            @test model_predict(knn, df[1:21, [:A]]) == df[!, :L]
            model_fit!(knn, df[1:11, [:A]], df[1:11, :L])
            @test model_predict(knn, df[1:21, [:A]]) == zeros(21)
        end
        @testset "Log_reg" begin
            @test_throws ErrorException Log_reg(lr = -1)
            log_reg = Log_reg()
            X = [1 1 1; 1 2 3]
            y = [1, -1, -1]
            w = [1.5, -0.5]
            df_mat = Matrix(Matrix(df[1:12, [:A]])')
            df_y = replace(df[1:12, :L], 0 => -1)
            w_w = ones(size(df_mat)[1])
            @test floor(logistic_loss(X, y, w); digits = 2) == 0.66
            @test floor.(logistic_loss_grad(X, y, w); digits = 2) == [0.28, 0.82]
            model_fit!(log_reg, df[1:12, [:A]], df[1:12, :L])
            @test model_predict(log_reg, df[1:21, [:A]]) == zeros(21)
        end
        @testset "Neural_network" begin
            # I don't even know...
        end
        @testset "Decision_tree" begin
            dt = Decision_tree(; criterion = entropy_local)
            model_fit!(dt, df[1:12, [:A]], df[1:12, :L])
            @test model_predict(dt, df[1:21, [:A]]) == df[1:21, :L]
        end
    end
end
