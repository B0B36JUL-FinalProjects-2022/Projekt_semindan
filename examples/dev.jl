using Revise
using Titanic
using DataFrames
using DataStructures
using Statistics
using StatsBase

df = read_csv_data("data/train.csv")
src = "male"
dst = "whale"
cols = names(df)

replace_in_cols!(df, cols, src, dst)
df.Sex

cols = names(df)
df
df_conc =
    transform(df, :Sex => (col -> map(entry -> entry == "male" ? 0 : 1, col)) => :Concat)
    
replace(df,  ByRow(x -> x*2) )
replace( df[!, [:Age, :Fare]], missing .=> 1)
transform!(df, [:Age, :Fare] .=> ByRow(x -> 2x) .=> [:Age, :Fare])
transform(df, "Age" .=> ByRow(x -> 1/231132*x) .=> :Age)
replace_in_cols(df, missing, "?")    
df
# transform(df, "Age" => (col -> map(entry -> entry == "male" ? 0 )
df
DataStructures.counter(df_conc)
unique(df[:, :Sex])

Matrix{Float64}(df)

mapcols(col -> map(entry -> reverse(entry), col), df[!, ["Name", "Sex"]]) # good

a = mapcols(
    col -> map(entry -> findfirst((value -> value == entry), unique(col)) - 1, col),
    df[!, [:Name, :Sex]],
) # good
identity(a)
s_col = df.Sex
find("male", unique(s_col))
findall((x -> x == "male"), unique(s_col))
getindex(unique(df[!, :Sex]), 2)
tst_df = categorical_to_int(df, ["Name", "Sex"])
tst_mean = mean.(eachcol(categorical_to_int(df, ["Name", "Sex"])))

map(x -> x, eachcol(tst_df))
standartize(df)
mapcols(col -> (col .- mean(col)) ./ std(col), tst_df)
cor(tst_df)
correlat
cor(Matrix(tst_df))

df
completecases(df, :Name)

nrow(df)
df
replace_in_cols(df, "Survived", 1, 90000)
replace_in_cols(df, "Age", 1, 90000)


tst_df = replace_in_cols(df, names(df), missing, "a")
ttt = copy(df)
ttt

col_median_2 = apply_to_cols(df, ["Age", "Fare"], median)
col_median_2 = apply_to_cols(df, median)
col_median_2 = apply_to_cols(df, "Age", median)
ttt = replace_in_cols(ttt, "Age", missing, 9000.0)
replace_in_cols(ttt, "Age", 22, 9000.02222)
replace_in_cols!(ttt, missing, 9000)
replace_in_cols(tst_df, "Age", 90000, 122222222)
replace_in_cols(df, "Age", missing, 9000)
mapcols!(col -> map(entry -> (ismissing(entry) ? 9000 : entry), col), df[!, ["Age"]])
df[!, :Age]


transform(median, df[!, :Age])
replace_in_cols!(df, ["Age"], missing, 90000)

df
median.(eachcol(skipmissing(df[!, :Age])))

median(skipmissing(df[!, [:Age, :Fare]]), dims=1)
df