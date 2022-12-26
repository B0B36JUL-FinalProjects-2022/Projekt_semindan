
function read_csv_data(name::String)
    ispath(name) || throw(error("Invalid data path"))
    return CSV.read(name, DataFrame; header=true)
end

function replace_in_cols(df::DataFrame, cols::Union{String, Symbol, Vector{<:Union{String, Symbol}}}, src::Any, dst::Any)
    return transform(df, cols .=> ByRow(entry -> entry == src ? dst : entry) .=> cols)
end

function replace_in_cols!(df::DataFrame, cols::Union{String, Symbol, Vector{<:Union{String, Symbol}}}, src::Any, dst::Any)
    transform!(df, cols .=> ByRow(entry -> entry == src ? dst : entry) .=> cols)
end

function replace_in_cols(df::DataFrame, cols::Union{String, Symbol, Vector{<:Union{String, Symbol}}}, src::Missing, dst::Any)
    return transform(df, cols .=> ByRow(entry -> ismissing(entry) ? dst : entry) .=> cols)
end

function replace_in_cols!(df::DataFrame, cols::Union{String, Symbol, Vector{<:Union{String, Symbol}}}, src::Missing, dst::Any)
    transform!(df, cols .=> ByRow(entry -> ismissing(entry) ? dst : entry) .=> cols)
end

replace_in_cols(df::DataFrame, src::Any, dst::Any) = replace_in_cols(df, names(df), src, dst)
replace_in_cols!(df::DataFrame, src::Any, dst::Any) = replace_in_cols!(df, names(df), src, dst)

function to_categorical(df::DataFrame, cols::Vector{String})
    return mapcols(
        col -> map(entry -> findfirst((value -> value == entry), unique(col)) - 1, col),
        df[!, cols],
    )
end

function standartize(df::DataFrame)
    return mapcols(col -> (col .- mean(col)) ./ std(col), df)
end

apply_to_cols(df::DataFrame, cols::Union{Symbol, String}, func::Function) = func(skipmissing(df[!, cols]))
apply_to_cols(df::DataFrame, cols::Vector{<: Union{String, Symbol}}, func::Function) = map(col -> apply_to_cols(df, col, func), cols) # we DO NOT have Point{Float64} <: Point{Real}
apply_to_cols(df::DataFrame, func::Function) = apply_to_cols(df, names(df), func)