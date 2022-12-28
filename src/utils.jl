
function read_csv_data(name::String)
    ispath(name) || throw(ErrorException("Invalid data path"))
    return CSV.read(name, DataFrame; header=true)
end

replace_eq(value::Any, src::Missing, dst::Any) = ismissing(value) ? dst : value # coalesce
replace_eq(value::Any, src::Any, dst::Any) = value == src ? dst : value

function replace_in_cols(df::DataFrame,
                         cols::Union{String,Symbol,Vector{<:Union{String,Symbol}}},
                         src::Any,
                         dst::Any)
    return transform(df, cols .=> ByRow(entry -> replace_eq(entry, src, dst)) .=> cols)
end

function replace_in_cols!(df::DataFrame,
                          cols::Union{String,Symbol,Vector{<:Union{String,Symbol}}},
                          src::Any,
                          dst::Any)
    return transform!(df, cols .=> ByRow(entry -> replace_eq(entry, src, dst)) .=> cols)
end

function replace_in_cols(df::DataFrame, src::Any, dst::Any)
    return replace_in_cols(df, names(df), src, dst)
end
function replace_in_cols!(df::DataFrame, src::Any, dst::Any)
    return replace_in_cols!(df, names(df), src, dst)
end

function categorize(df::DataFrame, cols::Vector{<:Union{String,Symbol}})
    return transform(df,
                     cols .=>
                         (col -> map(entry -> col_to_categorical(col, entry), col)) .=>
                             cols)
end

categorize(df::DataFrame) = categorize(df, names(df))
categorize(df::DataFrame, cols::Union{Symbol,String}) = categorize(df, [cols])
function col_to_categorical(col::AbstractArray, value::Any)
    return findfirst(entry -> entry == value, unique(col)) - 1
end
col_to_categorical(col::AbstractArray, value::Number) = value

function standartize(df::DataFrame, cols::Vector{<:Union{String,Symbol}})
    return transform(df,
                     cols .=> (col -> map(entry -> col_to_norm(col, entry), col)) .=> cols)
end

col_to_norm(col::AbstractArray{<:Number}, value::Number) = (value - mean(col)) / std(col)
standartize(df::DataFrame) = standartize(df, names(df))
standartize(df::DataFrame, cols::Union{Symbol,String}) = standartize(df, [cols])

function apply_to_cols(df::DataFrame, cols::Union{Symbol,String}, func::Function)
    return func(collect(skipmissing(df[!, cols])))
end
function apply_to_cols(df::DataFrame, cols::Vector{<:Union{String,Symbol}}, func::Function)
    return map(col -> apply_to_cols(df, col, func), cols)
end # we DO NOT have Point{Float64} <: Point{Real}
apply_to_cols(df::DataFrame, func::Function) = apply_to_cols(df, names(df), func)

most_common(arr::AbstractArray) = argmax(countmap(arr))
strip_cabin_numbers(arr::AbstractArray{<:AbstractString}) = map(entry -> entry[begin], arr)

function get_title_groups()
    default_titles = ["Mr.", "Mrs.", "Mlle.", "Miss.", "Mme.", "Ms."]
    royal_titles = ["Master.", "Don.", "Lady.", "Sir.", "Countess.", "Jonkheer."]
    other_titles = ["Major.", "Col.", "Capt.", "Dr.", "Rev."]

    return [(default_titles, "D"), (royal_titles, "R"), (other_titles, "O")]
end

function replace_names_with_title_categories(df, groups)
    for group in groups
        transform!(df,
                   :Name => ByRow(entry -> any(occursin.(group[begin], entry)) ? group[end] : entry) => :Name)
    end
    return df
end

function random_split(df::DataFrame, ratios::AbstractArray{<: Number}; seed = 42)
    all(0.0 .<= ratios .<= 1.0) || 0.0 < sum(ratios) <= 1.0 || throw(ErrorException("Invalid size ratios"))
    rng = MersenneTwister(seed)
    df_shuffled = shuffle(rng, df)
    splits = []
    rest = copy(df_shuffled)
    for ratio in ratios
        split = rest[begin:Int(floor(ratio*nrow(df))), :]
        rest = rest[(Int(floor(ratio*nrow(df))) + 1):end, :] 
        splits = vcat(splits, split)
    end
    return splits
end

accuracy(y1, y2) = sum(y1 .== y2) / length(y2)
