using DataFrames
using CSV
using Gadfly
using TextAnalysis
using MLJ
using Chain
using Pipe
using StableRNGs
using StringEncodings

df = CSV.File(open("training.1600000.processed.noemoticon.csv", enc"ISO-8859-1");  header=[:Target, :Id, :Date, :Flag, :User, :Tweet]) |> DataFrame
# df = CSV.read("training.1600000.processed.noemoticon.csv", DataFrames.DataFrame; header=[:Target, :Id, :Date, :Flag, :User, :Tweet])
# first(df, 10) |> pretty

transform!(df, :Tweet => ByRow(x -> StringDocument(x)) => :Tweet2, renamecols=false)

remove_case!.(df[:, :Tweet2])
prepare!.(df[:, :Tweet2], strip_punctuation| strip_numbers)
stem!.(df[:, :Tweet2])

crps = Corpus(df[:, :Tweet2])

update_lexicon!(crps)

matrix = DocumentTermMatrix(crps)
tfdif_mat = tf_idf(matrix)

feat, target = round.(tfdif_mat), df[:, :Target]

MultinomialNBClassifier = @load MultinomialNBClassifier pkg=NaiveBayes

model = MultinomialNBClassifier()

mach = machine(model, coerce(feat, Count), coerce(target, Finite); cache=false)

rng = StableRNG(100)
train, test = partition(eachindex(target), 0.001, shuffle=true, rng=rng);

MLJ.fit!(mach, rows=train)
