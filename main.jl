using DataFrames
using CSV
using Gadfly
using TextAnalysis
using MLJ
using Chain
using Pipe
using StableRNGs
using StringEncodings
using MLJBase

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


partial_test_dataset = test[1:50]
probability = MLJ.predict(mach, rows=partial_test_dataset)

println("Log loss on the test set of index 1-10: $(log_loss(probability, target[partial_test_dataset]) |> mean)")

println("Accuracy on the test set of index 1-10: $(accuracy(mode.(probability), target[partial_test_dataset]))")