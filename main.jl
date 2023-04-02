using DataFrames
using CSV
using TextAnalysis
using MLJ
using Chain
using Pipe
using StableRNGs
using StringEncodings
using MLJText
using MLJBase
using Plots

df = CSV.File(open("sentiment_tweets3.csv", enc"ISO-8859-1")) |> DataFrame
rename!(df, [:Id, :Tweet, :Target])
# df = CSV.read("sentiment_tweets3.csv", DataFrames.DataFrame; header=[:Target, :Id, :Date, :Flag, :User, :Tweet])

Plots.bar(["0", "1"], )
tweet_string_docs = TextAnalysis.StringDocument.(df[:, :Tweet])

# remove_case!.(tweet_string_docs)
prepare!.(tweet_string_docs, strip_case| strip_punctuation| strip_numbers| strip_non_letters| strip_pronouns| strip_stopwords| stem_words)

crps = Corpus(tweet_string_docs)

update_lexicon!(crps)

tweet_string_docs =  tokenize.(TextAnalysis.text.(tweet_string_docs))
sizeof(tweet_string_docs)

tf_idf_transformer = TfidfTransformer()
mach = machine(tf_idf_transformer, tweet_string_docs; cache=false) |> fit!

fitted_params(mach)

tfdif_mat = MLJ.transform(mach, tweet_string_docs)


feat, target = round.(tfdif_mat), df[:, :Target]

MultinomialNBClassifier = @load MultinomialNBClassifier pkg=NaiveBayes

model = MultinomialNBClassifier()

mach = machine(model, coerce(feat, Count), coerce(target, Finite); cache=false)

rng = StableRNG(100)
train, test = partition(eachindex(target), 0.7, shuffle=true, rng=rng);

MLJ.fit!(mach, rows=train)

probability = MLJ.predict(mach, rows=test)

println("Log loss on the test set of index 1-10: $(log_loss(probability, target[test]) |> mean)")

println("Accuracy on the test set of index 1-10: $(accuracy(mode.(probability), target[test]))")

confusion_mat = ConfusionMatrix()(mode.(probability), coerce(target[test], OrderedFactor))
cm_plt = Plots.heatmap(confusion_mat.mat, xlabel="ground truth", ylabel="predicted values", title="Confusion Matrix of\nMNB Classifier using Rounded TF-IDF Vectorizer", size= (800, 800))

