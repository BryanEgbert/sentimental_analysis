using Distributed

process = addprocs(2, exeflags=["-t 2", "--project=$(Base.active_project())"])

@everywhere using DataFrames
@everywhere using CSV
@everywhere using TextAnalysis
@everywhere using MLJ
@everywhere using Chain
@everywhere using Pipe
@everywhere using StableRNGs
@everywhere using StringEncodings
@everywhere using MLJText
@everywhere using MLJBase
@everywhere using Plots
@everywhere using WordTokenizers
@everywhere using Tables
@everywhere using ComputationalResources
@everywhere using StatsAPI

tweet_df = CSV.File(open("sentiment_tweets3.csv", enc"ISO-8859-1")) |> DataFrame
rename!(tweet_df, [:Id, :Tweet, :Target])

suicide_df = CSV.File(open("Suicide_Detection.csv")) |> DataFrame
rename!(suicide_df, [:Id, :Tweet, :Target])

tweet_df = select!(tweet_df, Not([:Id]))
suicide_df = select!(suicide_df, Not([:Id]))
suicide_df.Target = [item == "non-suicide" ? 0 : 1 for item in suicide_df[!, "Target"]]

df = vcat(tweet_df, suicide_df, cols = :union)

df.Target = coerce(df.Target, OrderedFactor)
df.Tweet = TextAnalysis.StringDocument.(df[:, :Tweet])

tweet_df = nothing
suicide_df = nothing
GC.gc()

feat, target = MLJ.unpack(df, ==(:Tweet), ==(:Target))

crps = Corpus(feat)

update_lexicon!(crps)

remove_patterns!(crps, r"http\S+")
remove_patterns!(crps, r"#\w+")
remove_patterns!(crps, r"@\w+")
prepare!(crps, strip_case| strip_punctuation| strip_whitespace| strip_numbers| strip_non_letters| strip_stopwords| stem_words)

rng = StableRNG(100)
train, test = partition(eachindex(target), 0.7, shuffle=true, rng=rng)

MultinomialNBClassifier = @load MultinomialNBClassifier pkg=NaiveBayes
multinomial_nb_classifier_count_pipe = (feat -> tokenize.(TextAnalysis.text.(feat))) |> CountTransformer() |> MultinomialNBClassifier()

df = nothing
GC.gc()

mach = machine(multinomial_nb_classifier_count_pipe, feat[train], target[train]; cache=false)

evaluation = evaluate!(mach, resampling=CV(), measure=[Accuracy(), Precision(), recall, FScore(), LogLoss()], acceleration=CPUProcesses(), verbosity=10)
println(evaluation)

# data_plot = Plots.bar(["0", "1"], [nrow(df[(df.Target .== 0), :]), nrow(df[(df.Target .== 1), :])])

# feat, target = MLJ.unpack(df, ==(:Tweet), ==(:Target))

# remove_case!.(feat)
# TextAnalysis.remove_whitespace!.(feat)
# remove_patterns!.(feat, r"(?<=\s|^)#[\p{L}\p{N}_]+")
# remove_patterns!.(feat, r"@\w+")
# prepare!.(feat, strip_indefinite_articles| strip_punctuation| strip_pronouns| strip_numbers| strip_non_letters| strip_stopwords| stem_words)

# crps = Corpus(feat)

# update_lexicon!(crps)

# rng = StableRNG(100)
# train, test = partition(eachindex(target), 0.7, shuffle=true, rng=rng);

# MultinomialNBClassifier = @load MultinomialNBClassifier pkg=NaiveBayes
# multinomial_nb_classifier_pipe = (feat -> tokenize.(TextAnalysis.text.(feat))) |> TfidfTransformer() |> (feat -> round.(feat)) |> MultinomialNBClassifier()

# # random_forest_pipeline = (feat -> tokenize.(TextAnalysis.text.(feat))) |> TfidfTransformer() |>

# mach = machine(multinomial_nb_classifier_pipe, feat, target)
# MLJ.fit!(mach, rows=train)
# fitted_params(mach)

# evaluation = evaluate!(mach, resampling=CV(nfolds=3), measure=[Accuracy(), Precision(), TruePositiveRate(), FScore(), LogLoss()], rows=test)
# println(evaluation)

# yhat = MLJ.predict(mach, rows=test)

# println("Accuracy on the test set: $(MLJ.accuracy(mode.(yhat), target[test]))")
# println("fscore on the test set: $(MLJ.f1score(mode.(yhat), target[test]))")

# confusion_mat = ConfusionMatrix()(mode.(yhat), coerce(target[test], OrderedFactor))
# cm_plt = Plots.heatmap(confusion_mat.mat, xlabel="ground truth", ylabel="predicted values", title="Confusion Matrix of\nMNB Classifier using Rounded TF-IDF Vectorizer", size= (800, 800))

# tweet_string_docs =  tokenize.(TextAnalysis.text.(tweet_string_docs))

# tf_idf_transformer = TfidfTransformer()
# tfidf = machine(tf_idf_transformer, source(tweet_string_docs)) |> MLJ.fit!

# fitted_params(tfidf)

# tfidf_mat = MLJ.transform(tfidf, tweet_string_docs)

# feat, target = round.(tfidf_mat), df[:, :Target]

# rng = StableRNG(100)
# train, test = partition(eachindex(target), 0.7, shuffle=true, rng=rng);

# MultinomialNBClassifier = @load MultinomialNBClassifier pkg=NaiveBayes

# model = MultinomialNBClassifier()

# mach = machine(model, coerce(feat, Count), coerce(target, Finite); cache=false)

# MLJ.fit!(mach, rows=train)
# fitted_params = fitted_params(mach)

# probability = MLJ.predict(mach, rows=test)

# println("Log loss on the test set: $(log_loss(probability, target[test]) |> mean)")

