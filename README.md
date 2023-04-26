# sentimental_analysis
Sentimental analysis for final projects using [Julia](https://julialang.org). [Setup Julia in VSCode](https://code.visualstudio.com/docs/languages/julia)

## Important Notes
Before compiling the code make sure to follow this step:
- Open command palette (ctrl + shift + p)
- Choose `Julia: Change Current Environment`
- Choose `sentimental_analysis` on the environment options

If you already installed the julia extension in VSCode, it is similar to jupyter notebook but in VSCode without using the `.ipynb` file extension, `Ctrl + Enter` to execute a line based on your cursor or multiple lines of highlighted code

data used: [Sentiment140 dataset with 1.6 million tweets by Μαριος Μιχαηλιδης KazAnova in Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&searchQuery=SVM) (my laptop cannot handle this much data)

## Other Datasets Options in Kaggle
- [Twitter Sentiment Analysis Uploaded by passionate-nlp](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Sentimental Analysis for Tweets Uploaded by SHINIGAMI](https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets) (currently being used)

## Model Pipeline Performance
Doing this just for fun
### **Physical Device Specs For This Benchmark:**
- CPU: Intel Core i5 8th Gen
- GPU: NVIDIA Geforce MX250
- RAM: 8GB

### **Data Cleaning Method**
Both in Julia and Python use the same data cleaning methods
- Lowercase string
- Remove hashtags
- Remove mentions
- Remove hyperlink
- Remove exessive whitespace
- Remove numbers and other non letters word
- Remove punctuation
- Remove stopwords

**Text Extraction Method Used:** Count vectorizer

### **Multinomial Naive Bayes Classifier**
| Measurements | Julia | Python |
|--------------|-------|--------|
| Accuracy | **96.4%** | 94.8% |
| F1 Score | **92.3%** | 88.5% |
| Precision | **89.8%** | 89.3% |
| Recall | **94.9%** | 87.7% |

### **SVC**
| Measurements | Julia | Python |
|--------------|-------|--------|
| Accuracy | **97.7%** | 93.8% |
| F1 Score | **94.6%** | 84.4% |
| Precision | **100%** | 99.4% |
| Recall | **89.9%** | 73.4% |

### **Random Forest Classifier**
| Measurements | Julia | Python |
|--------------|-------|--------|
| Accuracy | **98%** | 94.9% |
| F1 Score | **95.3%** | 87.6% |
| Precision | **100%** | 98.7% |
| Recall | **91%** | 78.8% |