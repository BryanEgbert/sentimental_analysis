# sentimental_analysis
Sentimental analysis for final projects using [Julia](https://julialang.org) and Python, it is recommended to use the latest version of python. See [Setup Julia in VSCode](https://code.visualstudio.com/docs/languages/julia)

## Setup
- **Julia:**  
    Go to file `depression_detection_jl.ipynb` and run the code. Make sure to run it from top to bottom.
- **Python:**  
    Before running the Python code in `depression_detection_python.ipynb`, make sure to install the required packages, it is 
    recommended to install them in a python virtual environment assuming that you run this in a code editor other than Google Colab.  
    - Creating python virtual environment 
        ```powershell
        > py -m venv env
        ```
    - Activate python virtual environment
        ```powershell
        > ./env/Scripts/activate
        ```
    - Install required packages
        ```powershell
        > pip install -r requirements.txt
        ```
data used: [Sentimental Analysis for Tweets Uploaded by SHINIGAMI](https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets)


## Other Datasets Options in Kaggle
- [Twitter Sentiment Analysis Uploaded by passionate-nlp](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Sentiment140 dataset with 1.6 million tweets by Μαριος Μιχαηλιδης KazAnova in Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&searchQuery=SVM)

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
