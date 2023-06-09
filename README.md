# sentimental_analysis
Sentimental analysis for final projects using Python

## Setup
Before running the Python code in `depression_detection_python.ipynb`, make sure to install the required packages, it is recommended to install them in a python virtual environment assuming that you run this in a code editor other than Google Colab.  
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
dataset used: 
- [Sentimental Analysis for Tweets Uploaded by SHINIGAMI](https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets)
- [Suicide and Depression Detection](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)


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
| Measurements | Python |
|--------------|-------|
| Accuracy | 88.90% |
| F1 Score | 89.30% |
| Precision | 84.44% |
| Recall | 94.76% |

### **SVC**
| Measurements | Python |
|--------------|--------|
| Accuracy | 89.20% |
| F1 Score | 88.29% |
| Precision | 93.81% |
| Recall | 83.39% |

### **Random Forest Classifier**
| Measurements  | Python |
|--------------|-------|
| Accuracy | 88.31% |
| F1 Score | 88.36% |
| Precision  | 86.52% |
| Recall | 90.29% |

### **XGBoost Classifier**
| Measurements  | Python |
|--------------|-------|
| Accuracy | 89.77% |
| F1 Score | 89.18% |
| Precision  | 92.28% |
| Recall | 86.27% |

### **CatBoost Classifier**
| Measurements  | Python |
|--------------|-------|
| Accuracy | 90.79% |
| F1 Score | 90.28% |
| Precision  | 93.14% |
| Recall | 87.6% |
