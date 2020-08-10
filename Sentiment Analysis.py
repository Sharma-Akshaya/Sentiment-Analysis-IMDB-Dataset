import glob
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
import string

import copy

import pickle
import re
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
file_list_tr_p = glob.glob(os.path.join(os.getcwd(), "dataset/train/pos", "*.txt"))
file_list_tr_n = glob.glob(os.path.join(os.getcwd(), "dataset/train/neg", "*.txt"))

train_data = []

for file_path in file_list_tr_p:
    with open(file_path) as f_input:
        train_data.append(f_input.read())

for file_path in file_list_tr_n:
    with open(file_path) as f_input:
        train_data.append(f_input.read())


file_list_te_p = glob.glob(os.path.join(os.getcwd(), "dataset/test/pos", "*.txt"))
file_list_te_n = glob.glob(os.path.join(os.getcwd(), "dataset/test/neg", "*.txt"))

test_data = []

for file_path in file_list_te_p:
    with open(file_path) as f_input:
        test_data.append(f_input.read())

for file_path in file_list_te_n:
    with open(file_path) as f_input:
        test_data.append(f_input.read())
'''


# Reading Data
trainningD = pd.read_csv("IMDB_Dataset_rep.csv",header=0)



# Split data to test preprocessing and modeling techniques
norm_trainData_review, norm_testData_review, trainData_senti, testData_senti = train_test_split(trainningD.drop(columns=['sentiment']),
                                                    trainningD.sentiment, test_size=0.2)

norm_trainData_review = norm_trainData_review.reset_index(drop=True)
norm_testData_review = norm_testData_review.reset_index(drop=True)

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
remove_stopword=nltk.corpus.stopwords.words('english')

#Removing the html strips
def remove_html(txtData):
    soup = BeautifulSoup(txtData, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_character_insquare(txtData):
    return re.sub('\[[^]]*\]', '', txtData)

#Removing the noisy text
def noise_remove(txtData):
    txtData = remove_html(txtData)
    txtData = remove_character_insquare(txtData)
    return txtData




#Define function for removing special characters
def removal_of_specialCharacter(txtData, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    txtData=re.sub(pattern,'',txtData)
    return txtData


#Stemming the text
def porter_stemming(txtData):
    ps=nltk.porter.PorterStemmer()
    txtData= ' '.join([ps.stem(word) for word in txtData.split()])
    return txtData


#set stopwords to english
stop=set(stopwords.words('english'))
#print(stop)

#removing the stopwords
def remove_stopwords(txtData, is_lower_case=False):
    tokensn = tokenizer.tokenize(txtData)
    tokensn = [token.strip() for token in tokensn]
    if is_lower_case:
        filtere_token = [token for token in tokensn if token not in remove_stopword]
    else:
        filtere_token = [token for token in tokensn if token.lower() not in remove_stopword]
    filtered_text = ' '.join(filtere_token)
    return filtered_text
#Apply function on review column
#movie_data['review']=movie_data['review'].apply(remove_stopwords)

def preprocessing(review, string = True, remove_stopwords=True):
 
   
    #Apply function on review column
    filterData =noise_remove(review)
    #Apply function on review column
    filterData = removal_of_specialCharacter(filterData)
    #Apply function on review column porter_stemming
    data = porter_stemming(filterData)
      # split data and change the data to lower case
    data = filterData.lower().split()
   
    # Removing stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        data = [token for token in data if not token in stops]
    if string:
        return " ".join(data)
    else:
        return data

# Get list of reviews
preprocessed_train = [preprocessing(norm_trainData_review["review"][i]) for i in range(len(norm_trainData_review))]
preprocessed_test = [preprocessing(norm_testData_review["review"][i]) for i in range(len(norm_testData_review))]

#Count vectorizer for bag of words
cntVect = CountVectorizer(analyzer = "word",  
                             tokenizer = None,    
                             preprocessor = None,
                             stop_words = None,  
                             max_features = 5000)

#transformed train reviews
cntVect_trainData_review = cntVect.fit_transform(preprocessed_train).toarray()
#transformed test reviews
cntVect_testData_review = cntVect.transform(preprocessed_test).toarray()

print('BOW_cntVect_train:',cntVect_trainData_review.shape)
print('BOW_cntVect_test:',cntVect_testData_review.shape)

# Toget feature names
vocab = cntVect.get_feature_names()
#training the model
def Classifier(cntVect_testData_review, cntVect_trainData_review, testData_senti, trainData_senti, model_name, Model_used='Random Forest'):
    #Fitting the model for Bag of words/tfidf
    model_name.fit(cntVect_trainData_review, trainData_senti)
    #Predicting the model for bag of words/tfidf
    Predictions = model_name.predict(cntVect_testData_review)
    falsePR, truePR, _ = roc_curve(testData_senti, Predictions)
    roc_areaUnderCurve = auc(falsePR, truePR)
#Accuracy score
    print('Accuracy score for ', Model_used,' :', roc_areaUnderCurve)
    plt.plot(falsePR, truePR)
    plt.title(Model_used)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return Predictions

RandomForest = Classifier(cntVect_testData_review, cntVect_trainData_review,
                  testData_senti, trainData_senti,
                  RandomForestClassifier(n_estimators = 100))
#Classification report for bag of words
RandomForest_bow_report=classification_report(testData_senti,RandomForest,target_names=['Positive','Negative'])
print(RandomForest_bow_report)
#confusion matrix for bag of words
Cmatrx_bow=confusion_matrix(testData_senti,RandomForest,labels=[1,0])
print(Cmatrx_bow)
mnb_bow = Classifier(cntVect_testData_review, cntVect_trainData_review,
                  testData_senti, trainData_senti,
                  MultinomialNB(), Model_used='Naive Bayes')
#Classification report for bag of words
mnb_bow_report=classification_report(testData_senti,mnb_bow,target_names=['Positive','Negative'])
print(mnb_bow_report)
#confusion matrix for bag of words
Cmatrx_bow=confusion_matrix(testData_senti,mnb_bow,labels=[1,0])
print(Cmatrx_bow)
preds_LogisticR = Classifier(cntVect_testData_review, cntVect_trainData_review,
                  testData_senti, trainData_senti,
                  LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42), Model_used='LogisticRegression')
#Classification report for bag of words
LogisticR_bow_report=classification_report(testData_senti,preds_LogisticR,target_names=['Positive','Negative'])
print(LogisticR_bow_report)
#confusion matrix for bag of words
Cmatrx_bow=confusion_matrix(testData_senti,preds_LogisticR,labels=[1,0])
print(Cmatrx_bow)

DecisionTree = Classifier(cntVect_testData_review, cntVect_trainData_review,
                  testData_senti, trainData_senti,
                  tree.DecisionTreeClassifier(),Model_used='DecisionTree')
#Classification report for bag of words
DecisionTree_bow_report=classification_report(testData_senti,DecisionTree,target_names=['Positive','Negative'])
print(DecisionTree_bow_report)
#confusion matrix for bag of words
Cmatrx_bow=confusion_matrix(testData_senti,DecisionTree,labels=[1,0])
print(Cmatrx_bow)


# TFIDF train set
tfidf_transformer = TfidfTransformer().fit(cntVect_trainData_review)
train_tfidf = tfidf_transformer.transform(cntVect_trainData_review)

# apply tfidf to test set
test_tfidf = tfidf_transformer.transform(cntVect_testData_review)
 
print('Tfidf_train:',train_tfidf.shape)
print('Tfidf_test:',test_tfidf.shape)
   
# Classifies for tfidf
preds_rf2 = Classifier(test_tfidf, train_tfidf,
                  testData_senti, trainData_senti,
                  RandomForestClassifier(n_estimators = 100))
#Classification report for tfidf of words
RandomForest_tfidf_report=classification_report(testData_senti,preds_rf2,target_names=['Positive','Negative'])
print(RandomForest_tfidf_report)
#confusion matrix for tfidf of words
Cmatrx_tfidf=confusion_matrix(testData_senti,preds_rf2,labels=[1,0])
print(Cmatrx_tfidf)
preds_Naive = Classifier(test_tfidf, train_tfidf,
                  testData_senti, trainData_senti,
                  MultinomialNB(), Model_used='Naive Bayes')
#Classification report for tfidf of words
mnb_tfidf_report=classification_report(testData_senti,preds_Naive,target_names=['Positive','Negative'])
print(mnb_tfidf_report)
#confusion matrix for tfidf of words
Cmatrx_tfidf=confusion_matrix(testData_senti,preds_Naive,labels=[1,0])
print(Cmatrx_tfidf)
preds_LogisticRegression2 = Classifier(test_tfidf, train_tfidf,
                  testData_senti, trainData_senti,
                  LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42), Model_used='LogisticRegression')
#Classification report for tfidf of words
LogisticR_tfidf_report=classification_report(testData_senti,preds_LogisticRegression2,target_names=['Positive','Negative'])
print(LogisticR_tfidf_report)
#confusion matrix for tfidf of words
Cmatrx_tfidf=confusion_matrix(testData_senti,preds_LogisticRegression2,labels=[1,0])
print(Cmatrx_tfidf)

DecisionTree = Classifier(test_tfidf, train_tfidf,
                  testData_senti, trainData_senti,
                  tree.DecisionTreeClassifier(),Model_used='DecisionTree')
#Classification report for tfidf of words
DecisionTree_tfidf_report=classification_report(testData_senti,DecisionTree,target_names=['Positive','Negative'])
print(DecisionTree_tfidf_report)
#confusion matrix for tfidf of words
Cmatrx_tfidf=confusion_matrix(testData_senti,DecisionTree,labels=[1,0])
print(Cmatrx_tfidf)