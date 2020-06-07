import pandas as pd
df_yelp=pd.read_table('yelp_labelled.txt')
df_imdb=pd.read_table('imdb_labelled.txt')
df_amz=pd.read_table('amazon_cells_labelled.txt')
frames=[df_yelp,df_imdb,df_amz]
#print(df_yelp.columns)
for colname in frames:
    colname.columns=["Message","Target"]
for colname in frames:
    print(colname.columns)
keys=['Yelp','IMDB','Amazon']
df=pd.concat(frames,keys=keys)
print(df.shape)
print(df.head())

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp=spacy.load('en_core_web_sm')
stopwords=list(STOP_WORDS)
#Getting Lemma and Stop words
#myfile=open("4.Events.txt").read()
docx=nlp("This is how John Walker was walking. He was also running  beside the lawn")
#Lemmatizing of tokens
for word in docx:
    print(word.text,"Lemma =>",word.lemma_)
#Lemma that are not pronouns
for word in docx:
    if word.lemma_!="-PRON-":
        print(word.lemma_.lower().strip())
#List Comprehensions of our Lemma
print([word.lemma_.lower().strip() if word.lemma_!="-PRON-"else word.lower_ for word in docx])
#Filtering out Stopwords and Punctuations
for word in docx:
    if word.is_stop==False and not word.is_punct:
        print(word)
import string
punctuations=string.punctuation
from spacy.lang.en import English
parser=English()
def spacy_tokenizer(sentence):
    mytokens=parser(sentence)
    mytokens=[word.lemma_.lower().strip() if word.lemma_!="-PRON-"else word.lower_ for word in mytokens]
    mytokens=[word for word in mytokens if word not in stopwords and word not in punctuations]
    return mytokens
#Machine Learning With SKlearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
class predictors(TransformerMixin):
    def transform(self,X,**transform_params):
        return [clean_text(text) for text in X]
    def fit(self,X,y=None,**fit_params):
        return self
    def get_params(self,deep=True):
        return {}
#Basic function to clean the text
def clean_text(text):
    return text.strip().lower()
#Vectorization
vectorizer=CountVectorizer(tokenizer=spacy_tokenizer,ngram_range=(1,1))
classifier=LinearSVC()
#Using Tfidf
tfvectorizer=TfidfVectorizer(tokenizer = spacy_tokenizer)
#Splitting Data Set
from sklearn.model_selection import train_test_split
#Feautures and label
X=df['Message']
ylabels=df['Target']
X_train,X_test,y_train,y_test=train_test_split(X,ylabels,test_size=0.2,random_state=42)
# Create the pipeline to clean,Tokenize,vectorize and classify
pipe=Pipeline([("cleaner",predictors()),
               ("vectorizer",vectorizer),
               ("classifier",classifier)])
#Fit our data
pipe.fit(X_train,y_train)
#Predicting with a test dataset
sample_prediction=pipe.predict(X_test)
#Prediction Results
#1 = Postive review
#0 = Negativev review
for (sample,pred) in zip(X_test,sample_prediction):
    print(sample,"Prediction",pred)
#Accuracy
print("Accuracy: ",pipe.score(X_test,y_test))
print("Accuracy: ",pipe.score(X_test,sample_prediction))
