import streamlit as st
st.title('MAJOR PROJECT')
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/charan-chaitanya/Datasets/main/Test.csv')
import requests
from bs4 import BeautifulSoup 
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('stopwords')
stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')
def html_tag(text):
  soup = BeautifulSoup(text,"html.parser")
  new_text = soup.get_text()
  return new_text
import contractions
def con(text):
  expand = contractions.fix(text)
  return expand
import re 
def remove_sp(text):
  pattern = r'[^A-Za-z\s]'
  text = re.sub(pattern,'',text)
  return text
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
def remove_stopwords(text):
  tokens = tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  filter_tokens = [token for token in tokens if token not in stopwords_list]
  filter_tokens =' '.join(filter_tokens)
  return filter_tokens
df.text = df.text.apply(lambda x:x.lower())
df.text = df.text.apply(html_tag)
df.text = df.text.apply(con)
df.text = df.text.apply(remove_sp)
df.text = df.text.apply(remove_stopwords)
from vaderSentiment.vaderSentiment import  SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()
df['compound'] = df['text'].apply(lambda x: vs.polarity_scores(x)['compound'])
def compoundcondition(row):
  if (row['compound'] > 0.05):
    return 1
  if (row['compound'] < -0.05):
    return -1
  if (row['compound'] > -0.05 and row['compound'] < 0.05):
    return 0
df['new_column'] = df.apply(lambda row : compoundcondition(row),axis =1)
x = df.iloc[:,0].values
y = df.iloc[:,3].values
select = st.text_input('Enter your message')
if st.button('PRED'):
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.svm import SVC
  text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())]) 
  text_model.fit(x,y)
  op = text_model.predict([select])
  if ( op == 1 ): 
    result = "POSITIVE 😃"
  elif (op ==0 ):
    result = "NEUTRAL 😐"
  elif ( op == -1):
    result = "NEGATIVE 😔"
  st.title(result)
