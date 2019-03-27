import pandas as pd 
import numpy as np 
import re
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


import nltk

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams,FreqDist
import matplotlib.pyplot as plt 



filePath="D:\mtech\machineLearning\GATE-Overflow-Question-Tagging-master\cleaned_dataset.csv"

#opening file 
f=open(filePath,'rU')

#reading the file
raw=f.read()

#tokenzing the file according to word
tokens = nltk.word_tokenize(raw)


#tokenizing the file according to sentence
#tokens=nltk.sent_tokenize(raw)
#print(tokens[0])



# Remove single-character tokens (mostly punctuation)
tokens = [word for word in tokens if len(word) > 1]

# Remove numbers
#tokens = [word for word in tokens if not word.isnumeric()]

# NLTK's default english stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
# Lowercase all words (default_stopwords are lowercase too)
tokens = [word.lower() for word in tokens]

# Remove stopwords
tokens = [word for word in tokens if word not in default_stopwords]


# Remove stopwords like ' '' 's
tokens = [word for word in tokens if word not in ['``' ,"''","'s" ] ]

#print(len(tokens))


#print(tokens)

k=50 #sizeOFEmbedding
#finding embedding

import gensim 
from gensim.models import Word2Vec

  
# Replaces escape character with space 
f = raw.replace("\n", " ") 
  
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = []   
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = k, window = 5) 
  
# Print results 

#print(model1.wv['turing'])

 


frequency={}
for i in tokens:
    if i not in frequency:
        frequency[i]=0

vocab=[]
for key,value in frequency.items():
	vocab.append(key);


x=[]

# for i in vocab:
# 	print(i)

Word2VecDic={}
for i in vocab:
	Word2VecDic[i]=model1.wv[i]
	#print(i,model1.wv[i])

# print(Word2VecDic['limited'])
# print(Word2VecDic['secret'])
# sent=[]
# for i in range(0,5):		
# 		sent.append(Word2VecDic['secret'][i]+Word2VecDic['limited'][i])
# print(sent)

# i=0
# for k,v in Word2VecDic.items():
# 	i=i+1
# 	print(k,v)
# 	if i>10:
# 		break

 #finding sentence vector
df=pd.read_csv(filePath,header=None,skiprows=0,delimiter=",")
rows,columns=df.shape

#df[column][row]
#so column is 0 or 1 where 0 is question and 1 is tag

x=[]
y=[]
for i in range(1,rows):
	x.append(df[0][i])
	y.append(df[1][i])



ques=[]
for sent in x:
	a=[]
	l=1
	for i in range(0,k):
		a.append(0.0)
	for word in word_tokenize(sent):
		if word in vocab:
				#word is in word2vec hence there is vector of size 5 for that word
				l=l+1
				for i in range(0,k):
					a[i]=a[i]+Word2VecDic[word][i]
	#b=np.array(a)
	#b.mean(axis=0)
	for i in range(0,k):
					a[i]=(a[i])*1.0/l
	ques.append(a)


#now every question is represented by vector of size 5


# j=0
# for i in ques:
# 	print(i)
# 	j=j+1
# 	if j>5:
# 		break





X_train, X_test, y_train, y_test = train_test_split(ques, y, test_size=0.33, random_state=42)


# #print(len(X_train))

# #txt_clf=Pipeline([('vect',CountVectorizer(stop_words='english',ngram_range=(1,2))),('tfidf',TfidfTransformer()),('clf',MultinomialNB(alpha=0.1))])

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn import svm


# clf = svm.SVC()
# clf.fit(X_train, y_train)

# from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, ), random_state=1)
# clf.fit(X_train, y_train)                         
# predict=clf.predict(X_test)

# j=0
# for i in X_train:
# 	print(y[j],predict[j])
# 	j=j+1
# 	if j>k:
# 		break
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(n_jobs=1, C=1e5)
clf.fit(X_train, y_train)
predict=clf.predict(X_test)


print(accuracy_score(y_test, predict))