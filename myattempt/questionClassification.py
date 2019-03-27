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
filePath="D:\mtech\machineLearning\GATE-Overflow-Question-Tagging-master\cleaned_dataset.csv"

df=pd.read_csv(filePath,header=None,skiprows=0,delimiter=",")
rows,columns=df.shape


#df[column][row]
#so column is 0 or 1 where 0 is question and 1 is tag


x=[]
y=[]
for i in range(1,rows):
	x.append(df[0][i])
	y.append(df[1][i])




#print(x[0])
#print(y[0])
#print(df.head())

#print(rows,columns)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#print(len(X_train))

txt_clf=Pipeline([('vect',CountVectorizer(stop_words='english',ngram_range=(1,2))),('tfidf',TfidfTransformer()),('clf',MultinomialNB(alpha=0.1))])
txt_clf.fit(X_train,y_train)

predicted=txt_clf.predict(X_test)
accuracy=accuracy_score(y_test,predicted)
print(accuracy)