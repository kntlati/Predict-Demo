#!/usr/bin/env python
# coding: utf-8

# In[9]:


##Import Python Libraries
import pandas as pd
import numpy as np
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import f1_score

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


##Data Import- reading data saved on a folder (train data)
train = pd.read_csv(r'C:\Users\kntlati\Desktop\Climate Competition\train.csv')
##train.columns = [col.replace(' ','_') for col in train.columns] 
##train = train.dropna()
train.head()
##test = pd.read_csv(r'C:\Users\kntlati\Desktop\Climate Competition\test.csv')


# In[11]:


##Data Import- reading data saved on a folder (test data)
test = pd.read_csv(r'C:\Users\kntlati\Desktop\Climate Competition\test.csv')
##test.columns = [col.replace(' ','_') for col in test.columns]
test.head()


# In[12]:


train.sentiment.value_counts()


# In[13]:


##Splitting out the X variable from the target
y = train['sentiment']
X = train['message']


# In[14]:


#Converting Text into numbers
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words="english")
X_vectorized = vectorizer.fit_transform(X)


# In[15]:


##Splitting the training data into a training and validation set
X_train,X_valid,y_train,y_valid = train_test_split(X_vectorized,y,test_size=.3,shuffle=True, stratify=y, random_state=11)


# In[17]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_valid)


# In[19]:


f1_score(y_valid, rfc_pred, average="macro")


# In[21]:


testx = test['message']
test_vect = vectorizer.transform(testx)


# In[22]:


#predictions on the test set and adding a sentiment column
y_pred = rfc.predict(test_vect)
test['sentiment'] = y_pred
test.head()


# In[23]:


test[['tweetid','sentiment']].to_csv('testsubmission.csv', index=False)


# In[ ]:




