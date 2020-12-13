#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np 
import pandas as pd 


import os


# In[9]:


train = pd.read_csv(r'C:\Users\deepa\Downloads\train_E6oV3lV.csv\train_E6oV3lV.csv')
test = pd.read_csv(r'C:\Users\deepa\Downloads\test_tweets_anuFYb8.csv\test_tweets_anuFYb8.csv')


# In[10]:


train.head()


# In[11]:


test.head()


# In[12]:


train['label'] = train['label'].astype('category')


# In[13]:


train.info()


# In[14]:


from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# In[15]:


import nltk
nltk.download('wordnet')


# In[16]:


train['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train['tweet']]
test['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in test['tweet']]


# In[17]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train['text_lem'],train['label'])


# In[18]:





# In[19]:


vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)


# In[20]:


vect_transformed_X_train = vect.transform(X_train)
vect_transformed_X_test = vect.transform(X_test)


# In[21]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


# In[29]:


modelSVC = SVC(C=100).fit(vect_transformed_X_train,y_train)


# In[30]:


predictionsSVC = modelSVC.predict(vect_transformed_X_test)
sum(predictionsSVC==1),len(y_test),f1_score(y_test,predictionsSVC)


# In[32]:


modelLR = LogisticRegression(C=100).fit(vect_transformed_X_train,y_train


# In[33]:


predictionsLR = modelLR.predict(vect_transformed_X_test)
sum(predictionsLR==1),len(y_test),f1_score(y_test,predictionsLR)


# In[34]:


modelNB = MultinomialNB(alpha=1.7).fit(vect_transformed_X_train,y_train)


# In[35]:


predictionsNB = modelNB.predict(vect_transformed_X_test)
sum(predictionsNB==1),len(y_test),f1_score(y_test,predictionsNB)


# In[36]:


modelRF = RandomForestClassifier(n_estimators=20).fit(vect_transformed_X_train,y_train)


# In[37]:


predictionsRF = modelRF.predict(vect_transformed_X_test)
sum(predictionsRF==1),len(y_test),f1_score(y_test,predictionsRF)


# In[39]:


modelSGD = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3).fit(vect_transformed_X_train,y_train)


# In[40]:


predictionsSGD = modelSGD.predict(vect_transformed_X_test)
sum(predictionsSGD==1),len(y_test),f1_score(y_test,predictionsSGD)


# In[41]:


vect = TfidfVectorizer(ngram_range = (1,4)).fit(train['text_lem'])
vect_transformed_train = vect.transform(train['text_lem'])
vect_transformed_test = vect.transform(test['text_lem'])


# In[42]:


FinalModel = LogisticRegression(C=100).fit(vect_transformed_train,train['label'])


# In[43]:


predictions = FinalModel.predict(vect_transformed_test)


# In[44]:


submission = pd.DataFrame({'id':test['id'],'label':predictions})


# In[45]:


file_name = 'test_predictions.csv'
submission.to_csv(file_name,index=False)


# In[ ]:




