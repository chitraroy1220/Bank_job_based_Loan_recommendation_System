#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

#import standard visualization
import matplotlib.pyplot as plt
import seaborn as sns

#import some necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

from sklearn.model_selection import train_test_split #split
from sklearn.metrics import accuracy_score, confusion_matrix #metrics
from sklearn.preprocessing import LabelEncoder #converting text into numerical data


# In[2]:


#Load dataset
df = pd.read_csv("bank.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


#preprocess the text data by removing the conjuction and prepostion words or hepling verbs
stop_words = set(stopwords.words('english'))
#to use base form or root form of word we use lematized the word
lematizer = WordNetLemmatizer()

def preprocess_text(job):
    #conver to lowercase
    job = str(job)
    job = job.lower()
    
    
    #remove punctuation marks by space
    job = re.sub(r'[^a-zA-z\s]',' ',job)
    #remove alphanumerical or whitespace
    job = re.sub(r'[^\w\s]',' ',job)
    
    #tokenize the text after removing unwanted things
    tokens=nltk.word_tokenize(job)
    
    #Remove stopwords
    tokens= [token for token in tokens if token not in stop_words]
    
    #lemmatize tokens
    tokens =[lematizer.lemmatize(token) for token in tokens]
    
    #Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

#apply preprocess_text to the dataframe
df['preprocessed_job_title'] = df['job'].apply(preprocess_text)
#printing the preprocessed text
print(df[['preprocessed_job_title']])


# In[6]:


# create a TF-IDF vectorizer to convert text to numerical features 
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(df['job'])


# In[7]:


#Perform clustering using K-means
k = 5
kmeans = KMeans(n_clusters=k,random_state=42)
kmeans.fit(x)


# In[8]:


#assigne cluster labels to articles
df['cluster_label'] =kmeans.labels_


# In[9]:


#print the articles in each cluster
for cluster in range(k):
    cluster_articles = df[df['cluster_label']==cluster]['job']
    print(f'Cluster {cluster}:')
    print(cluster_articles)
    print('\n')
          


# In[10]:


# Count the number of jobs in each cluster
cluster_counts = df['cluster_label'].value_counts().sort_index()

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(cluster_counts.index, cluster_counts.values, tick_label=[f'Cluster {i}' for i in cluster_counts.index])
plt.title('Cluster Visualization')
plt.xlabel('Cluster')
plt.ylabel('Number of Jobs')
plt.show()


# In[ ]:


# Sample job categories and associated loan amounts
job_loan_mapping = {
    "admin": 50000,
    "technician": 60000,
    "services": 30000,
    "management": 35000,
    # ... more job categories ...
}

# Sample candidate's job
candidate_job = input("Enter the candidate's job: ")

# Function to recommend a loan based on the candidate's job
def recommend_loan(job):
    if job in job_loan_mapping:
        loan_amount = job_loan_mapping[job]
        return f"Recommended loan amount for {job}: ${loan_amount}"
    else:
        return "No loan recommendation available for this job."

# Get loan recommendation based on candidate's job
loan_recommendation = recommend_loan(candidate_job)

# Print loan recommendation
print(loan_recommendation)


# In[ ]:




