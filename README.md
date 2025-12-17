# Bank_job_based_Loan_recommendation_System
An intelligent Loan Recommendation System that utilizes NLP (TF-IDF) and Unsupervised Machine Learning (K-Means) to cluster bank customer job profiles and provide data-driven loan amount suggestions.
#Project Overview
This project implements an intelligent system that categorizes bank customers' job roles using Unsupervised Machine Learning (Clustering) and provides personalized loan amount recommendations. By leveraging Natural Language Processing (NLP), the system cleans job titles and groups similar professional profiles together to understand workforce distributions and financial eligibility.

---
##Key Features
    Text Preprocessing Pipeline: Automated cleaning of job titles using Regex, tokenization, stop-word removal, and lemmatization.
NLP Vectorization: Converts raw text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
Unsupervised Clustering: Uses the K-Means algorithm to segment the workforce into 5 distinct professional clusters.
Dynamic Loan Recommendation: A rule-based engine that suggests specific loan amounts (e.g., $60,000 for Technicians) based on the applicant's job category.
Data Visualization: Bar charts to visualize the distribution of candidates across different professional clusters.
---
##Dataset Description

The system uses the bank.csv dataset, which contains customer demographic and financial data.

    Target Feature: job (e.g., admin, technician, management, blue-collar, etc.).

Other Features: Age, marital status, education level, balance, and housing/loan status.
---
##Tech Stack

Data Analysis: Pandas, NumPy
Machine Learning: Scikit-Learn (KMeans, TfidfVectorizer)
NLP: NLTK (Natural Language Toolkit)
Visualization: Matplotlib, Seaborn

---
##Logic Workflow

    Preprocessing: Job titles like "admin." are cleaned and lemmatized into base forms.

Feature Extraction: TF-IDF identifies the importance of words across the dataset.

Clustering: K-Means groups the jobs. For example, Cluster 0 might contain administrative roles while Cluster 1 contains technical roles.

Recommendation: The user inputs a job, and the system matches it against a pre-defined risk/income mapping to suggest a loan amount.
---

##Directory Structure

├── bank.csv                # Raw bank marketing dataset
├── recommendation_system.py # Main Python script for NLP and Clustering
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

## Example Usage: To get a recommendation:
Enter the candidate's job: technician
Output: Recommended loan amount for technician: $60000
