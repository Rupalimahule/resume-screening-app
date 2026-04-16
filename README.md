# AI Resume Screening Web App

This is an AI-powered Resume Screening Web Application built using Python, Streamlit, and Natural Language Processing (NLP).

# Features
- Upload resume in PDF format
- Enter job description
- Analyze resume-job match using TF-IDF and Cosine Similarity
- Display match score in percentage
- Visualize results using a bar chart
- Provide feedback based on match score (Low, Good, Excellent)

# Technologies Used
- Python
- Streamlit
- Scikit-learn (TF-IDF, Cosine Similarity)
- NLTK (text processing)
- PyPDF2 (PDF extraction)
- Matplotlib

# How to Run Locally
1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   streamlit run app.py

# Live Demo
[Click here to view the app](https://resume-screening-app-grjsydgrueicfteosatjzu.streamlit.app/)

# Project Description
This application extracts text from resumes and compares it with job descriptions using NLP techniques like text cleaning, TF-IDF, and cosine similarity to generate a match score.

# Deployment
Deployed on Streamlit Cloud
