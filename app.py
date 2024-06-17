import string
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
import tempfile
import os

# Download NLTK data if not already done
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Function to clean text
def preprocess_text(text):
    text = text.lower()
    text = nltk.tokenize.word_tokenize(text)
    new_text = []
    for i in text:
        if i.isalnum() and i not in stop_words and i not in string.punctuation:
            new_text.append(stemmer.stem(i))

    return " ".join(new_text)


# Function to extract text from PDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


st.title('Job Description and Resume Matcher')
job_description = st.text_area("Enter the job description")

uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True, type=['txt', 'pdf'])

if st.button("Match Resumes"):
    if job_description and uploaded_files:
        # Preprocess the job description
        preprocessed_job_description = preprocess_text(job_description)

        # Initialize list to store all resumes and original texts
        resumes = []
        resume_texts = []
        resume_titles = []

        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
                resume_title = uploaded_file.name
            else:
                resume_text = uploaded_file.read().decode('utf-8')
                resume_title = uploaded_file.name

            preprocessed_resume = preprocess_text(resume_text)
            resumes.append(preprocessed_resume)
            resume_texts.append(resume_text)
            resume_titles.append(resume_title)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        resume_vecs = vectorizer.fit_transform(resumes)
        job_vec = vectorizer.transform([preprocessed_job_description])

        # Compute cosine similarity
        similarities = cosine_similarity(job_vec, resume_vecs).flatten()

        # Get top matching resumes
        top_indices = similarities.argsort()[-5:][::-1]

        st.write("Top matching resumes:")
        for idx in top_indices:
            resume_title = resume_titles[idx]
            similarity_score = similarities[idx]

            # Save the resume to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
                tmp_file.write(resume_texts[idx].encode('utf-8'))
                tmp_file_path = tmp_file.name

            st.markdown(f"**{resume_title}** - Similarity: {similarity_score:.2f}",
                        unsafe_allow_html=True)
    else:
        st.warning("Please provide a job description and upload resumes.")
