import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ATS System", layout="wide")

st.title("Resume Screening System")

# -----------------------------
# SIDEBAR 
# -----------------------------
st.sidebar.title(" About")
st.sidebar.write("""
This ATS system evaluates resumes against a job description using NLP techniques.
It helps automate candidate screening and ranking.
""")

st.sidebar.title(" How It Works")
st.sidebar.write("""
1. Upload multiple resumes (PDF)  
2. Enter Job Description  
3. Text is extracted  
4. TF-IDF vectorization applied  
5. Cosine similarity calculated  
6. Candidates are ranked  
7. Final decision based on threshold  
""")

st.sidebar.title(" Features")
st.sidebar.write("""
- Multi Resume Screening  
- Duplicate Detection  
- Ranking System  
- Match Level Analysis  
- Threshold-based Shortlisting  
- Download Report  
""")

THRESHOLD = st.sidebar.slider("Screening Threshold (%)", 50, 90, 70)

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

# -----------------------------
# INPUT
# -----------------------------
jd = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------
# PROCESS
# -----------------------------
if st.button("Run Evaluation"):

    if not jd or not uploaded_files:
        st.warning("Please provide input")
    else:

        resumes = []
        names = []
        seen = set()

        for file in uploaded_files:
            text = extract_text(file)

            if text in seen:
                continue
            seen.add(text)

            resumes.append(text)
            names.append(file.name)

        docs = [jd.lower()] + resumes

        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(docs)

        jd_vector = vectors[0]
        resume_vectors = vectors[1:]

        scores = cosine_similarity(jd_vector, resume_vectors)[0]

        results = []

        for i in range(len(names)):

            score = round(scores[i] * 100, 2)

            # MATCH LEVEL (analysis only)
            if score >= 80:
                level = "High Match"
            elif score >= 60:
                level = "Medium Match"
            else:
                level = "Low Match"

            # FINAL DECISION (only rule)
            status = "Shortlisted" if score >= THRESHOLD else "Rejected"

            results.append([names[i], score, level, status])

        results = sorted(results, key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(results, columns=[
            "Candidate",
            "Match Score (%)",
            "Match Level",
            "Decision"
        ])

        st.subheader("Evaluation Dashboard")
        st.dataframe(df, use_container_width=True)

        st.subheader("Summary")
        st.success(f"Shortlisted: {len(df[df['Decision']=='Shortlisted'])}")
        st.error(f"Rejected: {len(df[df['Decision']=='Rejected'])}")
        st.info(f"Average Score: {round(df['Match Score (%)'].mean(), 2)}%")

        st.subheader("Top Candidates")
        st.dataframe(df.head(3))

        st.download_button(
            "Download Report",
            df.to_csv(index=False),
            "ats_report.csv",
            "text/csv"
        )