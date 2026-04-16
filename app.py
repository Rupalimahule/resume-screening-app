import streamlit as st
import matplotlib.pyplot as plt
from sklearn .feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt',quiet=True)
from nltk import pos_tag


nltk.download('punkt')
nltk.download ('stopwords')
nltk.download('averaged_perceptron_tagger')

#page setup

st.set_page_config(page_title="Resume Job Match Scorer",page_icon="🗎",layout="wide")

st.markdown("""
Upload your resume (PDF) and paste a job description to see how well they match.
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirement.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Measures how your resume matches a job description
    - Identify important job keywords
    - Improve your resume based on missing terms
    """)
    st.header("How it works")
    st.write("""
    This tool helps you:
    - Upload your resume(PDF)
    - Paste job description
    - Click **Analyze Match**
    - Review score & suggetion
    """)

 #Functions

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader=PyPDF2.PdfReader(uploaded_file)
        text=""
        for page in pdf_reader.pages:
            text=text+page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF:{e}")
        return ""  

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text) 
    text=re.sub(r'\s+',' ',text).strip()
    return text              


def remove_stopwords(text):
    try:
        stop_words=set(stopwords.words('english'))
        words=word_tokenize(text)
    except:
        words = text.split()
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text,job_description):
    resume_processed=remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer=TfidfVectorizer()
    tfidf_matrix=vectorizer.fit_transform([resume_processed,job_processed])
    score=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]*100
    return round(score,2),resume_processed,job_processed


def extract_keywords(text,num_keywords=10):
    words=nltk.word_tokenize(text)
    words=[w for w in words if len(w)>2]
    tagged_words=pos_tag(words)
    nouns=[w for w, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
    word_freq=Counter(nouns)
    return word_freq.most_common(num_keywords)


#main app

def main():
    uploaded_file=st.file_uploader("upload your resume (PDF)",type=['pdf'])
    job_description=st.text_area("paste the job description",height=200)

    if st.button("analyze match"):
        if not uploaded_file:
            st.warning("please upload your resume")
            return
        if not job_description:
            st.warning("please paste job description")
            return

        with st.spinner("analyzing your resume"):
            resume_text=extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("could not extract text from pdf. please try another pdf")
                return

            #calculate similarity
            similarity_score,resume_processed,job_processed=calculate_similarity (resume_text,job_description)

            #result
            st.subheader("Result")
            st.metric("match score",f"{similarity_score:.2f}%")

            #gauge chart
            fig,ax=plt.subplots(figsize=(6,0.2))
            colors=['#ff4b4b','#ffa726','#0f9d58']
            color_index=min(int(similarity_score/33),2)
            ax.barh([0],[similarity_score],color=colors[color_index])
            ax.set_xlabel("match percentage")
            ax.set_yticks([])
            ax.set_title("Resume screening system")
            st.pyplot(fig)

            if similarity_score<40:
                st.warning("Low match,consider tailoring your resume more closely.")
            elif similarity_score<70:
                st.info("Good match. your resume aligns fairly well")
            else:
                st.success ("Excellent match. your resume strongly aligns")    

if __name__ =="__main__":
    main()                


