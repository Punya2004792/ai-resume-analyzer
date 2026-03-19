import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# UI
st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

if uploaded_file and job_desc:
    resume_text = extract_text(uploaded_file)

    # Compare using ML
    text = [resume_text, job_desc]
    cv = CountVectorizer()
    matrix = cv.fit_transform(text)
    score = cosine_similarity(matrix)[0][1]

    st.subheader(f"Match Score: {round(score*100, 2)}%")

    # Simple skill extraction
    skills = ["python", "java", "ai", "machine learning", "data analysis"]
    found = [skill for skill in skills if skill in resume_text.lower()]
    missing = [skill for skill in skills if skill not in resume_text.lower()]

    st.write("✅ Skills Found:", found)
    st.write("❌ Missing Skills:", missing)

    if score < 0.5:
        st.warning("Improve your resume by adding more relevant skills!")
    else:
        st.success("Good match!")