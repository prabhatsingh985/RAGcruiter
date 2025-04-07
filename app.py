# app.py
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Gemini function
def get_gemini_response(prompt, resume_text, job_description):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
        prompt,
        f"Resume:\n{resume_text}",
        f"Job Description:\n{job_description}"
    ])
    return response.text

# Load resume using PyPDFLoader
def extract_resume_text(uploaded_file):
    file_path = os.path.join("temp_resume.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    all_text = "\n".join([p.page_content for p in pages])
    os.remove(file_path)  # Cleanup
    return all_text

# Prompts
prompt_analysis = """
You are a Technical HR recruiter. Analyze the given resume against the job description.
Highlight:
1. Profile summary.
2. Strengths.
3. Weaknesses.
4. Any missing important skills.
"""

prompt_match = """
You are an ATS system. Based on the resume and job description provided:
1. Return a matching percentage.
2. List missing or unmatched keywords.
3. Give brief final thoughts.
"""

# Streamlit UI
st.set_page_config(page_title="RAGcruiter")
st.title("RAGcruiter 🚀 — Resume Evaluator with Gemini")

job_desc = st.sidebar.text_area("Paste the Job Description here:", height=250)
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

tab1, tab2, tab3 = st.tabs(["Individual Analysis", "Dashboard", "Chat"])

if uploaded_file and job_desc:
    if tab1:
        resume_text = extract_resume_text(uploaded_file)
        response=""        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Analyze Resume"):
                with st.spinner("Analyzing..."):
                    response = get_gemini_response(prompt_analysis, resume_text, job_desc)
                    #st.subheader("🔍 Resume Analysis")                    

        with col2:
            if st.button("📊 Match Percentage"):
                with st.spinner("Evaluating..."):
                    response = get_gemini_response(prompt_match, resume_text, job_desc)
                    #st.subheader("📈 Match Result")                    

        st.write(response)
else:
    st.sidebar.info("Please upload a resume and paste a job description to get started.")
