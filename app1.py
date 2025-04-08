# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pymupdf  # PyMuPDF
import docx2txt
from pathlib import Path
import re
import pandas as pd
import io
import warnings
import uuid
import shutil

# Streamlit UI
st.set_page_config(page_title="RAGcruiter")
st.title("RAGcruiter 🚀 — Resume Evaluator with Gemini")

# Load environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

warnings.filterwarnings("ignore")

VECTOR_DIR = "chroma_store"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

#Extract Resume Text & Split into Chunks
def extract_resume_chunks(file):
    file_path = "temp_resume.pdf"
    with open(file_path, "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    os.remove(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_or_load_vector_store(filename, chunks):
    store_path = os.path.join(VECTOR_DIR, filename.replace(".", "_"))
    if not os.path.exists(store_path):
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embedding_model,
            persist_directory=store_path
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory=store_path,
            embedding_function=embedding_model
        )
    return vectorstore

def retrieve_matching_chunks(job_description, vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_chunks = retriever.get_relevant_documents(job_description)
    return relevant_chunks

def get_gemini_response(prompt, retrieved_chunks, job_description):
    context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
        prompt,
        f"Relevant Resume Info:\n{context}",
        f"Job Description:\n{job_description}"
    ])
    return response.text

def extract_text(file_bytes, file_type):
    if file_type == "application/pdf":
        with pymupdf.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        temp_file = io.BytesIO(file_bytes)
        return docx2txt.process(temp_file)
    elif file_type == "text/plain":
        return file_bytes.decode("utf-8")
    else:
        return ""

def extract_required_skills(job_description):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""Extract a list of key skills required for the following job description. Just list them as a comma-separated string.\n\nJob Description:\n{job_description}"""
    response = model.generate_content(prompt)
    return [skill.strip() for skill in response.text.split(",")]

def extract_skills(text, dynamic_skills_list):
    found = [skill for skill in dynamic_skills_list if re.search(rf'\\b{re.escape(skill)}\\b', text, re.IGNORECASE)]
    return list(set(found))

# Prompts
prompt_analysis = """
You are a Technical HR Recruiter. Your task is to analyze the following resume against the provided job description and highlight the key aspects in a crisp and to-the-point manner.

Highlight candidate Name.

Your analysis should be structured into the following four sections:

1.  **Profile Summary:** Provide a concise (1-2 sentences) summary of the candidate's core experience and key skills relevant to the job description.

2.  **Strengths:** List key strengths of the candidate in bullet points, focusing on direct relevance to the job description's requirements and preferences. Use specific technologies, tools, and action verbs.

3.  **Weaknesses:** Identify potential weaknesses or areas for development in bullet points. Frame these objectively and professionally, focusing on gaps compared to the job description.

4.  **Missing Important Skills:** List critical skills explicitly mentioned in the job description that are not evident in the resume in bullet points. Be specific about the missing skills.

Ensure your analysis is concise, apt, and directly related to the information provided in the job description and resume. Avoid unnecessary elaboration or subjective opinions.
"""

prompt_match = """
You are an ATS (Applicant Tracking System). Analyze the following resume against the provided job description and provide a structured output.

Highlight candidate Name.

Your output should include the following three sections:

1.  **Matching Percentage:** Calculate and return a numerical percentage representing the overall match between the resume and the job description based on keyword presence and relevance.

2.  **Missing or Unmatched Keywords:** List the keywords and key phrases explicitly mentioned in the job description that are either absent or not prominently featured (unmatched) in the resume.

3.  **Final Thoughts:** Provide a brief (1-2 sentences) overall assessment of the candidate's suitability based on the keyword analysis.
"""

job_desc = st.sidebar.text_area("Paste the Job Description here:", height=250)
uploaded_files = st.sidebar.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

resume_texts = {}
analysis_results = []
dynamic_skills = extract_required_skills(job_desc) if job_desc else []

# Prepare data for dashboard and reuse vector store
uploaded_file_map = {}
for file in uploaded_files:
    file_bytes = file.read()
    uploaded_file_map[file.name] = file  # store file object for later
    text = extract_text(file_bytes, file.type)
    resume_texts[file.name] = text
    skills = extract_skills(text, dynamic_skills)
    word_count = len(text.split())
    analysis_results.append({
        "Filename": file.name,
        "Word Count": word_count,
        "Detected Skills": ", ".join(skills)
    })
    file.seek(0)

tab1, tab2, tab3 = st.tabs(["Individual Analysis", "Dashboard", "Chat"])

# Individual analysis
with tab1:
    selected_file = st.selectbox("Select a resume for analysis:", list(resume_texts.keys()))
    if selected_file and job_desc:
        resume_text = resume_texts[selected_file]
        col1, col2,col3 = st.columns(3)
        response = ""

        chunks = extract_resume_chunks(uploaded_file_map[selected_file])
        vectorstore = create_or_load_vector_store(selected_file, chunks)
        retrieved_chunks = retrieve_matching_chunks(job_desc, vectorstore)

        # Flags for conditional input
        show_question_input = st.session_state.get("show_question_input", False)

        with col1:
            if st.button("📝 Analyze Resume"):
                with st.spinner("Analyzing..."):
                    response = get_gemini_response(prompt_analysis, retrieved_chunks, job_desc)

        with col2:
            if st.button("📊 Match Percentage"):
                with st.spinner("Evaluating..."):
                    response = get_gemini_response(prompt_match, retrieved_chunks, job_desc)
        
        # Q&A
        with col3:
            if st.button("🤔 Ask a Question"):
                st.session_state.show_question_input = True  # Toggle flag to show input

        # Show input box below columns if toggled
        if st.session_state.get("show_question_input", False):
            user_query = st.text_input("Enter your question about this resume:")
            if st.button("Submit Question"):
                if user_query:
                    with st.spinner("Getting answer..."):
                        qa_prompt = f"""
                        You are "ResumeBot," an intelligent assistant for an ATS system. You have been provided with structured data extracted from a candidate's resume. 
                        Your goal is to answer user questions about this candidate accurately and efficiently.

                        **User Question:** "{user_query}"

                        Respond to the user's question based on the provided data. Be direct and informative. 
                        If the question is ambiguous, ask for clarification. 
                        If the information is not available, state "The resume does not contain information about that." or a similar polite refusal.
                        """
                        response = get_gemini_response(qa_prompt, retrieved_chunks, job_desc)
                else:
                    st.warning("Please enter a question.")

        st.write(response)

    else:
        st.info("Please select a resume and paste a job description.")

# Dashboard tab
with tab2:
    st.subheader("📊 Resume Dashboard")
    if analysis_results:
        df = pd.DataFrame(analysis_results)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Filename")["Word Count"])
    else:
        st.info("Upload resumes to view dashboard analysis.")

# Chat tab - Placeholder


# Helper: Check if a file is already indexed
def resume_exists_in_chroma(file_name: str, persist_path: str):
    # You could use filename or file hash as the unique identifier
    existing_dirs = os.listdir(persist_path) if os.path.exists(persist_path) else []
    return any(file_name in fname for fname in existing_dirs)

# Helper: Generate a unique directory for a file
def get_resume_vector_dir(file_name: str):
    safe_name = file_name.replace(" ", "_").replace(".", "_")
    return os.path.join(VECTOR_DIR, f"resume_{safe_name}")

# In your Streamlit tab
with tab3:
    st.subheader("💬 Chat with All Resumes")    

    