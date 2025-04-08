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
from google.api_core.exceptions import ResourceExhausted
import time
from dashboard import render_dashboard
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

if "match_percentages" not in st.session_state:
    st.session_state.match_percentages = {}

if "analysis_results" in st.session_state:
    del st.session_state["analysis_results"]
st.session_state["analysis_results"] = []

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
            #print("\n".join([page.get_text() for page in doc]))
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
    #print(list(set(found)))
    return list(set(found))

def count_matching_skills(resume_text, dynamic_skills_list):
    matched_skills = set()
    resume_lower = resume_text.lower()
    for skill in dynamic_skills_list:
        skill_lower = skill.lower()
        pattern = re.escape(skill_lower)
        if re.search(pattern, resume_lower):
            matched_skills.add(skill)
    #print(len(matched_skills), list(matched_skills))
    return len(matched_skills), list(matched_skills)
    
def get_candidate_names(resume_text):
    prompt_names=f"""Get the candidate name from this resume text. Just the candidate name.  Dont print any other words"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt_names, resume_text])  
    return response.text.strip()

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

promp_match_percentage="""Get the match percentage with the job description from this resume text. Just the match percentage digit. No need to include %.  Dont print any other words"""

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
    match_count, matching_skills = count_matching_skills(text, dynamic_skills)
    candidate_name_match=get_candidate_names(text)
    word_count = len(text.split())
    # Reuse match percent if available in session, else None
    match_percent = st.session_state.match_percentages.get(file.name, None)
    time.sleep(1.5)
    
    analysis_results.append({
        "Filename": file.name,        
        "Candidate Name": candidate_name_match.upper(),  
        "Job Match Percentage": match_percent,      
        "Word Count": word_count,
        "Detected Skills": matching_skills,
        "Skill Match Count": match_count
    })
    #print(analysis_results)
    file.seek(0)

#print(analysis_results)
#print(st.session_state["analysis_results"])

tab1, tab2 = st.tabs(["Individual Analysis", "Dashboard"])

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
                st.session_state.show_question_input = False
                with st.spinner("Analyzing..."):
                    response = get_gemini_response(prompt_analysis, retrieved_chunks, job_desc)

        with col2:
            if st.button("📊 Match Percentage"):
                st.session_state.show_question_input = False
                with st.spinner("Evaluating..."):
                    response = get_gemini_response(prompt_match, retrieved_chunks, job_desc)
                    match_percent = re.search(r'\d{1,3}%', response)
                    match_percent = match_percent.group(0) if match_percent else "0%"

                    # ✅ Update session
                    st.session_state.match_percentages[selected_file] = match_percent

                    # ✅ Update the corresponding row in analysis_results
                    for row in analysis_results:
                        if row["Filename"] == selected_file:
                            row["Job Match Percentage"] = match_percent
                            break
        
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

        # Convert % to integer for sorting if not None
        df["Parsed Match %"] = df["Job Match Percentage"].apply(lambda x: int(re.search(r'\d+', x).group(0)) if isinstance(x, str) and re.search(r'\d+', x) else 0)

        df_sorted = df.sort_values(by="Parsed Match %", ascending=False).drop(columns=["Parsed Match %"])
        #st.dataframe(df_sorted[['Filename', "Candidate Name", "Job Match Percentage", 'Word Count', 'Skill Match Count']], use_container_width=True)
        render_dashboard(df)
    else:
        st.info("Upload resumes to view dashboard analysis.")

    
