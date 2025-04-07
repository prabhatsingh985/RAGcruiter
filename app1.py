# app.py
import os
import streamlit as st
#from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Streamlit UI
st.set_page_config(page_title="RAGcruiter")
st.title("RAGcruiter 🚀 — Resume Evaluator with Gemini")

# Load environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Extract Resume Text & Split into Chunks
def extract_resume_chunks(uploaded_file):
    if uploaded_file is None or uploaded_file.size == 0:
        st.warning("Uploaded file is empty or not valid.")
        return []

    file_path = os.path.join("temp_resume.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        os.remove(file_path)
        return []

    os.remove(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="chroma_store")
    vectorstore.persist()
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



job_desc = st.sidebar.text_area("Paste the Job Description here:", height=250)
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

tab1, tab2, tab3 = st.tabs(["Individual Analysis", "Dashboard", "Chat"])

if uploaded_file and job_desc:
    if tab1:
        #resume_text = extract_resume_chunks(uploaded_file)
        response=""        
        col1, col2 = st.columns(2)
        resume_chunks = extract_resume_chunks(uploaded_file)
        if not resume_chunks:
            st.warning("No valid content found in the uploaded PDF.")
            st.stop()

        vectorstore = create_vector_store(resume_chunks)
        retrieved_chunks = retrieve_matching_chunks(job_desc, vectorstore)

        with col1:
            if st.button("📝 Analyze Resume"):
                with st.spinner("Analyzing..."):
                    response = get_gemini_response(prompt_analysis, retrieved_chunks, job_desc)
                    #response = get_gemini_response(prompt_analysis, resume_text, job_desc)
                    #st.subheader("🔍 Resume Analysis")                    

        with col2:
            if st.button("📊 Match Percentage"):
                with st.spinner("Evaluating..."):
                    response = get_gemini_response(prompt_match, retrieved_chunks, job_desc)
                    #st.subheader("📈 Match Result")                    

        st.write(response)
else:
    st.sidebar.info("Please upload a resume and paste a job description to get started.")
