# 🧠 RAGcruiter — AI-Powered Resume Evaluator with Gemini + ChromaDB

**RAGcruiter** is an intelligent resume evaluator and recruiter dashboard that leverages **Generative AI (Gemini)** and **Semantic Search (ChromaDB)** to help HR teams:

- 📄 Analyze uploaded resumes
- 📌 Match candidates to job descriptions using AI
- 📊 Visualize resume statistics in a smart dashboard
- 💬 Interact with resumes using natural language queries

---

## 🚀 Features

✅ Upload multiple resumes (PDF, DOCX, TXT)  
✅ Extract and chunk resume text for semantic matching  
✅ Analyze candidate fit against any Job Description  
✅ Get AI-generated analysis and match percentage  
✅ Cache results into SQLite and ChromaDB for fast re-access  
✅ Ask candidate-specific questions (RAG-based Q&A)  
✅ Clear all cached resume data in one click  

---

## 🛠 Tech Stack

| Layer           | Tech Used                                |
|----------------|-------------------------------------------|
| 🧠 LLM          | Gemini 1.5 Flash (Google Generative AI)   |
| 📦 Vector Store | ChromaDB                                  |
| 📊 Embeddings   | HuggingFace MiniLM-L6-v2                  |
| 🔍 NLP Framework| LangChain                                 |
| 🧾 Text Parsing | PyMuPDF, docx2txt                         |
| 🗃️ DB Cache     | SQLite3                                    |
| 🌐 Frontend     | Streamlit                                 |
| 📈 Visualization| Matplotlib, Seaborn                       |

---

## 🧬 Folder Structure
.
├── app.py
├── dashboard.py
├── temp_files/
│   └── ... (uploaded resumes)
├── chroma_store1/
│   └── ... (ChromaDB persistent storage)
├── resume_data.db
├── requirements.txt
├── .env
└── README.md


---
## 🔐 Environment Setup
Create a `.env` file (local only):
GOOGLE_API_KEY=your_gemini_api_key_here


---

## 🧪 Running Locally
```bash
1. **Clone the repo**   
   git clone https://github.com/yourusername/ragcruiter.git
   cd ragcruiter
2. **Install dependencies**
   pip install -r requirements.txt
3. **Run the app**
   streamlit run app.py
   ```

## Sample Dashboard
=====================

📊 Match Percentage by Candidate

Top matched resumes
Word count & skill distribution
Visualizations:
Donut charts
Bar graphs
Data tables


Clearing Cache
================

🧼 Clear Resume Data Cache

Click the 🗑️ Clear Resume Data Cache button in the sidebar to:

Delete saved files
Clear vector stores
Reset the database


## Flow Diagram

                          ┌──────────────────────────┐
                          │    📁 Upload Resumes     │
                          └──────────┬───────────────┘
                                     ▼
                     ┌────────────────────────────────────┐
                     │ 📝 Save Locally in /temp_files      │
                     └──────────┬─────────────────────────┘
                                ▼
              ┌──────────────────────────────────────────────┐
              │ 📄 Extract Text (PyMuPDF / docx2txt / .txt)   │
              └──────────┬────────────────────────────────────┘
                         ▼
              ┌────────────────────────────────────┐
              │ 🔍 Extract Candidate Name (Gemini)  │
              └──────────┬─────────────────────────┘
                         ▼
              ┌────────────────────────────────────┐
              │ 🧠 Generate Embeddings (HF Model)   │
              └──────────┬─────────────────────────┘
                         ▼
              ┌────────────────────────────────────┐
              │ 📦 Store in Chroma Vector Database  │
              └──────────┬─────────────────────────┘
                         ▼
              ┌────────────────────────────────────────────┐
              │  📜 Get Job Description Input from Sidebar  │
              └──────────┬─────────────────────────────────┘
                         ▼
              ┌──────────────────────────────────────────────┐
              │ 🛠️  Extract Required Skills (Gemini Prompt)   │
              └──────────┬────────────────────────────────────┘
                         ▼
              ┌──────────────────────────────────────────────┐
              │ 🔍 Retrieve Top-k Relevant Chunks from Chroma │
              └──────────┬────────────────────────────────────┘
                         ▼
         ┌────────────────────────────────────────────────────────┐
         │ 🤖 Feed Chunks + JD into Gemini for Match Evaluation   │
         └──────────┬─────────────────────────────────────────────┘
                    ▼
         ┌────────────────────────────────────────────┐
         │ 📊 Get: Match %, Skill Gaps, Analysis Notes │
         └──────────┬─────────────────────────────────┘
                    ▼
         ┌────────────────────────────────────┐
         │ 💾 Cache Results into SQLite DB    │
         └──────────┬─────────────────────────┘
                    ▼
         ┌──────────────────────────────────────┐
         │ 📈 Display Dashboard (Match %, Charts)│
         └──────────────────────────────────────┘


## 🔮 Future Roadmap
✅ Export top-matched resumes as Excel/CSV

✅ Detailed feedback section from Gemini

🔒 Login system for recruiters

💼 Multi-job JD evaluation

☁️ Optional OpenAI fallback


