# --- README.md ---

# 🧠 SmartClause AI

SmartClause AI is a lightweight legal contract assistant built using `DistilBERT` for clause classification and a `MiniLM` embedding model with a HuggingFace-hosted LLM for Retrieval-Augmented Generation (RAG). This project allows users to upload a legal contract (PDF or DOCX), extract and classify its key clauses, and ask questions about the content via natural language.

---

## 🚀 Features

- 🔍 Clause Classification using DistilBERT (fine-tuned on LexGLUE datasets)
- 🤖 Question Answering using RAG and MiniLM + LLM
- 🧾 Contract Upload Support (.pdf and .docx)
- 🖥️ Streamlit UI for quick testing
- 🔒 Hugging Face Token integration with `.env`

---

## 🧰 Tech Stack

- Python 3.10
- Transformers (HuggingFace)
- LangChain (with langchain_community)
- Sentence Transformers (MiniLM)
- FAISS for vector search
- Streamlit for frontend
- CUAD / LexGLUE Datasets

---

## 🧪 Installation & Setup

### 📦 1. Clone the Repo
```bash
git clone https://github.com/yourusername/smartclause-ai.git
cd smartclause-ai
```

### 🐍 2. Create and Activate Conda Environment
```bash
conda create --name smartclause python=3.10
conda activate smartclause
```

### 📥 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔐 4. Setup `.env`
Create a `.env` file in your root directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

---

## 🏁 Usage

### ▶️ 1. Run the Streamlit App
```bash
streamlit run app.py
```
- Upload a `.pdf` or `.docx` file
- Ask a question like: "What are the termination terms?"

### 🎓 2. Fine-tune DistilBERT on LexGLUE
```bash
python finetune_model.py
```
- Uses `case_hold`, `ecthr_a`, and `ecthr_b` datasets from LexGLUE
- Tokenization uses `context + endings`
- Classification handled with DistilBERT in multiple-choice mode

---

## 🗃️ Project Structure
```
smartclause-ai/
├── app.py                  # Streamlit app
├── clause_classifier.py   # Zero-shot classification
├── rag_engine.py          # RAG setup (MiniLM + FAISS + LLM)
├── finetune_model.py      # DistilBERT training on LexGLUE
├── requirements.txt       # Python dependencies
├── .env                   # HuggingFace API token
└── README.md              # Project overview
```

---

## 📌 Notes
- Use `distilbert-base-uncased` for fast training on CPU/Apple M2.
- Uses `sentence-transformers/all-MiniLM-L6-v2` for lightweight embedding.
- Ensure `.env` is not tracked by Git.

---

## 💡 Future Ideas
- Add clause risk scoring
- Extend to CUAD dataset for deeper legal extraction
- Deploy as a SaaS platform

---

## 📄 License
MIT License © 2025 Jidhnyasa Mahajan
