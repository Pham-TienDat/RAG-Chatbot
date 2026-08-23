# 🩺 Medical Chatbot — Medical Q&A Chatbot (RAG)

A **medical question-answering chatbot** built with **RAG (Retrieval-Augmented Generation)**. Instead of relying on the model's general knowledge, the chatbot answers questions based on content retrieved from the **Gale Encyclopedia of Medicine** (PDF), keeping responses grounded in the source document.

- **UI**: Streamlit (chat interface)
- **LLM**: ChatGroq — model `openai/gpt-oss-120b`
- **Embeddings**: HuggingFace Embeddings — model `BAAI/bge-m3`
- **Vector store**: FAISS (`faiss-cpu`)
- **Gen AI framework**: LangChain / LangChain-Classic / LangSmith

---

## ✨ Features

- 🔎 **Document-grounded answers** via RAG over a medical encyclopedia PDF (no hallucinated answers outside the source).
- 💬 **Interactive chat UI** built with Streamlit, with full conversation history.
- 🧠 **Powerful multilingual embeddings** (`BAAI/bge-m3`) — supports English, Vietnamese and more.
- ⚡ **Fast responses** backed by the Groq inference API.
- 🗂 **Easily extensible**: drop new PDFs into `data/` and re-run the embedding script to rebuild the knowledge base.

---

## 🧱 Tech Stack

| Component            | Technology                                           |
| -------------------- | ---------------------------------------------------- |
| Language             | Python ≥ 3.12                                          |
| UI                   | Streamlit (1.58.0)                                    |
| LLM                | Groq — `openai/gpt-oss-120b` (langchain-groq)      |
| Embeddings           | `BAAI/bge-m3` via `langchain-huggingface`            |
| Vector store         | FAISS (`faiss-cpu`)                                        |
| Document loader      | `PyPDFLoader` / `DirectoryLoader` (langchain-community) |
| Text splitter        | `RecursiveCharacterTextSplitter`                        |
| Prompt management    | LangSmith (`langchain-ai/retrieval-qa-chat`)          |
| Package manager      | `uv` (or `pip` with `requirements.txt`)             |

---

## 📁 Project Structure

```
Medical Chatbot/
├── medical_chatbot.py          # Streamlit app — the main entry point (RAG chatbot)
├── embedded.py                 # Preprocessing script: PDF → chunks → embeddings → FAISS
├── data/                       # Source PDFs (medical encyclopedia)
│   └── Gale Encyclopedia of Medicine Vol. 2 (C-F).pdf
├── vectorstore/
│   └── db_faiss/               # FAISS vector store (index.faiss + index.pkl)
├── src/medical_chatbot/        # Python package (entry point declared in pyproject.toml)
├── pyproject.toml              # Project config & dependencies (uv)
├── requirements.txt            # Dependencies for pip
├── uv.lock                     # Locked dependency versions (uv)
├── .env                        # Environment variables — holds GROQ_API_KEY ⚠️
├── .python-version             # Required Python version (3.12)
└── .gitignore
```

## ⚙️ Installation

### 1. Prerequisites — Python ≥ 3.12

```bash
python --version   # should print >= 3.12
```

### 2. Install dependencies

**With `uv` (recommended):**

```bash
uv venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate     # Windows
uv sync
```

**With `pip`:**

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

> 💡 `pyproject.toml` and `requirements.txt` both list the complete set of dependencies (LangChain, Streamlit, FAISS, PyTorch, etc.).

### 3. Configure the API key (required)

1. Create an account at [console.groq.com](https://console.groq.com) and generate an **API Key**.
2. Create a `.env` file in the project root:

```
GROQ_API_KEY=your-api-key-here
```

> ⚠️ **Security**: Do **NOT** commit the `.env` file to Git — it contains a secret key. If the key is ever exposed, **revoke / rotate it** immediately in the Groq console.

---

## 🚀 Usage

### Step 1 — (One-time) Build the vector store from your PDFs

Place your medical PDF file(s) in the `data/` directory, then run:

```bash
python embedded.py
```

This script will:

1. 📄 Load all PDFs in `data/` with `PyPDFLoader`.
2. ✂️ Split the text into chunks: **chunk_size = 500**, **chunk_overlap = 50**.
3. 🧬 Generate embeddings with the `BAAI/bge-m3` model.
4. 💾 Save the index to `vectorstore/db_faiss/` (as `index.faiss` and `index.pkl`).

> 💡 Whenever you add **new** documents, re-run this step to update the vector store.

### Step 2 — Run the Streamlit app

```bash
streamlit run medical_chatbot.py
```

Or, if the package was installed via `uv sync`, you can also use:

```bash
medical-chatbot          # entry point declared in pyproject.toml
```

Open **http://localhost:8501** in your browser and start asking questions! 🎉

---

## 🧠 How It Works (RAG Pipeline)

```
User question
       │
       ▼
┌─────────────────────────────┐
│  Streamlit UI               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ FAISS vector store (bge-m3) │  → retriever fetches top-3 relevant chunks (k = 3)
└─────────────┬───────────────┘
              │ context
              ▼
┌──────────────────────────────────┐
│ LLM: Groq (openai/gpt-oss-120b)  │  → prompt "retrieval-qa-chat" (from LangSmith)
│ combine_docs_chain + retrieval_chain (langchain-classic)
└─────────────┬────────────────────┘
              │
              ▼
   Answer grounded in the source document
```

### Configurable parameters (in the source code)

| Parameter             | Value                      | Location                            |
| --------------------- | -------------------------- | ----------------------------------- |
| Chunk size            | `500`                      | `embedded.py`                       |
| Chunk overlap         | `50`                       | `embedded.py`                       |
| Embedding model       | `BAAI/bge-m3`              | `embedded.py` / `medical_chatbot.py` |
| LLM model (Groq)      | `openai/gpt-oss-120b`      | `medical_chatbot.py`                |
| Temperature           | `0.5`                      | `medical_chatbot.py`                |
| Max tokens            | `1024`                     | `medical_chatbot.py`                |
| Retrieval top-k (k)   | `3`                        | `medical_chatbot.py`                |

---

## 🛠 Troubleshooting

| Error | Cause / Fix |
| ----- | ----------- |
| `Index could not be loaded` | `embedded.py` has not been run yet, or `vectorstore/db_faiss` is missing → re-run Step 1. |
| `GROQ_API_KEY` auth errors | Missing or wrong key in `.env`. Check your key in the Groq console. |
| LangSmith prompt pull error | Internet access is required; the prompt is pulled from the LangSmith Hub (`langchain-ai/retrieval-qa-chat`). |
| FAISS import error | Reinstall with: `pip install faiss-cpu`. |
| `ModuleNotFoundError` | Dependencies not installed or wrong venv active → run `uv sync` / `pip install -r requirements.txt` again. |

---

## 🔐 Security

- Never share or commit the `.env` file / API key.
- The key is read from `.env` via `load_dotenv()` — keep it that way and avoid hardcoding secrets.
- Make sure `.env` is listed in `.gitignore`, and **rotate the key** immediately if it was ever pushed to Git.

---

## 📌 Disclaimer

⚠️ This application is **for research, educational and reference purposes only — it is not a medical device**. Do not use the chatbot's answers to diagnose or treat any medical condition. Always consult a qualified physician or healthcare professional.

---

## 👨‍💻 Author

- **Pham-TienDat** — [GitHub](https://github.com/Pham-TienDat)