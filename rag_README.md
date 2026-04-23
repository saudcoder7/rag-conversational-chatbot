# 🤖 RAG Conversational Chatbot

> **Task 2 | AI Internship Portfolio Project**
> A context-aware chatbot using LangChain, FAISS vector store,
> and Retrieval-Augmented Generation (RAG) with Streamlit UI.

---

## 📌 About This Project

This is **Task 2** of my AI Internship at CodeAlpha.

The goal is to build a conversational chatbot that remembers
context across turns and retrieves accurate answers from a
vectorized document store using the RAG architecture.

---

## 📂 Project Structure

```
rag-chatbot/
│
├── rag_chatbot.py        # Core chatbot + demo (runs offline)
├── rag_pipeline.py       # Full LangChain + FAISS RAG pipeline
├── streamlit_app.py      # Streamlit web interface
├── requirements.txt      # All dependencies
├── docs/                 # Put your documents here (.txt, .pdf)
│   ├── ai_overview.txt
│   ├── python_guide.txt
│   └── langchain_docs.txt
├── vectorstore/          # FAISS index (auto-generated)
└── README.md             # Project documentation
```

---

## 🎯 Objective

Build a RAG chatbot that:
- Loads and vectorizes a custom document corpus
- Retrieves relevant chunks using FAISS similarity search
- Generates grounded answers using an LLM
- Remembers conversation history across multiple turns
- Deploys via a clean Streamlit web interface

---

## 🏗️ RAG Architecture

```
User Query
    ↓
Embed Query (sentence-transformers)
    ↓
FAISS Vector Store
(cosine similarity search → top-3 chunks)
    ↓
Retrieved Context + Chat History + Query
    ↓
Prompt Template
    ↓
LLM (GPT-3.5 / Mistral-7B)
    ↓
Grounded Response + Source Citations
    ↓
ConversationBufferMemory (stores turn)
```

---

## 💾 Context Memory

| Memory Type | Description |
|-------------|-------------|
| ConversationBufferMemory | Stores last N full turns |
| Max Turns | 10 (configurable) |
| Context Window | Last 3 turns fed to LLM |
| Clear Command | Type 'clear' in CLI or click button in UI |

---

## 📚 Knowledge Base Topics

| Topic | Description |
|-------|-------------|
| 🤖 Artificial Intelligence | AI overview and milestones |
| 🧠 Machine Learning | ML types and algorithms |
| 🔬 Deep Learning | Neural networks and architectures |
| 💬 NLP | Transformers, BERT, GPT |
| 🔗 LangChain | Framework components and usage |
| 📦 RAG | Pipeline steps and benefits |
| 🗄️ Vector Databases | FAISS, ChromaDB, Pinecone |
| ⚡ Transformers | Self-attention and key models |
| 🐍 Python | Libraries for data science |
| 📊 Data Science | Workflow and tools |

---

## ⚙️ RAG Pipeline Config

| Parameter | Value |
|-----------|-------|
| Embedding Model | all-MiniLM-L6-v2 (free) |
| Vector Store | FAISS (local) |
| Chunk Size | 500 tokens |
| Chunk Overlap | 50 tokens |
| Top-k Retrieval | 3 chunks |
| LLM | GPT-3.5-turbo / Mistral-7B |
| Memory | ConversationBufferMemory |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG orchestration framework |
| FAISS | Vector similarity search |
| sentence-transformers | Document & query embedding |
| OpenAI / HuggingFace | LLM for response generation |
| Streamlit | Web interface |
| PyPDF | PDF document loading |

---

## 🚀 How to Run

**Step 1 — Clone repository**
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

**Step 2 — Install dependencies**
```bash
pip install langchain langchain-community langchain-openai
           faiss-cpu sentence-transformers openai streamlit
           tiktoken pypdf chromadb
```

**Step 3 — Add your documents**
```bash
mkdir docs
# Copy your .txt or .pdf files into the docs/ folder
```

**Step 4 — Run demo (no API key needed)**
```bash
python rag_chatbot.py
```

**Step 5 — Build vector store (needs API key)**
```bash
export OPENAI_API_KEY="your-key"
python rag_pipeline.py
```

**Step 6 — Launch Streamlit app**
```bash
streamlit run streamlit_app.py
```

---

## 💡 Key Learnings

- How to build a full RAG pipeline from scratch
- Document loading, chunking, and embedding strategies
- FAISS vector store creation and similarity search
- LangChain chains, memory, and prompt templates
- Conversation history management in chatbots
- Deploying AI apps with Streamlit

---

## 👤 Author

**Your Name Here**

LinkedIn — your linkedin link here
GitHub  — your github link here

---

## 📄 License

MIT License — free to use and adapt for your portfolio.
