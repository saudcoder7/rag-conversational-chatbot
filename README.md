# 🤖 RAG Conversational Chatbot

> **Task 2 | AI Internship Portfolio Project**
> A context-aware chatbot using LangChain, FAISS vector store,
> Retrieval-Augmented Generation (RAG), and Streamlit UI.

---

## 📌 About This Project

This is **Task 2** of my AI Internship at CodeAlpha.

The goal is to build a conversational chatbot that remembers
context across turns and retrieves accurate answers from a
vectorized document store using the RAG architecture.

---

## 🎯 Objective

Build a RAG chatbot that:
- Loads and vectorizes a custom document corpus
- Retrieves relevant chunks using FAISS similarity search
- Generates grounded answers using an LLM
- Remembers conversation history across multiple turns
- Deploys via a clean Streamlit web interface

---

## 📂 Project Structure
rag-chatbot/
│
├── rag_chatbot.py        # Core chatbot logic and CLI demo
├── rag_pipeline.py       # Full LangChain + FAISS pipeline
├── streamlit_app.py      # Streamlit web interface
├── requirements.txt      # All Python dependencies
│
├── docs/                 # Your documents go here
│   ├── ai_overview.txt
│   ├── python_guide.txt
│   └── langchain_docs.txt
│
├── vectorstore/          # FAISS index saved here automatically
│
└── README.md             # Project documentation

---

## 🏗️ RAG Architecture
User Query
↓
Embed Query using sentence-transformers
↓
Search FAISS Vector Store
(cosine similarity — retrieve top 3 chunks)
↓
Combine Retrieved Context + Chat History + Query
↓
Fill into Prompt Template
↓
Send to LLM (GPT-3.5-turbo or Mistral-7B)
↓
Generate Grounded Response with Source Citations
↓
Save turn to ConversationBufferMemory
↓
Display Response to User

---

## 💾 Context Memory System

| Property             | Value                         |
|----------------------|-------------------------------|
| Memory Type          | ConversationBufferMemory      |
| Maximum Turns Stored | 10 turns                      |
| Context Fed to LLM   | Last 3 turns of conversation  |
| Follow-up Support    | Yes — handles tell me more    |
| Reset Option         | CLI: type clear — UI: button  |

---

## 📚 Knowledge Base Topics

| Topic                      | What It Covers                        |
|----------------------------|---------------------------------------|
| 🤖 Artificial Intelligence | AI overview, history, and milestones  |
| 🧠 Machine Learning        | Types, algorithms, and frameworks     |
| 🔬 Deep Learning           | Neural networks and architectures     |
| 💬 NLP                     | Transformers, BERT, and GPT models    |
| 🔗 LangChain               | Framework components and use cases    |
| 📦 RAG                     | Full pipeline steps and benefits      |
| 🗄️ Vector Databases        | FAISS, ChromaDB, Pinecone, Weaviate   |
| ⚡ Transformers             | Self-attention and key model families |
| 🐍 Python                  | Essential libraries for data science  |
| 📊 Data Science            | Workflow, tools, and best practices   |

---

## ⚙️ Pipeline Configuration

| Parameter        | Value                         |
|------------------|-------------------------------|
| Embedding Model  | all-MiniLM-L6-v2 (free)       |
| Vector Store     | FAISS stored locally on disk  |
| Chunk Size       | 500 tokens per chunk          |
| Chunk Overlap    | 50 tokens between chunks      |
| Top-k Retrieval  | 3 most relevant chunks        |
| LLM Option 1     | OpenAI GPT-3.5-turbo          |
| LLM Option 2     | Mistral-7B-Instruct (free)    |
| Memory Type      | ConversationBufferMemory      |

---

## 💬 Example Conversation
User : What is RAG?
Bot  : Retrieval-Augmented Generation combines document retrieval
with language model generation to produce grounded and
accurate answers. RAG pipeline: Load docs → Split into
chunks → Embed into vectors → Store in FAISS →
Retrieve top-k on query → Generate response.
📄 Source: RAG Documentation
User : Tell me more about it
Bot  : Expanding on RAG — it significantly reduces hallucinations
and keeps answers factual by grounding responses in your
actual documents rather than relying only on model memory.
It is ideal for Q&A over private documents.
📄 Source: LangChain Framework Docs
User : How does FAISS work?
Bot  : FAISS (Facebook AI Similarity Search) stores
high-dimensional vector embeddings and enables extremely
fast cosine similarity search to find the most relevant
document chunks for any given query in milliseconds.
📄 Source: Vector Databases Overview

---

## 🛠️ Tech Stack

| Tool                  | Purpose                             |
|-----------------------|-------------------------------------|
| LangChain             | RAG orchestration, chains, memory   |
| FAISS                 | Fast vector similarity search       |
| sentence-transformers | Document and query embedding        |
| OpenAI GPT-3.5        | Primary LLM for response generation |
| Mistral-7B            | Free open-source LLM alternative    |
| Streamlit             | Web interface for deployment        |
| PyPDF                 | Loading PDF documents into pipeline |
| ChromaDB              | Alternative vector store option     |

---

## 🚀 How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

**Step 2 — Install all dependencies**
```bash
pip install langchain langchain-community langchain-openai faiss-cpu sentence-transformers openai streamlit tiktoken pypdf chromadb
```

**Step 3 — Run instant demo with no API key needed**
```bash
python rag_chatbot.py
```

**Step 4 — Add your own documents to the docs folder**
```bash
mkdir docs
# Copy your .txt or .pdf files into the docs/ folder
```

**Step 5 — Set your API key and build the vector store**
```bash
export OPENAI_API_KEY="your-openai-key-here"
python rag_pipeline.py
```

**Step 6 — Launch the Streamlit web app**
```bash
streamlit run streamlit_app.py
```
Opens automatically at: http://localhost:8501

---

## 💡 Key Learnings

- How to build a complete RAG pipeline from scratch
- Document loading, chunking, and embedding strategies
- Creating and querying a FAISS vector store
- Using LangChain chains, memory, and prompt templates
- Managing conversation history across chat turns
- Reducing LLM hallucinations with grounded retrieval
- Deploying a full conversational AI app with Streamlit

---

## ⚠️ Important Note

The demo mode in rag_chatbot.py works completely offline without any API key and is great for testing and portfolios. To use real LLM responses with actual AI-generated answers, follow Step 5 above to set your API key and run rag_pipeline.py to build the FAISS vector store first.

---

## 👤 Author

**Saood Faisal Shiekh**

LinkedIn — https://www.linkedin.com/in/saood-faisal-259b40316/
GitHub   — https://github.com/saudcoder7

---

## 📄 License

MIT License — free to use and adapt for your own portfolio.
