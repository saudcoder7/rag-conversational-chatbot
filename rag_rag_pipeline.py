# ============================================================
#  rag_pipeline.py
#  Full RAG pipeline using LangChain + FAISS + OpenAI/HuggingFace
#  Run ONCE to build vector store, then use streamlit_app.py
# ============================================================
#
#  Requirements:
#    pip install langchain langchain-community langchain-openai
#              faiss-cpu sentence-transformers openai tiktoken
#              pypdf chromadb
#
#  Set API key (choose one):
#    export OPENAI_API_KEY="your-key"           # OpenAI
#    export HUGGINGFACEHUB_API_TOKEN="your-key" # Free HuggingFace
# ============================================================

import os
from pathlib import Path

# ── LangChain Imports ─────────────────────────────────────────
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import (
    OpenAIEmbeddings,         # Option 1: OpenAI
    HuggingFaceEmbeddings,    # Option 2: Free HuggingFace
)
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ── CONFIG ────────────────────────────────────────────────────
DOCS_DIR        = "./docs"           # Put your documents here
VECTORSTORE_DIR = "./vectorstore"    # FAISS index saved here
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Free HuggingFace embeddings
LLM_PROVIDER    = "huggingface"      # "openai" or "huggingface"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

# ── SYSTEM PROMPT ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful AI assistant with access to a knowledge base.
Use the retrieved context to answer questions accurately.
If the answer is not in the context, say so honestly.
Always be clear, concise, and helpful.

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:
"""

def build_vectorstore():
    """Load documents, split, embed, and save FAISS index"""

    print("=" * 55)
    print("   BUILDING RAG VECTOR STORE")
    print("=" * 55)

    # ── Step 1: Load Documents ────────────────────────────────
    print("\n[1/4] Loading documents from ./docs ...")
    Path(DOCS_DIR).mkdir(exist_ok=True)

    # Create sample documents if none exist
    sample_docs = [
        ("ai_overview.txt",
         "Artificial Intelligence Overview\n"
         "AI is the simulation of human intelligence by machines...\n"
         "Machine learning is a subset of AI that uses data...\n"
         "Deep learning uses neural networks with multiple layers..."),
        ("python_guide.txt",
         "Python for Data Science\n"
         "Python is the most popular language for data science...\n"
         "Key libraries: NumPy, Pandas, Matplotlib, Scikit-learn...\n"
         "TensorFlow and PyTorch are used for deep learning..."),
        ("langchain_docs.txt",
         "LangChain Documentation\n"
         "LangChain is a framework for LLM-powered applications...\n"
         "RAG combines retrieval with language generation...\n"
         "FAISS enables fast similarity search over vectors..."),
    ]

    for fname, content in sample_docs:
        fpath = Path(DOCS_DIR) / fname
        if not fpath.exists():
            fpath.write_text(content)
            print(f"   ✅ Created sample: {fname}")

    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt",
                             loader_cls=TextLoader)
    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} documents")

    # ── Step 2: Split into Chunks ─────────────────────────────
    print("\n[2/4] Splitting documents into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # ── Step 3: Create Embeddings ─────────────────────────────
    print(f"\n[3/4] Creating embeddings ({EMBEDDING_MODEL}) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    print("   ✅ Embeddings model loaded")

    # ── Step 4: Build & Save FAISS Index ─────────────────────
    print("\n[4/4] Building FAISS vector store ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(VECTORSTORE_DIR).mkdir(exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"   ✅ Vector store saved to {VECTORSTORE_DIR}")
    print(f"   📊 Total vectors: {vectorstore.index.ntotal}")

    return vectorstore, embeddings


def build_rag_chain(vectorstore):
    """Build the full conversational RAG chain"""

    print("\n🔗 Building RAG chain ...")

    # ── LLM ──────────────────────────────────────────────────
    if LLM_PROVIDER == "openai":
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
        )
    else:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0.7, "max_length": 500},
        )

    # ── Memory ────────────────────────────────────────────────
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # ── Retriever ─────────────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},        # top-3 chunks
    )

    # ── Prompt ────────────────────────────────────────────────
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT
    )

    # ── Chain ─────────────────────────────────────────────────
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    print("✅ RAG chain ready!")
    return chain


def load_vectorstore():
    """Load existing FAISS vector store from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Loaded vector store: {vectorstore.index.ntotal} vectors")
    return vectorstore, embeddings


if __name__ == "__main__":
    vectorstore, embeddings = build_vectorstore()
    chain = build_rag_chain(vectorstore)

    # Quick test
    print("\n🧪 Testing RAG pipeline ...")
    result = chain({"question": "What is RAG?"})
    print(f"Q: What is RAG?")
    print(f"A: {result['answer'][:200]}...")
    print("\n✅ Pipeline ready!")
    print("   Run: streamlit run streamlit_app.py")
