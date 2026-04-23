# ============================================================
#  Task 2 (AI Internship): RAG Conversational Chatbot
#  Framework : LangChain + RAG (Retrieval-Augmented Generation)
#  Vector DB : FAISS
#  Memory    : ConversationBufferMemory
#  Deploy    : Streamlit
# ============================================================

import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  HOW TO USE WITH REAL LLM (internet + API key)
#
#  Step 1 — Install:
#    pip install langchain langchain-community langchain-openai
#              faiss-cpu sentence-transformers openai streamlit
#              tiktoken pypdf chromadb
#
#  Step 2 — Set your API key:
#    export OPENAI_API_KEY="your-key-here"
#    OR use HuggingFace free models (see rag_pipeline.py)
#
#  Step 3 — Add your documents to /docs folder
#
#  Step 4 — Run:
#    python rag_pipeline.py      (builds vector store)
#    streamlit run streamlit_app.py
# ============================================================

# ── KNOWLEDGE BASE (Custom Corpus) ───────────────────────────
# This simulates a vectorized document store
# In real RAG, these come from embedded documents via FAISS

KNOWLEDGE_BASE = {
    "artificial_intelligence": {
        "title": "Artificial Intelligence",
        "content": (
            "Artificial Intelligence (AI) refers to the simulation of human "
            "intelligence in machines that are programmed to think and learn. "
            "AI includes machine learning, deep learning, natural language "
            "processing, computer vision, and robotics. Major AI milestones "
            "include IBM Deep Blue defeating chess champion Garry Kasparov in "
            "1997, Google DeepMind's AlphaGo beating world champion Lee Sedol "
            "in 2016, and OpenAI's GPT models revolutionizing NLP from 2020 onwards."
        )
    },
    "machine_learning": {
        "title": "Machine Learning",
        "content": (
            "Machine Learning (ML) is a subset of AI that enables systems to "
            "learn from data without being explicitly programmed. Key types "
            "include supervised learning (labeled data), unsupervised learning "
            "(unlabeled data), and reinforcement learning (reward-based). "
            "Popular algorithms include Linear Regression, Decision Trees, "
            "Random Forest, SVM, and Neural Networks. Scikit-learn, TensorFlow, "
            "and PyTorch are the most widely used ML frameworks."
        )
    },
    "deep_learning": {
        "title": "Deep Learning",
        "content": (
            "Deep Learning is a subset of machine learning using neural networks "
            "with multiple hidden layers. It excels at image recognition, speech "
            "recognition, and natural language processing. Key architectures "
            "include CNNs for images, RNNs/LSTMs for sequences, Transformers "
            "for NLP, and GANs for generative tasks. Deep learning requires "
            "large datasets and significant computational power, typically GPUs."
        )
    },
    "natural_language_processing": {
        "title": "Natural Language Processing",
        "content": (
            "Natural Language Processing (NLP) is the branch of AI that deals "
            "with the interaction between computers and human language. NLP tasks "
            "include text classification, sentiment analysis, machine translation, "
            "named entity recognition, question answering, and text generation. "
            "The Transformer architecture, introduced in the paper 'Attention Is "
            "All You Need' in 2017, revolutionized NLP. BERT, GPT, and T5 are "
            "the most influential transformer models."
        )
    },
    "langchain": {
        "title": "LangChain Framework",
        "content": (
            "LangChain is a framework for developing applications powered by "
            "large language models. It provides tools for prompt management, "
            "memory, chains, agents, and document retrieval. Key components "
            "include LLMs (language model wrappers), Chains (sequences of calls), "
            "Agents (dynamic tool use), Memory (conversation history), and "
            "Document Loaders/Vector Stores for RAG pipelines. LangChain supports "
            "OpenAI, Anthropic, HuggingFace, and many other LLM providers."
        )
    },
    "rag": {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "Retrieval-Augmented Generation (RAG) combines document retrieval "
            "with language model generation to produce grounded, accurate answers. "
            "RAG pipeline steps: (1) Load documents, (2) Split into chunks, "
            "(3) Embed chunks into vectors, (4) Store in vector database like FAISS "
            "or ChromaDB, (5) On query, retrieve top-k similar chunks, (6) Feed "
            "retrieved context + query to LLM, (7) Generate grounded response. "
            "RAG reduces hallucinations and keeps answers factual and up-to-date."
        )
    },
    "vector_database": {
        "title": "Vector Databases",
        "content": (
            "Vector databases store high-dimensional embeddings for semantic "
            "search. Unlike traditional databases, they enable similarity search "
            "using distance metrics like cosine similarity or Euclidean distance. "
            "Popular vector databases include FAISS (Facebook AI Similarity Search), "
            "ChromaDB, Pinecone, Weaviate, and Qdrant. They are essential for RAG "
            "pipelines, recommendation systems, and semantic search applications."
        )
    },
    "transformers": {
        "title": "Transformer Models",
        "content": (
            "Transformers are neural network architectures based on self-attention "
            "mechanisms introduced by Google in 2017. They process entire sequences "
            "in parallel, making them faster than RNNs. Key models include BERT "
            "(bidirectional encoder for classification), GPT series (autoregressive "
            "decoder for generation), T5 (text-to-text), and BART (sequence-to-sequence). "
            "Hugging Face hosts over 300,000 pretrained transformer models."
        )
    },
    "python": {
        "title": "Python Programming",
        "content": (
            "Python is the most popular programming language for AI and data science. "
            "Key libraries include NumPy and Pandas for data manipulation, "
            "Matplotlib and Seaborn for visualization, Scikit-learn for ML, "
            "TensorFlow and PyTorch for deep learning, and Hugging Face Transformers "
            "for NLP. Python's simple syntax, large community, and rich ecosystem "
            "make it the go-to language for data scientists and AI engineers."
        )
    },
    "data_science": {
        "title": "Data Science",
        "content": (
            "Data Science is an interdisciplinary field combining statistics, "
            "mathematics, and computer science to extract insights from data. "
            "The data science workflow includes data collection, cleaning, "
            "exploratory data analysis (EDA), feature engineering, model training, "
            "evaluation, and deployment. Key tools are Python, R, SQL, Jupyter "
            "Notebooks, and cloud platforms like AWS, GCP, and Azure."
        )
    },
}

# ── SIMPLE VECTOR SEARCH (cosine-like keyword matching) ──────
def retrieve_documents(query, top_k=2):
    """
    Simulates FAISS vector similarity search.
    In real RAG, this uses sentence-transformers embeddings
    and FAISS IndexFlatIP for cosine similarity search.
    """
    query_lower = query.lower()
    scores = {}

    for key, doc in KNOWLEDGE_BASE.items():
        score = 0
        # Score based on keyword overlap
        words = query_lower.split()
        content_lower = doc["content"].lower()
        title_lower   = doc["title"].lower()

        for word in words:
            if len(word) > 3:
                if word in title_lower:
                    score += 3       # title match = higher weight
                if word in content_lower:
                    score += 1
                if word in key:
                    score += 2

        if score > 0:
            scores[key] = score

    # Return top_k documents
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    retrieved = []
    for key, score in sorted_docs[:top_k]:
        retrieved.append({
            "title"  : KNOWLEDGE_BASE[key]["title"],
            "content": KNOWLEDGE_BASE[key]["content"],
            "score"  : round(score / 10, 2),
        })
    return retrieved

# ── CONTEXT MEMORY ────────────────────────────────────────────
class ConversationMemory:
    """
    Simulates LangChain ConversationBufferMemory.
    Stores full conversation history for context.
    """
    def __init__(self, max_turns=10):
        self.history  = []
        self.max_turns = max_turns

    def add_turn(self, human, ai):
        self.history.append({"human": human, "ai": ai})
        if len(self.history) > self.max_turns:
            self.history.pop(0)     # keep last N turns

    def get_context(self):
        if not self.history:
            return ""
        ctx = "Previous conversation:\n"
        for turn in self.history[-3:]:   # last 3 turns
            ctx += f"Human: {turn['human']}\n"
            ctx += f"AI: {turn['ai']}\n"
        return ctx

    def clear(self):
        self.history = []

# ── RESPONSE GENERATOR ────────────────────────────────────────
def generate_response(query, retrieved_docs, memory):
    """
    Simulates LLM generating a grounded response from context.
    In real RAG:
      prompt = PromptTemplate(context + history + query)
      response = llm(prompt)
    """
    context = memory.get_context()
    query_lower = query.lower()

    # Greet handler
    greetings = ["hello", "hi", "hey", "good morning",
                 "good evening", "how are you"]
    if any(g in query_lower for g in greetings):
        return (
            "Hello! 👋 I'm your AI Knowledge Assistant powered by RAG. "
            "I can answer questions about AI, Machine Learning, Deep Learning, "
            "NLP, LangChain, Python, and Data Science by retrieving information "
            "from my knowledge base. What would you like to learn about today?"
        )

    # Follow-up detection using memory
    follow_ups = ["tell me more", "explain more", "what else",
                  "can you elaborate", "more details", "go on"]
    if any(f in query_lower for f in follow_ups) and context:
        # Extract topic from last conversation
        last = memory.history[-1]["human"] if memory.history else ""
        docs = retrieve_documents(last, top_k=1)
        if docs:
            return (
                f"Expanding on {docs[0]['title']}:\n\n"
                f"{docs[0]['content']}\n\n"
                f"💡 This information was retrieved from my knowledge base. "
                f"Feel free to ask any follow-up questions!"
            )

    # No documents retrieved
    if not retrieved_docs:
        return (
            "I couldn't find specific information about that in my knowledge base. "
            "I can help you with topics like:\n"
            "• 🤖 Artificial Intelligence & Machine Learning\n"
            "• 🧠 Deep Learning & Transformers\n"
            "• 💬 NLP & BERT/GPT models\n"
            "• 🔗 LangChain & RAG pipelines\n"
            "• 🗄️ Vector Databases (FAISS, ChromaDB)\n"
            "• 🐍 Python for Data Science\n\n"
            "Please try asking about one of these topics!"
        )

    # Build grounded response from retrieved docs
    response_parts = []

    if context:
        response_parts.append(
            "*(Based on our conversation and retrieved documents)*\n"
        )

    for i, doc in enumerate(retrieved_docs):
        if i == 0:
            response_parts.append(
                f"📄 **{doc['title']}** "
                f"(Relevance: {doc['score']})\n\n"
                f"{doc['content']}"
            )
        else:
            response_parts.append(
                f"\n\n📎 **Related — {doc['title']}**\n"
                f"{doc['content'][:200]}..."
            )

    response_parts.append(
        "\n\n💡 *This answer was generated using RAG — "
        "retrieved from vectorized documents in my knowledge base.*"
    )

    return "\n".join(response_parts)

# ── MAIN CHAT FUNCTION ────────────────────────────────────────
memory = ConversationMemory(max_turns=10)

def chat(user_input):
    """Main RAG pipeline: retrieve → generate → remember"""
    retrieved = retrieve_documents(user_input, top_k=2)
    response  = generate_response(user_input, retrieved, memory)
    memory.add_turn(user_input, response)
    return response, retrieved

# ── AUTO DEMO ─────────────────────────────────────────────────
def run_demo():
    print("=" * 62)
    print("   🤖 RAG CONVERSATIONAL CHATBOT — AUTO DEMO")
    print("   LangChain + FAISS + Context Memory")
    print("=" * 62)

    demo_queries = [
        "Hello!",
        "What is Retrieval-Augmented Generation?",
        "Tell me more",
        "How does FAISS vector database work?",
        "What is LangChain used for?",
        "Explain deep learning and transformers",
        "How is Python used in data science?",
    ]

    for query in demo_queries:
        print(f"\n👤 User : {query}")
        print("─" * 55)
        response, docs = chat(query)
        print(f"🤖 Bot  :\n{response[:350]}...")
        if docs:
            print(f"\n📚 Retrieved: {', '.join(d['title'] for d in docs)}")
        print("─" * 55)

    print(f"\n💾 Conversation Memory: {len(memory.history)} turns stored")
    print("\n✅ Demo complete!")
    print("💡 For full RAG: python rag_pipeline.py")
    print("💡 For web UI  : streamlit run streamlit_app.py")
    print("=" * 62)

# ── CLI INTERFACE ─────────────────────────────────────────────
def run_cli():
    memory.clear()
    print("=" * 62)
    print("   🤖 RAG CHATBOT — INTERACTIVE MODE")
    print("=" * 62)
    print("Ask me anything about AI, ML, NLP, LangChain, or Python!")
    print("Type 'clear' to reset memory | 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        if user_input.lower() == 'clear':
            memory.clear()
            print("🗑️ Memory cleared!\n")
            continue

        response, docs = chat(user_input)
        print(f"\n🤖 Bot:\n{response}")
        if docs:
            print(f"\n📚 Sources: {', '.join(d['title'] for d in docs)}")
        print(f"💾 Memory: {len(memory.history)} turns\n")
        print("─" * 62 + "\n")

if __name__ == "__main__":
    run_demo()
