# 🧠 Dual Ensemble-ContextualCompression-Retrieval Augmented Generation

This project demonstrates a **robust, context-aware retriever pipeline** using LangChain's `ContextualCompressionRetriever`. It enhances standard retrieval by compressing context — filtering or summarizing — before passing it to an LLM. This is useful for token-limited models and improves precision in Retrieval-Augmented Generation (RAG).

---

## 🔍 What This Project Covers

- Loading and parsing PDFs
- Text chunking with overlaps
- Generating semantic embeddings (Google / OpenAI)
- Vector indexing and similarity search with FAISS
- Building an advanced **Contextual Compression Retriever**:
  - LLM-based compression (summarization/filtering)
  - Embedding similarity filtering
- Querying with a Groq-hosted LLM (Gemma 2B)
- Constructing a Retrieval-Augmented Generation (RAG) pipeline

---

## 📂 File Overview

| File | Description |
|------|-------------|
| `ContextualCompressionRetriever.ipynb` | Main notebook showcasing the pipeline |
| `ReAct.pdf` | Sample input document (academic paper) |
| `README.md` | This documentation file |
| `.env` | Environment variable file with API keys |

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

Install required Python packages:

```bash
pip install langchain langchain-openai langchain-google-genai langchain-community langchain-groq python-dotenv faiss-cpu
```

### 2. Environment Setup

Create a `.env` file with the following content:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## 🧠 Key Components

| Component | Description |
|----------|-------------|
| `PyPDFLoader` | Extracts text from PDF files |
| `RecursiveCharacterTextSplitter` | Splits documents into overlapping chunks |
| `GoogleGenerativeAIEmbeddings` | Converts text into vector embeddings |
| `FAISS` | Fast similarity search over document vectors |
| `ContextualCompressionRetriever` | Wraps a base retriever and compresses results |
| `LLMChainExtractor` | Uses an LLM to extract most relevant parts |
| `EmbeddingsFilter` | Filters based on cosine similarity to query |
| `ChatGroq` | Access to fast LLMs via Groq (e.g., Gemma, LLaMA3) |
| `RetrievalQA` | LLM + retriever chain for question answering |

---

## 🚀 How It Works

### ✅ Step-by-Step Flow

1. **PDF Loading**
   - Input document is parsed using `PyPDFLoader`.

2. **Chunking**
   - Text is split using `RecursiveCharacterTextSplitter` with chunk size 512 and 128 overlap.

3. **Embedding**
   - Embeddings are generated using Google’s `embedding-001` model.

4. **Vector Store (FAISS)**
   - FAISS is used to index and retrieve top-k relevant chunks.

5. **Compression Layer (Optional but Powerful)**
   - Either:
     - `LLMChainExtractor`: uses LLM to extract only relevant info
     - `EmbeddingsFilter`: filters based on similarity scores

6. **Retriever**
   - A `ContextualCompressionRetriever` wraps the base retriever and compresses retrieved results.

7. **Question Answering**
   - A `RetrievalQA` chain is constructed using Groq-hosted LLM to answer user queries.

---

## 🧪 Sample Query Code

```python
user_question = "Explain the concept of ReAct Algorithm"
result = chain.invoke(user_question)
print(result["result"])
```

---

## 🛠️ Customization Tips

- To switch to **OpenAI Embeddings**, update:
  ```python
  embedding = OpenAIEmbeddings()
  ```
- Replace compression logic:
  ```python
  compressor = EmbeddingsFilter(...) or LLMChainExtractor(...)
  ```

- For LLMs via Groq, use any of:
  - `llama3-70b-8192`
  - `gemma2-9b-it`
  - `qwen-qwq-32b`

---

## 📌 Use Cases

- Token-efficient RAG for large document corpora
- Domain-specific QA systems
- Legal, academic, or technical document assistants
- Precise summarization & answer generation

---

## 🧠 Author

Built with ❤️ by a GenAI Architect using LangChain, Gemini, Groq, and FAISS for high-performance document intelligence systems.

