# 🧠 Contextual Compression Retriever with LangChain

This project demonstrates how to build a **context-aware retriever** pipeline using LangChain's `ContextualCompressionRetriever`. It compresses retrieved content before feeding it into an LLM, improving the quality and efficiency of Retrieval-Augmented Generation (RAG) systems.

---

## 🔍 What It Does

This notebook showcases:

- PDF ingestion using LangChain's document loaders
- Chunking and embedding using Google Generative AI
- Vector storage and similarity search via FAISS
- An advanced **Contextual Compression Retriever** that filters or summarizes content
- RetrievalQA pipeline using Groq-hosted LLMs (like Gemma)

---

## 📦 Components Used

| Component                      | Description |
|-------------------------------|-------------|
| `PyPDFLoader`                 | Loads PDF documents |
| `RecursiveCharacterTextSplitter` | Chunks documents with overlap |
| `GoogleGenerativeAIEmbeddings` | Embeds chunks using Gemini Embeddings |
| `FAISS and ChromaDB`                       | Stores embeddings for similarity search |
| `ContextualCompressionRetriever` | Wraps base retriever and filters docs |
| `LLMChainExtractor` or `EmbeddingsFilter` | Compresses irrelevant content |
| `ChatGroq`                    | LLMs like Gemma or LLaMA via Groq |
| `RetrievalQA`                 | RAG pipeline from retriever + LLM |

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

Install required packages:

```bash
pip install langchain langchain-openai langchain-google-genai langchain-community langchain-groq python-dotenv
```

### 2. Environment Variables

Create a `.env` file in the root directory and add:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## 🚀 How It Works

### Step-by-step Pipeline

1. **Load PDF:** A document (e.g., `ReAct.pdf`) is loaded and parsed.
2. **Chunking:** Text is split into overlapping chunks.
3. **Embedding:** Chunks are embedded using Gemini (`embedding-001`).
4. **Store Vectors:** FAISS vector store is created for similarity search.
5. **Query Input:** User inputs a query (e.g., "Explain the ReAct Algorithm").
6. **Retrieve Chunks:** FAISS retrieves top `k` relevant chunks.
7. **Compress Context:** 
    - Uses either LLM or embedding-based filtering
    - Removes irrelevant or redundant parts
8. **Generate Answer:** Filtered context is sent to an LLM (via Groq) to generate the final answer.

---

## 🧪 Sample Query

```python
user_question = "Explain the concept of ReAct Algorithm"
result = chain.invoke(user_question)
print(result["result"])
```

---

## 📌 Notes

- You can switch between Google or OpenAI embeddings by toggling `embedding_provider`.
- Replace `retriever` in `RetrievalQA` with the `ContextualCompressionRetriever` to enable advanced filtering.
- Useful for token-efficient and precision-focused RAG setups.

---

## 🧠 Author

Built with ❤️ by a GenAI Architect leveraging LangChain, Groq, and Gemini for advanced document QA pipelines.
