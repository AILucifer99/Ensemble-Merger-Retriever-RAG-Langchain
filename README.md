# 🧠 Dual Ensemble-ContextualCompression-Retrieval Augmented Generation

This project demonstrates a **robust, context-aware retriever pipeline** using LangChain's `ContextualCompressionRetriever`. It enhances standard retrieval by compressing context — filtering or summarizing — before passing it to an LLM. This is useful for token-limited models and improves precision in Retrieval-Augmented Generation (RAG). The project demonstrates how to use ensemble retrievers via two separate vectorstores `ChromaDB` and `FAISS`. Langchain's `MergerRetriever` plays the most important role in creating the ensemble system to ensure better output when provided a user question.

---
## Architecture of the designed pipeline
![Ensemble-ContextualCompression-Retrieval Augmented Generation](https://github.com/AILucifer99/Ensemble-Merger-Retriever-RAG-Langchain/blob/main/architecture/detailed-rag.png)

---
## Demo Links
| Google Colab demo |
|:-:|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HPqK-iN5616M_A1rCwIpHlgNuVsaX3Kk#scrollTo=AgJk0I8KZMaD) |

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
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with the following content:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Running the code
```bash
python inference.py
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
---

## 🧪 Most Important Modules for the working.

```python
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import (
    TokenTextSplitter, 
    RecursiveCharacterTextSplitter
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_chroma import Chroma
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.chains import (
    RetrievalQAWithSourcesChain, 
    RetrievalQA
)
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.merger_retriever import MergerRetriever

from dotenv import load_dotenv, find_dotenv
import os
from langchain.chains import RetrievalQA
import langchain
langchain.debug = False
```

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

## Dual Retriever RAG System: Code Explanation
This document explains a sophisticated Retrieval-Augmented Generation (RAG) system that implements a dual retriever approach with contextual compression. The system enhances document retrieval quality by combining two vector databases (FAISS and Chroma) with advanced filtering techniques.

## Code Structure

The code consists of three main functions:
1. `DataAugmentationWithDualRetriever`: Processes documents and creates dual retrievers
2. `EnsembleContextualCompressionRetriever`: Implements compression and filtering
3. `generateAnswerFunction`: Handles question answering with different RAG pipelines

Let's examine each function in detail.

## DataAugmentationWithDualRetriever

This function handles document loading, chunking, and creating vector stores.

```python
def DataAugmentationWithDualRetriever(
    data_path, embedding_provider, chunk_size, chunk_overlap, 
    faiss_retriever_search_type, faiss_retriever_k_documents, 
    chroma_retriever_search_type, chroma_retriever_k_documents, **kwargs):
    # Function implementation...
```

### Parameters:
- `data_path`: Path to the PDF document to process
- `embedding_provider`: Embedding model provider ("Google" or "OpenAI")
- `chunk_size`: Size of document chunks in characters
- `chunk_overlap`: Overlap between chunks in characters
- `faiss_retriever_search_type`: Search method for FAISS (typically "similarity")
- `faiss_retriever_k_documents`: Number of documents to retrieve with FAISS
- `chroma_retriever_search_type`: Search method for Chroma
- `chroma_retriever_k_documents`: Number of documents to retrieve with Chroma
- `**kwargs`: Additional parameters, including `execute_function`

### Process:
1. Loads PDF documents using `PyMuPDFLoader`
2. Splits documents into chunks using `RecursiveCharacterTextSplitter`
3. Creates embeddings using either Google AI or OpenAI models
4. Builds two vector stores:
   - FAISS vector database
   - Chroma vector database
5. Returns both retrievers and the embedding model

### Example Usage:
```python
faissRetriever, chromaRetriever, embeddings = DataAugmentationWithDualRetriever(
    data_path = "Data\\ReAct.pdf",
    embedding_provider = "Google",
    chunk_size = 512,
    chunk_overlap = 128,
    faiss_retriever_search_type = "similarity",
    faiss_retriever_k_documents = 3,
    chroma_retriever_search_type = "mmr",
    chroma_retriever_k_documents = 5,
    execute_function = True
)
```

## EnsembleContextualCompressionRetriever

This function creates an advanced retrieval pipeline with document filtering and compression.

```python
def EnsembleContextualCompressionRetriever(**kwargs):
    # Function implementation...
```

### Parameters (via kwargs):
- `embeddings`: Embedding model for similarity calculations
- `faiss_embeddingfilter_threshold`: Similarity threshold for FAISS filtering
- `chroma_embeddingfilter_threshold`: Similarity threshold for Chroma filtering
- `faissRetriever`: FAISS retriever object from previous function
- `chromaRetriever`: Chroma retriever object from previous function
- `execute_pipeline`: Boolean to control execution

### Process:
1. Creates three document transformers:
   - `EmbeddingsRedundantFilter`: Removes duplicate or highly similar documents
   - `EmbeddingsFilter`: Filters documents based on relevance thresholds
   - `LongContextReorder`: Optimizes document ordering for LLM processing
2. Builds document compression pipelines for each retriever
3. Creates contextual compression retrievers for both FAISS and Chroma
4. Merges both retrievers using `MergerRetriever`
5. Returns the combined retriever

### Example Usage:
```python
retriever = EnsembleContextualCompressionRetriever(
    faiss_embeddingfilter_threshold = 0.75,
    chroma_embeddingfilter_threshold = 0.6,
    embeddings = embeddings, 
    execute_pipeline = True,
    faissRetriever = faissRetriever, 
    chromaRetriever = chromaRetriever,
)
```

## generateAnswerFunction

This function handles question answering using the retrieval system.

```python
def generateAnswerFunction(
    question, llm_model, retriever_function, 
    verbose=-1, **kwargs):
    # Function implementation...
```

### Parameters:
- `question`: User query text
- `llm_model`: Language model to use for answer generation
- `retriever_function`: The combined retriever object
- `verbose`: Controls output verbosity
- `**kwargs`: Additional parameters including pipeline configuration

### Process:
1. Selects RAG pipeline type based on `ragPipelineConfig`:
   - `RetrievalQAWithSourcesChain`: Includes source attribution
   - `RetrievalQAChain`: Standard QA chain
2. Initializes the chosen RAG pipeline
3. Processes the user question
4. Returns the generated answer

### Pipeline Options:
- `chain_type`: Processing method (e.g., "stuff", "map_reduce")
- `return_source_documents`: Whether to include source documents in response

## Implementation Example

The code includes a simple implementation example:

```python
dataPath = "Data\\ReAct.pdf"
embedding = "Google"

faissRetriever, chromaRetriever, embeddings = DataAugmentationWithDualRetriever(
    # Parameters...
)

retriever = EnsembleContextualCompressionRetriever(
    # Parameters...
)
```

## Technical Highlights

1. **Dual Retriever Approach**:
   - Uses two different vector stores (FAISS and Chroma)
   - Allows different parameters for each retriever
   - Combines results for more comprehensive retrieval

2. **Contextual Compression**:
   - Reduces redundancy in retrieved documents
   - Filters documents based on relevance thresholds
   - Optimizes document ordering for LLM context windows

3. **Flexible Pipeline Options**:
   - Supports different RAG pipelines
   - Allows for source attribution
   - Configurable chain types

4. **Embedding Model Flexibility**:
   - Supports Google AI embeddings
   - Supports OpenAI embeddings

## Advanced Features

### Document Filtering
The system implements three levels of document filtering:
- **Redundancy filtering**: Removes duplicate or near-duplicate content
- **Relevancy filtering**: Only keeps documents above a similarity threshold
- **Reordering**: Optimizes document order for LLM processing

### Merger Retriever
The `MergerRetriever` combines results from both retrievers, providing a more robust set of documents for the language model.

### Configurable Thresholds
Different similarity thresholds can be set for FAISS and Chroma retrievers, allowing fine-tuning of the retrieval process.

## Dependencies

This code relies on several libraries:
- LangChain for the RAG components
- PyMuPDF for PDF processing
- FAISS and Chroma for vector stores
- Google AI or OpenAI for embeddings

## Conclusion

This implementation represents an advanced RAG system that goes beyond basic retrieval by:
1. Utilizing multiple vector stores
2. Implementing sophisticated document filtering
3. Supporting different question-answering pipelines
4. Providing flexible configuration options

The dual retriever approach with contextual compression helps overcome limitations of single-retriever systems and improves the quality of context provided to language models.

## 🧠 Author

Built with ❤️ by AILucifer using LangChain, Gemini, Groq, and FAISS for high-performance document intelligence systems.

