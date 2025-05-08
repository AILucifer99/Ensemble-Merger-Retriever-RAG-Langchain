"""
Example script demonstrating how to use the EnsembleMergerRetriever for RAG-based document querying.
This script sets up a complete Retrieval Augmented Generation pipeline using the EnsembleMergerRetriever class
to provide enhanced document retrieval and high-quality answers.
"""

# Import the EnsembleMergerRetriever class from the custom module
# This assumes the class has been defined in a file named 'EnsembleMergerContextualCompressionRetriever.py'
# in a module named 'mergerRetriever'
from EnsembleMergerContextualCompressionRetriever import mergerRetriever


# Step 1: Instantiate the EnsembleMergerRetriever class
# This creates a new instance of our RAG pipeline manager that will handle all processing steps
retriever = mergerRetriever.EnsembleMergerRetriever()


# Step 2: Set up the Large Language Model (LLM)
# Here we're configuring the Groq API to use the Gemma 2 9B instruction-tuned model
# This model will be responsible for generating the final answers based on retrieved content
# Other options could include "llama3-70b-8192" or "qwen-qwq-32b"
retriever.setup_llm(model_name="gemma2-9b-it")


# Step 3: Load and split the PDF document into manageable chunks
# - data_path: Path to the PDF document we want to analyze
# - chunk_size: The size of each text chunk (512 characters)
# - chunk_overlap: Number of characters that overlap between adjacent chunks (128 characters)
#   Overlap helps maintain context between chunks for better semantic understanding
retriever.load_and_split_documents(
    data_path="Data\\ReAct.pdf",  # Path to the ReAct research paper PDF
    chunk_size=512,               # Each chunk will contain ~512 characters
    chunk_overlap=128             # Chunks will overlap by 128 characters
)

# Step 4: Configure the embedding model
# We're using Google's embedding model to convert text chunks into vector representations
# These embeddings capture semantic meaning and enable similarity-based retrieval
# Alternative option: "OpenAI" would use OpenAI's embedding models instead
retriever.setup_embeddings(embedding_provider="Google")

# Step 5: Set up dual vector stores with FAISS and Chroma
# This creates two different retrieval systems that will later be combined
# - FAISS and Chroma are different vector databases with complementary strengths
# - Using both improves retrieval quality through diversity of results
# - Each retriever is configured with its own parameters for search type and number of results
retriever.create_retrievers(
    faiss_search_type="similarity",  # Use similarity search for FAISS (alternative: "mmr")
    faiss_k_documents=3,             # Retrieve 3 most relevant documents with FAISS
    chroma_search_type="similarity",  # Use similarity search for Chroma (alternative: "mmr")
    chroma_k_documents=4              # Retrieve 4 most relevant documents with Chroma
)

# Step 6: Build the ensemble merger retriever with contextual compression
# This combines results from both retrievers and applies filtering techniques:
# - Removes redundant information
# - Filters out irrelevant content
# - Reorders results for optimal context usage
# Different thresholds control the strictness of relevance filtering for each retriever
retriever.build_merger_retriever(
    faiss_embedding_filter_threshold=0.75,  # Higher threshold = stricter filtering for FAISS results
    chroma_embedding_filter_threshold=0.6   # Lower threshold = more permissive filtering for Chroma results
)

# Step 7: Define the query to ask about the document
# This specific query asks about the Chain of Thought prompting concept from the ReAct paper
query = "How the authors have designed ReAct, explain with as much details as possible"

# Step 8: Generate an answer using the complete RAG pipeline
# The pipeline will:
# 1. Convert the query to an embedding vector
# 2. Retrieve relevant document chunks from both vector stores
# 3. Filter and reorder these chunks
# 4. Feed the processed chunks to the LLM along with the query
# 5. Return the generated answer

answer = retriever.generate_answer(
    question=query,                      # The query we want to answer
    pipeline_type="RetrievalQAChain",    # Use the simpler RetrievalQA chain
                                         # Alternative: "RetrievalQAWithSourcesChain" would include sources
    chain_type="stuff",                  # Use the "stuff" method (put all chunks in one prompt)
                                         # Alternatives: "map_reduce" or "refine"
    verbose=False                        # Don't print extra information during processing
)

# Step 9: Print the final answer with a clear separator
print("\n")
print("===" * 100)  # Print a visual separator line
print(f"Final Answer:\n{answer}")  # Display the generated answer