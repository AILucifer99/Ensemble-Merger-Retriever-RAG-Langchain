"""
Script: RAG Pipeline with EnsembleMergerRetriever + Argument Parsing

This script demonstrates how to use the EnsembleMergerRetriever for a full Retrieval-Augmented Generation (RAG)
workflow. It includes argument parsing to allow runtime configuration from the command line.

Steps:
1. Load and split a document into chunks
2. Generate embeddings
3. Create dual retrievers (FAISS and Chroma)
4. Build an ensemble retriever with contextual compression
5. Run a retrieval-based QA pipeline using a chosen LLM
"""

import argparse
from EnsembleMergerContextualCompressionRetriever import mergerRetriever
import os


def main(args):
    # Step 1: Create an instance of EnsembleMergerRetriever
    # This acts as the orchestrator for the full RAG pipeline
    retriever = mergerRetriever.EnsembleMergerRetriever()

    # Step 2: Set up the Large Language Model (LLM) used for final answer generation
    retriever.setup_llm(model_name=args.model_name)

    # Step 3: Load the document and split it into smaller overlapping chunks
    # This improves semantic retrieval and context preservation
    retriever.load_and_split_documents(
        data_path=args.data_path,           # PDF or text document path
        chunk_size=args.chunk_size,         # Length of each chunk (in characters)
        chunk_overlap=args.chunk_overlap    # Overlap between chunks to preserve context
    )

    # Step 4: Initialize the embedding model for converting text to vector format
    # Embeddings enable semantic similarity search
    retriever.setup_embeddings(embedding_provider=args.embedding_provider)

    # Step 5: Configure individual vector-based retrievers using FAISS and Chroma
    # Both use different mechanisms to improve diversity in results
    retriever.create_retrievers(
        faiss_search_type=args.faiss_search_type,         # Similarity search or MMR
        faiss_k_documents=args.faiss_k,                   # Top K documents for FAISS
        chroma_search_type=args.chroma_search_type,       # Similarity search or MMR
        chroma_k_documents=args.chroma_k                  # Top K documents for Chroma
    )

    # Step 6: Build the ensemble retriever that merges and compresses results
    # Reduces redundancy and improves relevance
    retriever.build_merger_retriever(
        faiss_embedding_filter_threshold=args.faiss_filter_threshold,     # FAISS relevance threshold
        chroma_embedding_filter_threshold=args.chroma_filter_threshold    # Chroma relevance threshold
    )

    # Step 7: Perform question-answering using the full pipeline
    # The chosen chain type determines how retrieved chunks are processed by the LLM
    answer = retriever.generate_answer(
        question=args.query,                  # User's input question
        pipeline_type=args.pipeline_type,     # RetrievalQAChain or RetrievalQAWithSourcesChain
        chain_type=args.chain_type,           # stuff, map_reduce, refine
        verbose=args.verbose                  # Print detailed logs if True
    )

    # Step 8: Output the final answer clearly
    print("\n")
    print("===" * 100)
    print(f"Final Answer:\n{answer}")

if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(
        description="Run a RAG pipeline using EnsembleMergerRetriever with configurable arguments."
    )

    # LLM Configuration
    parser.add_argument("--model_name", type=str, default="gemma2-9b-it",
                        help="Name of the language model to use (e.g., 'gemma2-9b-it', 'llama3-70b-8192')")

    # Document Input Settings
    parser.add_argument("--data_path", type=str, default="Data{}ReAct.pdf".format(os.sep),
                        help="Path to the PDF or text document for analysis")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Character length of each document chunk")
    parser.add_argument("--chunk_overlap", type=int, default=128,
                        help="Overlap between chunks to maintain context")

    # Embedding Settings
    parser.add_argument("--embedding_provider", type=str, default="Google", choices=["Google", "OpenAI"],
                        help="Embedding provider for vectorization")

    # FAISS Retriever Settings
    parser.add_argument("--faiss_search_type", type=str, default="similarity",
                        help="Search type for FAISS retriever ('similarity' or 'mmr')")
    parser.add_argument("--faiss_k", type=int, default=3,
                        help="Number of top documents to retrieve with FAISS")

    # Chroma Retriever Settings
    parser.add_argument("--chroma_search_type", type=str, default="similarity",
                        help="Search type for Chroma retriever ('similarity' or 'mmr')")
    parser.add_argument("--chroma_k", type=int, default=4,
                        help="Number of top documents to retrieve with Chroma")

    # Filtering thresholds for each retriever
    parser.add_argument("--faiss_filter_threshold", type=float, default=0.75,
                        help="Relevance filtering threshold for FAISS (0 to 1)")
    parser.add_argument("--chroma_filter_threshold", type=float, default=0.6,
                        help="Relevance filtering threshold for Chroma (0 to 1)")

    # Query Settings
    parser.add_argument("--query", type=str, required=True,
                        help="User query/question to be answered by the system")

    # QA Pipeline Settings
    parser.add_argument("--pipeline_type", type=str, default="RetrievalQAChain",
                        help="Pipeline type: 'RetrievalQAChain' or 'RetrievalQAWithSourcesChain'")
    parser.add_argument("--chain_type", type=str, default="stuff",
                        help="Chain type: 'stuff', 'map_reduce', or 'refine'")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output during processing")

    # Parse and pass arguments to main
    args = parser.parse_args()
    main(args)
