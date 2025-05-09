#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AsyncEnsembleMergerRetriever Usage Example

This script demonstrates how to use the AsyncEnsembleMergerRetriever class for
document question answering with a RAG (Retrieval-Augmented Generation) pipeline.
It shows both automated and interactive question answering modes.

Requirements:
- Valid API keys in .env file (GOOGLE_API_KEY, GROQ_API_KEY)
- PDF document for analysis
"""

# Standard library imports
import asyncio
import os

# Third-party imports
from dotenv import load_dotenv  # For loading environment variables from .env file

# Import the optimized retriever class from the module
# Make sure 'async_ensemble_retriever.py' is in your Python path
from asyncInference import asyncRetriever


async def ask_question(retriever, question):
    """
    Ask a question to the retriever and print the answer.
    
    This function demonstrates how to use the retriever's generate_answer method
    with specific parameters for the RAG pipeline.
    
    Args:
        retriever: An instance of AsyncEnsembleMergerRetriever
        question (str): The question to be answered based on the document content
        
    Returns:
        str: The generated answer from the RAG pipeline
    """
    # Display the question for user reference
    print(f"\nQuestion: {question}")
    print("Generating answer...\n")
    
    # Call the retriever's generate_answer method with specific parameters
    # - pipeline_type: Using RetrievalQAWithSourcesChain to get sources with the answer
    # - chain_type: 'stuff' means all retrieved documents are stuffed into the prompt
    # - verbose: Set to False to avoid printing debug information
    answer = await retriever.generate_answer(
        question=question,
        pipeline_type="RetrievalQAWithSourcesChain",  # Includes source information in response
        chain_type="stuff",                          # Alternative options: 'refine', 'map_reduce'
        verbose=False                                # Set to True for debugging information
    )
    
    # Display the generated answer with formatting
    print(f"Answer:\n{answer}\n")
    print("-" * 80)  # Separator line for readability
    
    return answer


async def main():
    """
    Main async function to demonstrate AsyncEnsembleMergerRetriever usage.
    
    This function shows:
    1. How to initialize the retriever using different methods
    2. How to process a batch of predefined questions
    3. How to use the retriever in an interactive mode
    """
    # Load environment variables from .env file (API keys)
    # This is required for Google AI embeddings and Groq LLM access
    load_dotenv()
    
    # Path to your PDF document - REPLACE THIS with your actual PDF path
    pdf_path = "Data\\Attention.pdf"
    
    print("Initializing the retriever...")
    
    # OPTION 1: Use the factory method for quick setup (RECOMMENDED)
    # This is the most convenient way to initialize the retriever in one line
    retriever = await asyncRetriever.AsyncEnsembleMergerRetriever.create(
        data_path=pdf_path,                       # Path to your PDF document
        embedding_provider="Google",              # Embedding provider: "Google" or "OpenAI"
        model_name="gemma2-9b-it",                # LLM model for answer generation
        chunk_size=512,                           # Document chunk size in characters
        chunk_overlap=128                         # Overlap between chunks to maintain context
    )
    
    # OPTION 2: Step-by-step setup (COMMENTED OUT)
    # Uncomment these lines if you need more control over the initialization process
    # This approach allows more fine-grained configuration at each step
    # 
    # retriever = AsyncEnsembleMergerRetriever()
    # await retriever.setup_llm(model_name="gemma2-9b-it")
    # await retriever.setup_embeddings(embedding_provider="Google")
    # await retriever.load_and_split_documents(
    #     data_path=pdf_path,
    #     chunk_size=512,
    #     chunk_overlap=128
    # )
    # await retriever.create_retrievers()
    # await retriever.build_merger_retriever()
    
    print("Retriever initialized successfully!")
    
    # =====================================================
    # PART 1: Process a list of predefined questions
    # =====================================================
    # This section demonstrates batch processing of questions
    questions = [
        "What is the main concept described in the document?",
        "Explain the methodology outlined in section 3.",
        "Summarize the key findings from the document."
    ]
    
    # Process each question in sequence
    # Note: Could be made concurrent with asyncio.gather if needed
    for question in questions:
        await ask_question(retriever, question)
    
    # =====================================================
    # PART 2: Interactive question answering mode
    # =====================================================
    # This section allows users to ask questions interactively
    while True:
        # Get user input
        user_question = input("\nEnter your question (or 'exit' to quit): ")
        
        # Check if user wants to exit the interactive mode
        if user_question.lower() in ['exit', 'quit', 'q']:
            break
            
        # Process the user's question
        await ask_question(retriever, user_question)


def run_example():
    """
    Helper function to run the async example safely in different environments.
    
    This function handles the different ways to run async code:
    1. In standard Python scripts
    2. In environments with existing event loops (like Jupyter notebooks)
    """
    try:
        # Standard approach for most Python scripts
        # This creates a new event loop, runs the coroutine, and closes the loop
        asyncio.run(main())
    except RuntimeError as e:
        # This exception occurs when there's already an event loop running
        # Common in Jupyter notebooks and some IDEs
        print(f"Using existing event loop: {e}")
        
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        if loop.is_running():
            # If the loop is already running (e.g., in Jupyter),
            # schedule the coroutine to run
            asyncio.ensure_future(main())
        else:
            # If the loop exists but isn't running,
            # run the coroutine until completion
            loop.run_until_complete(main())


# Standard Python idiom to check if the script is being run directly
if __name__ == "__main__":
    # Start the example
    run_example()
