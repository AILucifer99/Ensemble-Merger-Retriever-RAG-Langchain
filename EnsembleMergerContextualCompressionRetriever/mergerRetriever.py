from langchain_community.document_loaders import (
    PyPDFLoader, PyMuPDFLoader  # Loaders for PDF documents
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google's embedding model
from langchain_openai import OpenAIEmbeddings  # OpenAI's embedding model
from langchain_groq import ChatGroq  # Groq's LLM API integration

from langchain_community.vectorstores import FAISS  # Facebook AI Similarity Search for vector storage
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting documents into chunks
from langchain.retrievers import ContextualCompressionRetriever  # Advanced retriever with compression
from langchain_chroma import Chroma  # Chroma vector database
from langchain.retrievers.document_compressors import DocumentCompressorPipeline  # Pipeline for document compression
from langchain_community.document_transformers import EmbeddingsRedundantFilter  # Filters out redundant documents
from langchain.retrievers.document_compressors import EmbeddingsFilter  # Filters documents based on relevance
from langchain_community.document_transformers import LongContextReorder  # Reorders documents for better context usage
from langchain.retrievers.merger_retriever import MergerRetriever  # Combines multiple retrievers
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA  # QA chains with retrieval
import langchain  # Import langchain main package
from dotenv import load_dotenv, find_dotenv  # For loading environment variables
import os
langchain.debug=True  # Enable debug mode to see detailed execution logs


class EnsembleMergerRetriever:
    """
    A class that implements an ensemble approach to document retrieval by combining
    multiple vector stores (FAISS and Chroma) with advanced filtering techniques.
    
    This class provides a full RAG (Retrieval Augmented Generation) pipeline that:
    1. Loads and chunks documents
    2. Creates embeddings
    3. Stores them in two different vector databases
    4. Implements contextual compression and filtering
    5. Merges results from both retrievers
    6. Generates answers using an LLM
    """
    
    def __init__(self, load_env=True):
        """
        Initialize the EnsembleMergerRetriever class.
        
        Args:
            load_env (bool): Whether to load environment variables from .env file
                            Set to True to automatically load API keys from a .env file
        """
        # Load environment variables if requested
        if load_env:
            load_dotenv(find_dotenv())
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Set Google API key
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Set Groq API key
        
        # Initialize class attributes
        self.faiss_retriever = None  # Will hold the FAISS retriever instance
        self.chroma_retriever = None  # Will hold the Chroma retriever instance
        self.embeddings = None  # Will hold the embedding model
        self.merger_retriever = None  # Will hold the combined retriever
        self.llm_model = None  # Will hold the LLM model
        self.documents = None  # Will hold the original loaded documents
        self.document_chunks = None  # Will hold the chunked documents
    
    def setup_llm(self, model_name="gemma2-9b-it", temperature=0.4, max_tokens=512, top_p=0.9):
        """
        Set up the Large Language Model (LLM) for generating answers.
        
        Args:
            model_name (str): Model name to use from Groq's API
                              Options: "llama3-70b-8192", "gemma2-9b-it", "qwen-qwq-32b"
            temperature (float): Controls randomness in generation (0.0 = deterministic, 1.0 = creative)
            max_tokens (int): Maximum number of tokens to generate in the response
            top_p (float): Nucleus sampling parameter - only consider tokens with top_p probability mass
            
        Returns:
            EnsembleMergerRetriever: self instance for method chaining
        """
        # Initialize the ChatGroq model with specified parameters
        self.llm_model = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={
                "top_p": top_p,
            },
        )
        return self  # Return self for method chaining
    
    def load_and_split_documents(self, data_path, chunk_size=512, chunk_overlap=128):
        """
        Load PDF document and split it into manageable chunks for vectorization.
        
        Args:
            data_path (str): Path to the PDF document to be loaded
            chunk_size (int): Size of text chunks in characters for splitting
            chunk_overlap (int): Number of characters to overlap between chunks
                                (helps maintain context between chunks)
            
        Returns:
            EnsembleMergerRetriever: self instance for method chaining
        """
        print("\n[INFO] ----> Loading and splitting the document, please wait....\n")
        
        # Load documents using PyMuPDFLoader (faster than PyPDFLoader)
        self.documents = PyMuPDFLoader(data_path).load()
        print("\n[INFO] ----> Total Pages in the original document are: {}".format(len(self.documents)))
        
        # Split documents into chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Use standard Python len function to measure text size
        )
        
        # Apply the splitter to our documents
        self.document_chunks = splitter.split_documents(self.documents)
        print("[INFO] ----> Total Document Chunks Created are: {}\n".format(len(self.document_chunks)))
        
        return self  # Return self for method chaining
    
    def setup_embeddings(self, embedding_provider="Google"):
        """
        Set up the embedding model for vectorizing text chunks.
        
        Args:
            embedding_provider (str): Provider for embeddings - "Google" or "OpenAI"
                                     Determines which embedding model to use
            
        Returns:
            EnsembleMergerRetriever: self instance for method chaining
        """
        # Set up Google's embedding model
        if embedding_provider == "Google":
            print("[INFO] ----> Using the Google-AI Embedding model...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"  # Google's embedding model
            )
            print("[INFO] ----> Google-AI Embedding model loaded.\n")
        
        # Set up OpenAI's embedding model
        elif embedding_provider == "OpenAI":
            print("[INFO] ----> Using the Open-AI Embedding model...")
            self.embeddings = OpenAIEmbeddings()  # Default OpenAI embedding model
            print("[INFO] ----> OpenAI Embedding model loaded.\n")
        
        return self  # Return self for method chaining
    
    def create_retrievers(self, 
                         faiss_search_type="similarity", 
                         faiss_k_documents=3,
                         chroma_search_type="similarity", 
                         chroma_k_documents=4):
        """
        Create FAISS and Chroma vector store retrievers for the document chunks.
        Using two different vector stores adds diversity to the retrieval process.
        
        Args:
            faiss_search_type (str): Search type for FAISS retriever ("similarity" or "mmr")
            faiss_k_documents (int): Number of documents to retrieve with FAISS
            chroma_search_type (str): Search type for Chroma retriever ("similarity" or "mmr")
            chroma_k_documents (int): Number of documents to retrieve with Chroma
            
        Returns:
            EnsembleMergerRetriever: self instance for method chaining
        """
        # Validate that prerequisites are set up
        if self.document_chunks is None or self.embeddings is None:
            print("[ERROR] ----> Documents and embeddings must be set up first")
            return self
            
        # Create FAISS retriever
        # FAISS is an efficient similarity search library
        print("[INFO] ----> Creating the FAISS Vectorstore, please wait.....")
        self.faiss_retriever = FAISS.from_documents(
            self.document_chunks,  # Documents to index
            self.embeddings  # Embedding model to use
        ).as_retriever(
            search_type=faiss_search_type,  # How to search (similarity or mmr)
            search_kwargs={
                "k": faiss_k_documents  # Number of documents to retrieve
            }
        )
        print("[INFO] ----> FAISS Retriever Created successfully.....\n")
        
        # Create Chroma retriever
        # Chroma is another vector database with different indexing characteristics
        print("[INFO] ----> Creating the Chroma Vectorstore, please wait.....")
        self.chroma_retriever = Chroma.from_documents(
            self.document_chunks,  # Documents to index
            self.embeddings,  # Embedding model to use
        ).as_retriever(
            search_type=chroma_search_type,  # How to search (similarity or mmr)
            search_kwargs={
                "k": chroma_k_documents,  # Number of documents to retrieve
            }
        )
        print("[INFO] ----> Chroma Retriever Created successfully.....\n")
        
        return self  # Return self for method chaining
    
    def build_merger_retriever(self, 
                              faiss_embedding_filter_threshold=0.75,
                              chroma_embedding_filter_threshold=0.6):
        """
        Build the ensemble merger retriever with contextual compression.
        This combines the results from both retrievers and applies advanced filtering.
        
        Args:
            faiss_embedding_filter_threshold (float): Similarity threshold for FAISS embeddings filter
                                                    Higher values mean more strict filtering
            chroma_embedding_filter_threshold (float): Similarity threshold for Chroma embeddings filter
                                                     Higher values mean more strict filtering
            
        Returns:
            EnsembleMergerRetriever: self instance for method chaining
        """
        # Validate that prerequisites are set up
        if self.faiss_retriever is None or self.chroma_retriever is None:
            print("[ERROR] ----> FAISS and Chroma retrievers must be created first")
            return self
            
        print("[INFO] ----> Building the ensemble merger retriever, please wait.....")
        
        # Create shared filters used by both retrievers
        # Removes redundant documents based on embedding similarity
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        
        # Reorders documents to optimize for long context windows
        reordering = LongContextReorder()
        
        # Create relevance filter for FAISS with its own threshold
        # This filter removes documents that aren't similar enough to the query
        relevant_filter_faiss = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=faiss_embedding_filter_threshold,
        )
        
        # Create relevance filter for Chroma with its own threshold
        relevant_filter_chroma = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=chroma_embedding_filter_threshold,
        )
        
        # Create compressor pipeline for FAISS
        # This applies the filters in sequence to the FAISS results
        document_pipeline_compressor_faiss = DocumentCompressorPipeline(
            transformers=[
                redundant_filter,  # First remove redundant documents
                relevant_filter_faiss,  # Then filter by relevance
                reordering,  # Finally reorder for optimal context usage
            ]
        )
        
        # Create compressor pipeline for Chroma
        # This applies the filters in sequence to the Chroma results
        document_pipeline_compressor_chroma = DocumentCompressorPipeline(
            transformers=[
                redundant_filter,  # First remove redundant documents
                relevant_filter_chroma,  # Then filter by relevance
                reordering,  # Finally reorder for optimal context usage
            ]
        )
        
        # Create contextual compression retriever for FAISS
        # This enhances the retriever with the compression pipeline
        compression_retriever_with_faiss = ContextualCompressionRetriever(
            base_compressor=document_pipeline_compressor_faiss,
            base_retriever=self.faiss_retriever,
        )
        
        # Create contextual compression retriever for Chroma
        # This enhances the retriever with the compression pipeline
        compression_retriever_with_chroma = ContextualCompressionRetriever(
            base_compressor=document_pipeline_compressor_chroma,
            base_retriever=self.chroma_retriever,
        )
        
        # Create merger retriever that combines results from both retrievers
        # This gives us the benefits of both vector stores
        self.merger_retriever = MergerRetriever(
            retrievers=[
                compression_retriever_with_faiss,
                compression_retriever_with_chroma
            ]
        )
        
        print("[INFO] ----> Ensemble merger retriever built successfully.....")
        return self  # Return self for method chaining
    
    def generate_answer(self, question, pipeline_type="RetrievalQAWithSourcesChain", chain_type="stuff", verbose=False):
        """
        Generate an answer to the given question using the RAG pipeline.
        
        Args:
            question (str): Question to answer
            pipeline_type (str): Type of RAG pipeline to use
                               "RetrievalQAWithSourcesChain" - Returns answer with sources
                               "RetrievalQAChain" - Returns just the answer
            chain_type (str): Type of chain for retrieval QA
                            "stuff" - Stuffs all documents into context
                            "map_reduce" - Processes documents individually then combines
                            "refine" - Iteratively refines the answer
            verbose (bool): Whether to print the result to console
            
        Returns:
            str: Generated answer to the question
        """
        # Validate that prerequisites are set up
        if self.merger_retriever is None or self.llm_model is None:
            print("[ERROR] ----> Merger retriever and LLM must be set up first")
            return None
            
        print(f"[INFO] ----> Running the {pipeline_type} Pipeline.....\n")
        
        # Use RetrievalQAWithSourcesChain - provides answer with source attribution
        if pipeline_type == "RetrievalQAWithSourcesChain":
            rag_pipeline = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm_model,  # LLM to use for generating answers
                chain_type=chain_type,  # How to process multiple documents
                retriever=self.merger_retriever,  # Retriever to use
                return_source_documents=True  # Include source documents in response
            )
            print(f"[INFO] ----> Generating the answer for:\nQuestion: {question}")
            generated_answer = rag_pipeline.invoke(question)
            result = generated_answer["answer"]  # Extract answer from response
            
        # Use RetrievalQAChain - simpler chain without source attribution
        elif pipeline_type == "RetrievalQAChain":
            rag_pipeline = RetrievalQA.from_chain_type(
                llm=self.llm_model,  # LLM to use for generating answers
                chain_type=chain_type,  # How to process multiple documents
                retriever=self.merger_retriever,  # Retriever to use
                return_source_documents=True  # Include source documents in response
            )
            print(f"[INFO] ----> Generating the answer for:\nQuestion: {question}")
            generated_answer = rag_pipeline.invoke(question)
            result = generated_answer["result"]  # Extract result from response
        
        print("[INFO] ----> Result Answer generated.....\n")
        
        # Print the result if verbose mode is enabled
        if verbose:
            print(f"Result:\n{result}")
            
        return result  # Return the generated answer
    
    def print_documents(self, docs):
        """
        Print documents for inspection and debugging.
        
        Args:
            docs (list): List of documents to print
        """
        # Format and print each document with a separator line
        print(
            f"\n{'-' * 100}\n".join(
                [
                    f"Document {index + 1}:- \n\n" + d.page_content for index, d in enumerate(docs)
                ]
            )
        )


# Example usage - This will only run if the script is executed directly
if __name__ == "__main__":
    
    local_inference = "False"  # Flag to control whether to run the full pipeline
    if local_inference == "True" :
        print("[INFO] ----> Running on the local inference.")
        # Create the ensemble retriever instance
        retriever = EnsembleMergerRetriever()
        
        # Set up the LLM model - using Gemma 2 9B instruction tuned model
        retriever.setup_llm(model_name="gemma2-9b-it")
        
        # Load and split the PDF document
        retriever.load_and_split_documents(
            data_path="Data\\ReAct.pdf",
            chunk_size=512,
            chunk_overlap=128
        )
        
        # Set up embeddings using Google's embedding model
        retriever.setup_embeddings(embedding_provider="OpenAI")
        
        # Create the FAISS and Chroma retrievers
        retriever.create_retrievers(
            faiss_search_type="similarity",
            faiss_k_documents=3,
            chroma_search_type="similarity",
            chroma_k_documents=4
        )
        
        # Build the ensemble merger retriever with filtering
        retriever.build_merger_retriever(
            faiss_embedding_filter_threshold=0.75,
            chroma_embedding_filter_threshold=0.6
        )
        
        # Generate an answer to a specific question
        query = "Explain in details about the Chain of thought prompting as mentioned in ReAct Paper."
        answer = retriever.generate_answer(
            question=query,
            pipeline_type="RetrievalQAChain",
            chain_type="stuff",
            verbose=False
        )
        
        # Print the final answer
        print("\n")
        print("===" * 100)
        print(f"Final Answer:\n{answer}")
    else :
        print("[Imported Successfully.....]")  # Confirmation message when imported as a module
