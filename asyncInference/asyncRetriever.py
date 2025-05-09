from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA

from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from functools import lru_cache
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncEnsembleMergerRetriever:
    """
    An optimized asynchronous ensemble retriever that combines FAISS and Chroma 
    for improved RAG (Retrieval-Augmented Generation) performance.
    """
    
    def __init__(self, load_env=True):
        """
        Initialize the AsyncEnsembleMergerRetriever.
        
        Args:
            load_env (bool): Whether to load environment variables from .env file
        """
        if load_env:
            try:
                load_dotenv(find_dotenv())
                for key in ["GOOGLE_API_KEY", "GROQ_API_KEY"]:
                    if key in os.environ:
                        continue
                    env_value = os.getenv(key)
                    if env_value:
                        os.environ[key] = env_value
                    else:
                        logger.warning(f"Environment variable {key} not found")
            except Exception as e:
                logger.error(f"Error loading environment variables: {e}")
        
        # Initialize components as None
        self.faiss_retriever = None
        self.chroma_retriever = None
        self.embeddings = None
        self.merger_retriever = None
        self.llm_model = None
        self.documents = None
        self.document_chunks = None
    
    async def setup_llm(self, 
                       model_name: str = "gemma2-9b-it", 
                       temperature: float = 0.4, 
                       max_tokens: int = 512, 
                       top_p: float = 0.9) -> 'AsyncEnsembleMergerRetriever':
        """
        Set up the LLM model asynchronously.
        
        Args:
            model_name: Model name (options: "llama3-70b-8192", "gemma2-9b-it", "qwen-qwq-32b")
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p value for generation
            
        Returns:
            Self instance for method chaining
        """
        logger.info(f"Setting up LLM model: {model_name}")
        
        # Create the LLM model
        self.llm_model = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"top_p": top_p},
        )
        return self
    
    async def load_and_split_documents(self, 
                                      data_path: str, 
                                      chunk_size: int = 512, 
                                      chunk_overlap: int = 128) -> 'AsyncEnsembleMergerRetriever':
        """
        Load and split documents for retrieval asynchronously.
        
        Args:
            data_path: Path to the PDF document
            chunk_size: Size of chunks for splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            Self instance for method chaining
        """
        logger.info(f"Loading and splitting document: {data_path}")
        
        # Create loader
        loader = PyMuPDFLoader(data_path)
        
        # Load documents in a separate thread
        loop = asyncio.get_running_loop()
        try:
            self.documents = await loop.run_in_executor(None, loader.load)
            logger.info(f"Total pages in original document: {len(self.documents)}")
            
            # Create and configure the text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            
            # Split documents in a separate thread
            self.document_chunks = await loop.run_in_executor(
                None, 
                lambda: splitter.split_documents(self.documents)
            )
            logger.info(f"Total document chunks created: {len(self.document_chunks)}")
            
        except Exception as e:
            logger.error(f"Error loading/splitting document: {e}")
            raise
            
        return self
    
    @lru_cache(maxsize=2)  # Cache embeddings to avoid recreating the same ones
    async def setup_embeddings(self, embedding_provider: str = "Google") -> 'AsyncEnsembleMergerRetriever':
        """
        Set up embeddings for vectorstores asynchronously with caching.
        
        Args:
            embedding_provider: Provider for embeddings ("Google" or "OpenAI")
            
        Returns:
            Self instance for method chaining
        """
        logger.info(f"Setting up {embedding_provider} embeddings")
        
        try:
            if embedding_provider == "Google":
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
            elif embedding_provider == "OpenAI":
                self.embeddings = OpenAIEmbeddings()
            else:
                raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
                
            logger.info(f"{embedding_provider} embeddings loaded successfully")
        except Exception as e:
            logger.error(f"Error setting up embeddings: {e}")
            raise
            
        return self
    
    async def create_retrievers(self, 
                              faiss_search_type: str = "similarity", 
                              faiss_k_documents: int = 3,
                              chroma_search_type: str = "similarity", 
                              chroma_k_documents: int = 4) -> 'AsyncEnsembleMergerRetriever':
        """
        Create FAISS and Chroma retrievers asynchronously.
        
        Args:
            faiss_search_type: Search type for FAISS retriever
            faiss_k_documents: Number of documents to retrieve with FAISS
            chroma_search_type: Search type for Chroma retriever
            chroma_k_documents: Number of documents to retrieve with Chroma
            
        Returns:
            Self instance for method chaining
        """
        if self.document_chunks is None or self.embeddings is None:
            logger.error("Documents and embeddings must be set up first")
            raise ValueError("Documents and embeddings must be set up first")
        
        logger.info("Creating FAISS and Chroma vectorstores concurrently")
        
        loop = asyncio.get_running_loop()
        
        # Create both retrievers concurrently for better performance
        try:
            # Define tasks for concurrent execution
            faiss_task = loop.run_in_executor(
                None,
                lambda: FAISS.from_documents(self.document_chunks, self.embeddings)
            )
            
            chroma_task = loop.run_in_executor(
                None,
                lambda: Chroma.from_documents(self.document_chunks, self.embeddings)
            )
            
            # Wait for both tasks to complete
            faiss_db, chroma_db = await asyncio.gather(faiss_task, chroma_task)
            
            # Set up the retrievers
            self.faiss_retriever = faiss_db.as_retriever(
                search_type=faiss_search_type,
                search_kwargs={"k": faiss_k_documents}
            )
            logger.info("FAISS retriever created successfully")
            
            self.chroma_retriever = chroma_db.as_retriever(
                search_type=chroma_search_type,
                search_kwargs={"k": chroma_k_documents}
            )
            logger.info("Chroma retriever created successfully")
            
        except Exception as e:
            logger.error(f"Error creating retrievers: {e}")
            raise
        
        return self
    
    async def build_merger_retriever(self, 
                                   faiss_embedding_filter_threshold: float = 0.75,
                                   chroma_embedding_filter_threshold: float = 0.6) -> 'AsyncEnsembleMergerRetriever':
        """
        Build the ensemble merger retriever with contextual compression asynchronously.
        
        Args:
            faiss_embedding_filter_threshold: Similarity threshold for FAISS embeddings filter
            chroma_embedding_filter_threshold: Similarity threshold for Chroma embeddings filter
            
        Returns:
            Self instance for method chaining
        """
        if self.faiss_retriever is None or self.chroma_retriever is None:
            logger.error("FAISS and Chroma retrievers must be created first")
            raise ValueError("FAISS and Chroma retrievers must be created first")
            
        logger.info("Building ensemble merger retriever")
        
        try:
            # Create shared filters - reuse for both pipelines
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
            reordering = LongContextReorder()
            
            # Create compression retrievers for both vectorstores
            compression_retrievers = []
            
            # Configure FAISS retriever
            faiss_pipeline = DocumentCompressorPipeline(
                transformers=[
                    redundant_filter,
                    EmbeddingsFilter(
                        embeddings=self.embeddings,
                        similarity_threshold=faiss_embedding_filter_threshold
                    ),
                    reordering
                ]
            )
            compression_retrievers.append(
                ContextualCompressionRetriever(
                    base_compressor=faiss_pipeline,
                    base_retriever=self.faiss_retriever
                )
            )
            
            # Configure Chroma retriever
            chroma_pipeline = DocumentCompressorPipeline(
                transformers=[
                    redundant_filter,
                    EmbeddingsFilter(
                        embeddings=self.embeddings,
                        similarity_threshold=chroma_embedding_filter_threshold
                    ),
                    reordering
                ]
            )
            compression_retrievers.append(
                ContextualCompressionRetriever(
                    base_compressor=chroma_pipeline,
                    base_retriever=self.chroma_retriever
                )
            )
            
            # Create merger retriever
            self.merger_retriever = MergerRetriever(retrievers=compression_retrievers)
            logger.info("Ensemble merger retriever built successfully")
            
        except Exception as e:
            logger.error(f"Error building merger retriever: {e}")
            raise
            
        return self
    
    async def generate_answer(self, 
                            question: str, 
                            pipeline_type: str = "RetrievalQAWithSourcesChain", 
                            chain_type: str = "stuff", 
                            verbose: bool = False) -> str:
        """
        Generate answer using the RAG pipeline asynchronously.
        
        Args:
            question: Question to answer
            pipeline_type: Type of RAG pipeline ("RetrievalQAWithSourcesChain" or "RetrievalQA")
            chain_type: Type of chain for retrieval QA
            verbose: Whether to print the result
            
        Returns:
            Generated answer
        """
        if self.merger_retriever is None or self.llm_model is None:
            logger.error("Merger retriever and LLM must be set up first")
            raise ValueError("Merger retriever and LLM must be set up first")
            
        logger.info(f"Running {pipeline_type} pipeline for question: {question}")
        
        loop = asyncio.get_running_loop()
        
        try:
            if pipeline_type == "RetrievalQAWithSourcesChain":
                rag_pipeline = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=self.llm_model,
                    chain_type=chain_type,
                    retriever=self.merger_retriever,
                    return_source_documents=True
                )
                
                # Run in a separate thread to not block the event loop
                generated_answer = await loop.run_in_executor(
                    None,
                    lambda: rag_pipeline.invoke(question)
                )
                
                result = generated_answer["answer"]
                
            elif pipeline_type == "RetrievalQA":
                rag_pipeline = RetrievalQA.from_chain_type(
                    llm=self.llm_model,
                    chain_type=chain_type,
                    retriever=self.merger_retriever,
                    return_source_documents=True
                )
                
                # Run in a separate thread to not block the event loop
                generated_answer = await loop.run_in_executor(
                    None,
                    lambda: rag_pipeline.invoke(question)
                )
                    
                result = generated_answer["result"]
            else:
                raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
            
            logger.info("Answer generated successfully")
            
            if verbose:
                logger.info(f"Result:\n{result}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Failed to generate answer due to: {str(e)}"
    
    def print_documents(self, docs: List[Any]) -> None:
        """
        Print documents for inspection.
        
        Args:
            docs: List of documents to print
        """
        for index, doc in enumerate(docs):
            logger.info(f"Document {index + 1}:\n{'-' * 50}\n{doc.page_content}\n{'-' * 50}")
    
    @classmethod
    async def create(cls, 
                    data_path: str, 
                    embedding_provider: str = "Google",
                    model_name: str = "gemma2-9b-it", 
                    chunk_size: int = 512, 
                    chunk_overlap: int = 128,
                    faiss_search_type: str = "similarity", 
                    faiss_k_documents: int = 3,
                    chroma_search_type: str = "similarity", 
                    chroma_k_documents: int = 4,
                    faiss_embedding_filter_threshold: float = 0.75,
                    chroma_embedding_filter_threshold: float = 0.6) -> 'AsyncEnsembleMergerRetriever':
        """
        Factory method to create and setup the retriever in one go.
        
        Args:
            data_path: Path to the PDF document
            embedding_provider: Provider for embeddings ("Google" or "OpenAI")
            model_name: Model name for LLM
            chunk_size: Size of chunks for splitting
            chunk_overlap: Overlap between chunks
            faiss_search_type: Search type for FAISS retriever
            faiss_k_documents: Number of documents to retrieve with FAISS
            chroma_search_type: Search type for Chroma retriever
            chroma_k_documents: Number of documents to retrieve with Chroma
            faiss_embedding_filter_threshold: Similarity threshold for FAISS embeddings
            chroma_embedding_filter_threshold: Similarity threshold for Chroma embeddings
            
        Returns:
            Fully configured instance
        """
        logger.info(f"Creating AsyncEnsembleMergerRetriever for {data_path}")
        instance = cls()
        
        # Setup LLM and embeddings can be done in parallel
        setup_llm_task = instance.setup_llm(model_name=model_name)
        setup_embeddings_task = instance.setup_embeddings(embedding_provider=embedding_provider)
        
        # Wait for both to complete
        await asyncio.gather(setup_llm_task, setup_embeddings_task)
        
        # Load and split documents (depends on the previous tasks)
        await instance.load_and_split_documents(
            data_path=data_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # These operations depend on documents and embeddings
        await instance.create_retrievers(
            faiss_search_type=faiss_search_type,
            faiss_k_documents=faiss_k_documents,
            chroma_search_type=chroma_search_type,
            chroma_k_documents=chroma_k_documents
        )
        
        await instance.build_merger_retriever(
            faiss_embedding_filter_threshold=faiss_embedding_filter_threshold,
            chroma_embedding_filter_threshold=chroma_embedding_filter_threshold
        )
        
        logger.info("AsyncEnsembleMergerRetriever fully configured")
        return instance


async def main():
    """Main function to demonstrate usage of AsyncEnsembleMergerRetriever."""
    try:
        # Use the factory method for cleaner setup
        retriever = await AsyncEnsembleMergerRetriever.create(
            data_path="Data\\ReAct.pdf",
            embedding_provider="OpenAI",
            model_name="gemma2-9b-it",
            chunk_size=512,
            chunk_overlap=128
        )
        
        # Generate answer
        query = "Explain the designing of the algorithm named as 'ReAct' as mentioned in the Paper."
        answer = await retriever.generate_answer(
            question=query,
            pipeline_type="RetrievalQAWithSourcesChain",
            chain_type="stuff",
            verbose=True
        )
        
        logger.info(f"Final Answer:\n{answer}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")


def run_async_retriever():
    """
    Helper function to safely run the async code in different environments.
    Handles cases where it might be called from inside another event loop.
    """
    try:
        # For standard Python scripts (when no event loop is running)
        asyncio.run(main())
    except RuntimeError as e:
        # For Jupyter notebooks or environments with existing event loops
        logger.info(f"Using existing event loop: {e}")
        loop = asyncio.get_event_loop()
        
        if loop.is_running():
            # If we're in a Jupyter notebook with running loop
            future = asyncio.ensure_future(main())
            # You might need to add loop.run_until_complete(future) in some environments
        else:
            # If loop exists but isn't running
            loop.run_until_complete(main())


# if __name__ == "__main__":
#     run_async_retriever()