from EnsembleMergerContextualCompressionRetriever import mergerRetriever as OriginalRetriever
import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil


class EnsembleMergerRetriever(OriginalRetriever.EnsembleMergerRetriever):
    """
    Enhanced version of EnsembleMergerRetriever that supports a persistent directory for Chroma.
    All methods maintain the same signatures as the original class but with added support for
    a Chroma persistent directory to avoid SQLite errors.
    """
    
    def __init__(self, load_env=True, chroma_persist_directory=None):
        """
        Initialize the EnsembleMergerRetriever class with Chroma persistent directory support.
        
        Args:
            load_env (bool): Whether to load environment variables from .env file
            chroma_persist_directory (str): Directory to use for Chroma database persistence
                                        If None, an in-memory database will be used
        """
        super().__init__(load_env=load_env)
        self.chroma_persist_directory = chroma_persist_directory
    
    def create_retrievers(self, 
                         faiss_search_type="similarity", 
                         faiss_k_documents=3,
                         chroma_search_type="similarity", 
                         chroma_k_documents=4):
        """
        Create FAISS and Chroma vector store retrievers for the document chunks.
        This overrides the original method to add Chroma persistent directory support.
        
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
        from langchain_community.vectorstores import FAISS
        self.faiss_retriever = FAISS.from_documents(
            self.document_chunks,  # Documents to index
            self.embeddings,       # Embedding model to use
        ).as_retriever(
            search_type=faiss_search_type,  # How to search (similarity or mmr)
            search_kwargs={
                "k": faiss_k_documents  # Number of documents to retrieve
            }
        )
        print("[INFO] ----> FAISS Retriever Created successfully.....\n")
        
        # Create Chroma retriever with persistent directory if provided
        # Chroma is another vector database with different indexing characteristics
        print("[INFO] ----> Creating the Chroma Vectorstore, please wait.....")
        from langchain_chroma import Chroma
        
        # Use persistent directory if provided (helps avoid SQLite errors)
        if self.chroma_persist_directory:
            print(f"[INFO] ----> Using persistent directory for Chroma: {self.chroma_persist_directory}")
            self.chroma_retriever = Chroma.from_documents(
                self.document_chunks,           # Documents to index
                self.embeddings,                # Embedding model to use
                persist_directory=self.chroma_persist_directory  # Directory for persistence
            ).as_retriever(
                search_type=chroma_search_type,  # How to search (similarity or mmr)
                search_kwargs={
                    "k": chroma_k_documents,     # Number of documents to retrieve
                }
            )
        else:
            # Fall back to original behavior if no directory is provided
            self.chroma_retriever = Chroma.from_documents(
                self.document_chunks,  # Documents to index
                self.embeddings,       # Embedding model to use
            ).as_retriever(
                search_type=chroma_search_type,  # How to search (similarity or mmr)
                search_kwargs={
                    "k": chroma_k_documents,  # Number of documents to retrieve
                }
            )
        print("[INFO] ----> Chroma Retriever Created successfully.....\n")
        
        return self  # Return self for method chaining


# Import the modified EnsembleMergerRetriever class
# This uses our modified version that handles persistent directory for Chroma


# Set page configuration
st.set_page_config(
    page_title="RAG Ensemble Retriever",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to improve UI appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 'retriever_built' not in st.session_state:
    st.session_state.retriever_built = False
if 'document_path' not in st.session_state:
    st.session_state.document_path = None
if 'llm_setup' not in st.session_state:
    st.session_state.llm_setup = False
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = {"faiss_docs": 0, "chroma_docs": 0, "total_docs": 0}
if 'chroma_dir' not in st.session_state:
    # Create a persistent directory for Chroma database
    chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    st.session_state.chroma_dir = chroma_dir

# Main header
st.markdown("<h1 class='main-header'>📚 Ensemble RAG with Dual Vector Stores</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
<div class='info-box'>
This application implements an advanced Retrieval Augmented Generation (RAG) pipeline that combines:
<ul>
    <li>Multiple vector stores (FAISS & Chroma) for diverse retrieval</li>
    <li>Contextual compression with redundancy filtering</li>
    <li>Embedding-based relevance filtering</li>
    <li>Long context reordering for optimal document usage</li>
</ul>
Upload a PDF document, configure the parameters, and start asking questions!
</div>
""", unsafe_allow_html=True)


# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "⚙️ RAG Pipeline Setup", "❓ Ask Questions"])

with tab1:
    st.markdown("<h2 class='sub-header'>📄 Document Upload</h2>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        st.markdown(f"<div class='success-box'>✅ File '{uploaded_file.name}' uploaded successfully!</div>", unsafe_allow_html=True)
        
        # Store the file path in session state
        st.session_state.document_path = temp_file_path
        st.session_state.document_uploaded = True
        
        # Display document preview (file name and size)
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 Document Name: {uploaded_file.name}")
        with col2:
            st.info(f"📏 File Size: {round(len(uploaded_file.getvalue())/1024, 2)} KB")
    else:
        st.markdown("<div class='warning-box'>⚠️ Please upload a PDF document to proceed.</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 class='sub-header'>⚙️ RAG Pipeline Setup</h2>", unsafe_allow_html=True)
    
    # Check if a document has been uploaded
    if not st.session_state.document_uploaded:
        st.markdown("<div class='warning-box'>⚠️ Please upload a document in the 'Document Upload' tab first.</div>", unsafe_allow_html=True)
    else:
        # Create columns for different configuration sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Document Processing")
            chunk_size = st.slider("Chunk Size", min_value=128, max_value=2048, value=512, step=128, 
                                help="Size of text chunks in characters. Smaller chunks are more precise, larger chunks preserve more context.")
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=512, value=128, step=32,
                                    help="Number of characters to overlap between chunks. This helps maintain context between chunks.")
            
            st.markdown("### LLM Configuration")
            llm_model = st.selectbox("LLM Model", ["gemma2-9b-it", "llama3-70b-8192", "qwen-qwq-32b"],
                                    help="Choose the Large Language Model to generate answers.")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                                help="Controls randomness in generation. Lower values produce more deterministic output.")
            max_tokens = st.slider("Max Tokens", min_value=256, max_value=2048, value=512, step=128,
                                help="Maximum number of tokens to generate in the response.")
            
        with col2:
            st.markdown("### Embeddings and Retrieval")
            embedding_provider = st.selectbox("Embedding Provider", ["Google", "OpenAI"],
                                            help="Provider for text embeddings. Google uses Gemini embeddings, OpenAI uses their embedding model.")
            
            st.markdown("#### FAISS Retriever")
            faiss_search_type = st.selectbox("FAISS Search Type", ["similarity", "mmr"], key="faiss_search",
                                            help="'similarity' finds most similar documents. 'mmr' (Maximum Marginal Relevance) balances relevance and diversity.")
            faiss_k_documents = st.slider("FAISS Documents to Retrieve", min_value=1, max_value=10, value=3, step=1,
                                        help="Number of documents to retrieve from FAISS vector store.")
            faiss_filter_threshold = st.slider("FAISS Filter Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                                            help="Similarity threshold for filtering FAISS results. Higher = stricter filtering.")
            
            st.markdown("#### Chroma Retriever")
            chroma_search_type = st.selectbox("Chroma Search Type", ["similarity", "mmr"], key="chroma_search",
                                            help="'similarity' finds most similar documents. 'mmr' (Maximum Marginal Relevance) balances relevance and diversity.")
            chroma_k_documents = st.slider("Chroma Documents to Retrieve", min_value=1, max_value=10, value=4, step=1,
                                        help="Number of documents to retrieve from Chroma vector store.")
            chroma_filter_threshold = st.slider("Chroma Filter Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                                            help="Similarity threshold for filtering Chroma results. Higher = stricter filtering.")
        
        # Pipeline setup button
        if st.button("🔧 Setup RAG Pipeline", use_container_width=True):
            # Create a spinner to show progress
            with st.spinner("Building RAG pipeline... This may take a few minutes."):
                try:
                    # Initialize the retriever with Chroma persistent directory
                    st.session_state.retriever = EnsembleMergerRetriever(
                        chroma_persist_directory=st.session_state.chroma_dir
                    )
                    
                    # Setup LLM
                    st.session_state.retriever.setup_llm(
                        model_name=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    st.session_state.llm_setup = True
                    
                    # Load and split documents
                    progress_bar = st.progress(0)
                    st.text("Loading and splitting document...")
                    st.session_state.retriever.load_and_split_documents(
                        data_path=st.session_state.document_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    progress_bar.progress(33)
                    
                    # Setup embeddings
                    st.text("Setting up embeddings...")
                    st.session_state.retriever.setup_embeddings(embedding_provider=embedding_provider)
                    progress_bar.progress(50)
                    
                    # Create retrievers
                    st.text("Creating vector stores...")
                    st.session_state.retriever.create_retrievers(
                        faiss_search_type=faiss_search_type,
                        faiss_k_documents=faiss_k_documents,
                        chroma_search_type=chroma_search_type,
                        chroma_k_documents=chroma_k_documents
                    )
                    progress_bar.progress(75)
                    
                    # Build merger retriever
                    st.text("Building ensemble merger retriever...")
                    st.session_state.retriever.build_merger_retriever(
                        faiss_embedding_filter_threshold=faiss_filter_threshold,
                        chroma_embedding_filter_threshold=chroma_filter_threshold
                    )
                    progress_bar.progress(100)
                    
                    # Save information for visualization
                    st.session_state.visualization_data = {
                        "faiss_docs": faiss_k_documents,
                        "chroma_docs": chroma_k_documents,
                        "total_docs": len(st.session_state.retriever.document_chunks)
                    }
                    
                    # Update session state
                    st.session_state.retriever_built = True
                    
                    # Success message
                    st.markdown("<div class='success-box'>✅ RAG pipeline successfully built!</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

with tab3:
    st.markdown("<h2 class='sub-header'>❓ Ask Questions</h2>", unsafe_allow_html=True)
    
    # Check if pipeline is built
    if not st.session_state.retriever_built:
        st.markdown("<div class='warning-box'>⚠️ Please complete the RAG pipeline setup in the previous tab first.</div>", unsafe_allow_html=True)
    else:
        # User query input
        query = st.text_area("Enter your question about the document", height=100)
        
        # RAG chain type selection
        col1, col2 = st.columns(2)
        with col1:
            pipeline_type = st.selectbox(
                "Pipeline Type", 
                ["RetrievalQAChain", "RetrievalQAWithSourcesChain"],
                help="RetrievalQAChain returns just the answer. RetrievalQAWithSourcesChain includes source attribution."
            )
        with col2:
            chain_type = st.selectbox(
                "Chain Type", 
                ["stuff", "map_reduce", "refine"],
                help="'stuff': All documents in one prompt. 'map_reduce': Process documents individually then combine. 'refine': Iteratively refine the answer."
            )
        
        # Generate answer button
        if st.button("🔍 Generate Answer", use_container_width=True) and query:
            with st.spinner("Generating answer..."):
                try:
                    # Get the answer
                    start_time = time.time()
                    answer = st.session_state.retriever.generate_answer(
                        question=query,
                        pipeline_type=pipeline_type,
                        chain_type=chain_type,
                        verbose=False
                    )
                    elapsed_time = time.time() - start_time
                    
                    # Add to answers history
                    st.session_state.answers.append({
                        "question": query,
                        "answer": answer,
                        "time": elapsed_time,
                        "pipeline_type": pipeline_type,
                        "chain_type": chain_type
                    })
                    
                    # Display the answer
                    st.markdown("<div class='sub-header'>📝 Answer</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='info-box'>{answer}</div>", unsafe_allow_html=True)
                    st.info(f"⏱️ Answer generated in {elapsed_time:.2f} seconds")
                    
                    # Display visualizations
                    st.markdown("<div class='sub-header'>📊 Retrieval Visualization</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create a pie chart of document sources
                        fig, ax = plt.subplots(figsize=(6, 6))
                        labels = ['FAISS Retriever', 'Chroma Retriever']
                        sizes = [
                            st.session_state.visualization_data["faiss_docs"],
                            st.session_state.visualization_data["chroma_docs"]
                        ]
                        colors = ['#3B82F6', '#10B981']
                        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                        ax.axis('equal')
                        st.pyplot(fig)
                        
                    with col2:
                        # Create a bar chart of documents used vs. total
                        fig, ax = plt.subplots(figsize=(6, 6))
                        categories = ['Total Documents', 'Retrieved Documents']
                        values = [
                            st.session_state.visualization_data["total_docs"],
                            st.session_state.visualization_data["faiss_docs"] + st.session_state.visualization_data["chroma_docs"]
                        ]
                        ax.bar(categories, values, color=['#3B82F6', '#10B981'])
                        ax.set_ylabel('Count')
                        ax.set_title('Document Usage')
                        for i, v in enumerate(values):
                            ax.text(i, v + 0.5, str(v), ha='center')
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        # Display previous answers
        if st.session_state.answers:
            st.markdown("<div class='sub-header'>📜 Previous Questions & Answers</div>", unsafe_allow_html=True)
            
            for i, qa in enumerate(reversed(st.session_state.answers)):
                with st.expander(f"Q{len(st.session_state.answers) - i}: {qa['question'][:50]}...", expanded=False):
                    st.markdown(f"**Question:**\n{qa['question']}")
                    st.markdown(f"**Answer:**\n{qa['answer']}")
                    st.info(f"⏱️ Generated in {qa['time']:.2f}s using {qa['pipeline_type']} with {qa['chain_type']} chain")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, LangChain, and Ensemble RAG techniques")

# Cleanup temporary files when the app is closing
def cleanup():
    # Clean up the uploaded document
    if st.session_state.document_path and os.path.exists(st.session_state.document_path):
        try:
            os.unlink(st.session_state.document_path)
        except:
            pass
    
    # Clean up Chroma DB directory on exit
    # Note: Comment this out if you want to persist the Chroma DB between sessions
    if 'chroma_dir' in st.session_state and os.path.exists(st.session_state.chroma_dir):
        try:
            shutil.rmtree(st.session_state.chroma_dir)
            print(f"Removed Chroma DB directory: {st.session_state.chroma_dir}")
        except Exception as e:
            print(f"Error removing Chroma DB directory: {e}")


# Register the cleanup function to be called when the app is closing
import atexit
atexit.register(cleanup)