import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Counselor Assistant",
    page_icon="🧠",
    layout="wide"
)

# Warning banner (as recommended in roadmap)
st.error("⚠️ EDUCATIONAL PROTOTYPE - NOT MEDICAL ADVICE")

# Title and description
st.title("🧠 Mental Health Counselor Assistant")
st.markdown("""
**Find relevant counseling strategies and responses from past successful sessions**

This RAG-powered system helps mental health counselors by retrieving similar patient situations 
and corresponding therapeutic responses from a database of 3,512 real counseling conversations.
""")

@st.cache_resource
def load_vector_store():
    """Load the FAISS vector store (cached for performance)"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = FAISS.load_local(
            "data/faiss_index", 
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Load vector store
with st.spinner("Loading AI system..."):
    vector_store = load_vector_store()

if vector_store is None:
    st.error("Failed to load the vector store. Please run the RAG pipeline notebook first.")
    st.stop()

st.success("✅ AI system loaded successfully!")

# Main interface
st.header("Search for Similar Patient Situations")

# User input
counselor_query = st.text_area(
    "Describe the patient situation or challenge you're facing:",
    placeholder="Example: My patient is experiencing anxiety about work and can't sleep at night...",
    height=100
)

# Number of results
num_results = st.slider("Number of similar cases to retrieve:", 1, 10, 3)

if st.button("🔍 Find Similar Cases", type="primary"):
    if counselor_query.strip():
        with st.spinner("Searching for similar cases..."):
            # Perform similarity search
            results = vector_store.similarity_search(counselor_query, k=num_results)
            
            st.header(f"📋 Found {len(results)} Similar Cases")
            
            # Display results
            for i, doc in enumerate(results, 1):
                with st.expander(f"Case {i}: {doc.metadata['patient_context'][:60]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("👤 Patient's Concern:")
                        st.write(doc.metadata['patient_context'])
                    
                    with col2:
                        st.subheader("🩺 Counselor's Response:")
                        st.write(doc.page_content)
                    
                    # Metadata
                    st.caption(f"Response length: {doc.metadata['response_length']} characters")
    else:
        st.warning("Please enter a patient situation to search for.")

# Sidebar with statistics
st.sidebar.header("📊 System Statistics")
st.sidebar.metric("Total Conversations", "3,512")
st.sidebar.metric("Response Types Analyzed", "5")
st.sidebar.metric("Average Response Length", "1,027 chars")

st.sidebar.header("🔧 How It Works")
st.sidebar.markdown("""
1. **Your query** is converted to embeddings
2. **Semantic search** finds similar patient contexts
3. **Relevant responses** are retrieved from counselors
4. **Multiple approaches** show different therapeutic styles
""")

st.sidebar.header("⚡ Quick Examples")
if st.sidebar.button("Anxiety & Sleep Issues"):
    st.session_state.example_query = "My patient has anxiety and trouble sleeping"
    
if st.sidebar.button("Relationship Problems"):
    st.session_state.example_query = "Patient struggling with relationship conflicts"
    
if st.sidebar.button("Work Stress"):
    st.session_state.example_query = "Feeling overwhelmed with work responsibilities"

# Handle example queries
if 'example_query' in st.session_state:
    counselor_query = st.session_state.example_query
    del st.session_state.example_query
    st.rerun()