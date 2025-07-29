import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Mental Health Counselor Assistant",
    page_icon="🧠",
    layout="wide"
)

# Warning banner
st.error("⚠️ EDUCATIONAL PROTOTYPE - NOT MEDICAL ADVICE")

# Title and description
st.title("🧠 Mental Health Counselor Assistant")
st.markdown("""
**AI-Powered RAG System for Mental Health Counselors**

This system combines semantic search with AI generation to help counselors find relevant examples 
and receive synthesized guidance from 3,512 real counseling conversations.
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

def generate_counselor_advice(query, retrieved_cases):
    """Generate AI advice based on retrieved cases"""
    try:
        # Set up OpenAI client
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create context from retrieved cases
        context = ""
        for i, case in enumerate(retrieved_cases, 1):
            context += f"\n**Case {i}:**\n"
            context += f"Patient: {case.metadata['patient_context']}\n"
            context += f"Counselor: {case.page_content[:500]}...\n"
        
        # Craft the prompt
        prompt = f"""You are an experienced clinical supervisor helping a mental health counselor.

COUNSELOR'S SITUATION:
{query}

RELEVANT EXAMPLES FROM PAST SESSIONS:
{context}

INSTRUCTIONS:
Based on the examples above, provide practical guidance for this counselor. Your response should:
1. Acknowledge the counselor's challenge
2. Suggest 2-3 specific therapeutic approaches based on the examples
3. Highlight which response style might work best (questions, validation, advice, empathy)
4. Keep response under 200 words
5. Maintain professional, supportive tone

IMPORTANT: This is guidance between professionals, not direct patient advice."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a supportive clinical supervisor with expertise in therapeutic techniques."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating advice: {str(e)}"

# Load vector store
with st.spinner("Loading AI system..."):
    vector_store = load_vector_store()

if vector_store is None:
    st.error("Failed to load the vector store. Please run the RAG pipeline notebook first.")
    st.stop()

# Check OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()

st.success("✅ AI system loaded successfully!")

# Main interface
st.header("🔍 Search & Generate Guidance")

# User input
counselor_query = st.text_area(
    "Describe the patient situation or challenge you're facing:",
    placeholder="Example: My patient feels overwhelmed by work and is showing signs of burnout...",
    height=100
)

# Options
col1, col2 = st.columns(2)
with col1:
    num_results = st.slider("Number of similar cases:", 1, 5, 3)
with col2:
    generate_advice = st.checkbox("🤖 Generate AI Guidance", value=True)

if st.button("🚀 Find Cases & Generate Advice", type="primary"):
    if counselor_query.strip():
        with st.spinner("Searching for similar cases..."):
            # Perform similarity search
            results = vector_store.similarity_search(counselor_query, k=num_results)
            
            st.header(f"📋 Found {len(results)} Similar Cases")
            
            # Display retrieved cases
            for i, doc in enumerate(results, 1):
                with st.expander(f"Case {i}: {doc.metadata['patient_context'][:60]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("👤 Patient's Concern:")
                        st.write(doc.metadata['patient_context'])
                    
                    with col2:
                        st.subheader("🩺 Counselor's Response:")
                        st.write(doc.page_content)
                    
                    st.caption(f"Response length: {doc.metadata['response_length']} characters")
            
            # Generate AI advice if requested
            if generate_advice:
                st.header("🤖 AI-Generated Guidance")
                with st.spinner("Generating personalized advice..."):
                    advice = generate_counselor_advice(counselor_query, results)
                    
                    st.markdown("### 💡 Synthesized Recommendation:")
                    st.info(advice)
                    
                    st.caption("💰 Cost: ~$0.01 per generation (well within your $10 budget)")
    else:
        st.warning("Please enter a patient situation to search for.")

# Sidebar
st.sidebar.header("📊 System Statistics")
st.sidebar.metric("Total Conversations", "3,512")
st.sidebar.metric("AI Model", "GPT-3.5-Turbo")
st.sidebar.metric("Budget Used", "< $1")

st.sidebar.header("🔧 How RAG Works")
st.sidebar.markdown("""
1. **Semantic Search**: Find similar patient situations
2. **Context Retrieval**: Get relevant counselor responses  
3. **AI Generation**: Synthesize personalized guidance
4. **Professional Output**: Receive actionable advice
""")

st.sidebar.header("⚡ Quick Examples")
if st.sidebar.button("Work Burnout"):
    st.session_state.example_query = "My patient feels overwhelmed by work and is showing signs of burnout"
    
if st.sidebar.button("Anxiety & Sleep"):
    st.session_state.example_query = "Patient has anxiety that's affecting their sleep patterns"
    
if st.sidebar.button("Relationship Issues"):
    st.session_state.example_query = "Client struggling with communication in their relationship"

# Handle example queries
if 'example_query' in st.session_state:
    counselor_query = st.session_state.example_query
    del st.session_state.example_query
    st.rerun()