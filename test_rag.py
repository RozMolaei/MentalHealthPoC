# test_rag.py - Run this to test your RAG system
import os
from dotenv import load_dotenv
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

def test_rag_system():
    print("🧪 Testing RAG System Components...")
    
    # Test 1: Environment variables
    print("\n1. Testing environment variables...")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key.startswith('sk-'):
        print("✅ OpenAI API key loaded successfully")
    else:
        print("❌ OpenAI API key not found or invalid")
        return
    
    # Test 2: Vector store loading
    print("\n2. Testing vector store...")
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
        print("✅ Vector store loaded successfully")
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return
    
    # Test 3: Retrieval
    print("\n3. Testing retrieval...")
    test_query = "My patient has anxiety and trouble sleeping"
    try:
        results = vector_store.similarity_search(test_query, k=2)
        print(f"✅ Retrieved {len(results)} similar cases")
        print(f"   Example case: {results[0].metadata['patient_context'][:60]}...")
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return
    
    # Test 4: OpenAI API
    print("\n4. Testing OpenAI generation...")
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'RAG test successful!' in exactly 3 words."}
            ],
            max_tokens=10
        )
        print(f"✅ OpenAI response: {response.choices[0].message.content}")
        print(f"   Cost: ~$0.001 (very cheap!)")
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return
    
    print("\n🎉 ALL TESTS PASSED! Your RAG system is ready!")
    print("🚀 Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    test_rag_system()