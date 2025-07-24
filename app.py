import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import streamlit as st
import os
import base64
import gc
import tempfile
import uuid
import re
import time
import groq
import networkx as nx
import community as community_louvain
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.vector_store = None
    st.session_state.bm25 = None
    st.session_state.graph = None
    st.session_state.conversation_history = []
    st.session_state.reranker = None

session_id = st.session_state.id
groq_client = None

@st.cache_resource
def initialize_reranker():
    """Initialize cross-encoder for reranking"""
    try:
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except Exception as e:
        st.warning(f"Could not load reranker: {e}")
        return None

@st.cache_resource
def initialize_groq_client(api_key):
    return groq.Groq(api_key=api_key)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.conversation_history = []
    gc.collect()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def format_context(docs):
    return '\n'.join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])

def rerank_documents(query, documents, reranker=None, top_k=4):
    """Rerank documents using cross-encoder"""
    if reranker is None or len(documents) <= top_k:
        return documents[:top_k]
    
    try:
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.predict(pairs)
        
        # Sort documents by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
    except Exception as e:
        st.warning(f"Reranking failed, using original order: {e}")
        return documents[:top_k]

def build_document_graph(documents):
    """Build a graph of document relationships"""
    try:
        G = nx.Graph()
        
        # Add nodes for each document
        for i, doc in enumerate(documents):
            G.add_node(i, content=doc.page_content, metadata=doc.metadata)
        
        # Calculate similarities between documents
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        contents = [doc.page_content for doc in documents]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarities = cosine_similarity(tfidf_matrix)
        
        # Add edges for similar documents (above threshold)
        threshold = 0.3
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                if similarities[i][j] > threshold:
                    G.add_edge(i, j, weight=similarities[i][j])
        
        return G
    except Exception as e:
        st.warning(f"Graph building failed: {e}")
        return None

def get_graph_enhanced_context(query, documents, graph=None, top_k=4):
    """Get context enhanced with graph relationships"""
    if graph is None:
        return documents[:top_k]
    
    try:
        # Find communities in the graph
        communities = community_louvain.best_partition(graph)
        
        # Group documents by community
        community_docs = {}
        for node_id, community_id in communities.items():
            if community_id not in community_docs:
                community_docs[community_id] = []
            if node_id < len(documents):
                community_docs[community_id].append((node_id, documents[node_id]))
        
        # Select representative documents from each community
        selected_docs = []
        for community_id, comm_docs in community_docs.items():
            # Sort by node degree (centrality in the community)
            comm_docs.sort(key=lambda x: graph.degree(x[0]), reverse=True)
            selected_docs.extend([doc for _, doc in comm_docs[:2]])  # Top 2 from each community
        
        return selected_docs[:top_k]
    except Exception as e:
        st.warning(f"Graph enhancement failed: {e}")
        return documents[:top_k]

def query_groq_api(prompt, model_name, api_key, conversation_history=None):
    try:
        groq_client = initialize_groq_client(api_key)
        
        # Build messages with conversation history
        messages = [{
            "role": "system",
            "content": "You are a helpful AI assistant. Answer the user questions based only on the retrieved context and previous conversation history when relevant."
        }]
        
        # Add conversation history (last 5 exchanges to keep context manageable)
        if conversation_history:
            for exchange in conversation_history[-5:]:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            stream=True
        )
        return chat_completion
    except Exception as e:
        return f'API request failed: {str(e)}'

def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        cleaned_chunks = [doc for doc in chunks if clean_text(doc.page_content)]
        
        # Create FAISS vector store
        embedding_model = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(cleaned_chunks, embedding_model)
        
        # Create BM25 index for hybrid search
        tokenized_docs = [doc.page_content.split() for doc in cleaned_chunks]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Build document graph for Graph RAG
        graph = build_document_graph(cleaned_chunks)
        
        # Store additional data in session state
        st.session_state.bm25 = bm25
        st.session_state.graph = graph
        st.session_state.documents = cleaned_chunks
        
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

with st.sidebar:
    st.markdown('<div class="sidebar-header">🤖 RAG Buddy Configuration</div>', unsafe_allow_html=True)
    
    # API Key input
    api_key_input = st.text_input("🔑 Enter your Groq API Key:", type="password", key="api_key_input")
    
    # Model selection with descriptions
    st.markdown("### 🧠 Select AI Model")
    model_descriptions = {
        "moonshotai/kimi-k2-instruct": "🌙 Kimi K2 - Advanced instruction following",
        "meta-llama/llama-4-maverick-17b-128e-instruct": "🦙 Llama 4 Maverick - High performance 17B model",
        "meta-llama/llama-4-scout-17b-16e-instruct": "🔍 Llama 4 Scout - Optimized for reasoning",
        "qwen/qwen3-32b": "🔮 Qwen3 32B - Large context understanding"
    }
    
    model_option = st.selectbox(
        "Choose your model:",
        list(model_descriptions.keys()),
        format_func=lambda x: model_descriptions[x]
    )
    
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
    
    # Features showcase
    st.markdown("### ✨ Advanced Features")
    st.markdown("""
    <div style="margin: 0.5rem 0;">
        <span class="feature-badge">🔄 Retrieval Reranking</span>
        <span class="feature-badge">🕸️ Graph RAG</span>
        <span class="feature-badge">💬 Multi-turn Chat</span>
        <span class="feature-badge">🔍 Hybrid Search</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf", help="Upload a PDF file to start chatting with your document")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Processing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    vector_store = process_pdf(file_path)
                    if vector_store:
                        st.session_state.file_cache[file_key] = vector_store
                        st.session_state.vector_store = vector_store
                        st.success("Document processed successfully! Ready to chat!")
                    else:
                        st.error("Failed to process document.")
                else:
                    st.session_state.vector_store = st.session_state.file_cache[file_key]
                    st.success("Document already processed. Ready to chat!")
                
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
        
    st.markdown("[Groq API Key ↗](https://console.groq.com/keys)", unsafe_allow_html=True)
st.markdown("""
<style>
    .main-title {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        text-align: center;
    }
    .feature-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .github-link {
        background: linear-gradient(135deg, rgba(45, 164, 78, 0.1), rgba(45, 164, 78, 0.05));
        border-radius: 15px;
        padding: 15px 20px;
        transition: all 0.3s ease;
        border: 2px solid rgba(45, 164, 78, 0.2);
        margin: 1rem 0;
    }
    .github-link:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(45, 164, 78, 0.15);
        border-color: rgba(45, 164, 78, 0.4);
    }
    .clear-btn {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(238, 90, 82, 0.3);
        transition: all 0.3s ease;
    }
    .clear-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(238, 90, 82, 0.4);
    }
    .stButton button {
        width: 100%;
    }
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("<h1 class='main-title'>🤖 RAG Buddy - Advanced PDF Assistant</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 1rem;">
            Chat with your PDF documents using advanced AI techniques
        </p>
        <div>
            <span class="feature-badge">🎯 Retrieval Reranking</span>
            <span class="feature-badge">🕸️ Graph RAG</span>
            <span class="feature-badge">💬 Multi-turn Conversations</span>
            <span class="feature-badge">🔍 Hybrid Search</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        github_url = "https://github.com/yogeshwaran10"
        github_image_url = "https://github.com/yogeshwaran10.png"
        
        st.markdown(f"""
        <div class="github-link">
            <a href="{github_url}" target="_blank" style="text-decoration: none;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <img src="{github_image_url}" 
                         style="width: 44px; height: 44px; border-radius: 50%; border: 2.5px solid #2DA44E; box-shadow: 0 3px 8px rgba(0,0,0,0.1);">
                    <span style="font-size: 16px; color: #2DA44E; font-weight: 600; letter-spacing: 0.3px;">
                        View My GitHub
                    </span>
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="github-link">
            <a href="{github_url}" target="_blank" style="text-decoration: none;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#2DA44E">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    <span style="font-size: 16px; color: #2DA44E; font-weight: 600;">
                        View My GitHub
                    </span>
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    st.button("Clear ↺", on_click=reset_chat, key="clear_chat")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    reset_chat()

api_key = st.session_state.get("groq_api_key", os.getenv("GROQ_API_KEY"))

if not api_key:
    st.error("Please enter your Groq API Key in the sidebar or set the GROQ_API_KEY environment variable.")
    st.stop() 
else:
    st.write(f"Using {model_option}...")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.vector_store:
        st.error("Please upload and process a PDF file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Step 1: Initial retrieval from vector store
                docs = st.session_state.vector_store.similarity_search(prompt, k=8)
                
                # Step 2: Hybrid search with BM25 if available
                if st.session_state.bm25 and hasattr(st.session_state, 'documents'):
                    query_tokens = prompt.split()
                    bm25_scores = st.session_state.bm25.get_scores(query_tokens)
                    
                    # Combine semantic and BM25 results
                    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:4]
                    bm25_docs = [st.session_state.documents[i] for i in bm25_top_indices if i < len(st.session_state.documents)]
                    
                    # Merge and deduplicate
                    all_docs = docs + bm25_docs
                    seen_content = set()
                    unique_docs = []
                    for doc in all_docs:
                        if doc.page_content not in seen_content:
                            unique_docs.append(doc)
                            seen_content.add(doc.page_content)
                    docs = unique_docs[:8]
                
                # Step 3: Graph RAG enhancement
                if st.session_state.graph:
                    docs = get_graph_enhanced_context(prompt, docs, st.session_state.graph, top_k=6)
                
                # Step 4: Reranking
                if st.session_state.reranker is None:
                    st.session_state.reranker = initialize_reranker()
                
                if st.session_state.reranker:
                    docs = rerank_documents(prompt, docs, st.session_state.reranker, top_k=4)
                else:
                    docs = docs[:4]
                
                context = format_context(docs)
                
                # Build conversation-aware prompt
                conversation_context = ""
                if st.session_state.conversation_history:
                    conversation_context = "\n### Previous Conversation Context:\n"
                    for i, exchange in enumerate(st.session_state.conversation_history[-3:]):
                        conversation_context += f"Q{i+1}: {exchange['user']}\nA{i+1}: {exchange['assistant'][:100]}...\n"
                
                full_prompt = f"""
You are an AI assistant specialized in document analysis. Answer the user's question using ONLY the retrieved information and previous conversation context when relevant.

{conversation_context}

---
### **Retrieved Context:**
{context}
---

### **Current Question:**
{prompt}

### **Instructions:**
- **ONLY** use information from the retrieved context above
- Reference previous conversation when relevant for continuity
- If the context doesn't contain enough information, say: "I don't have enough information to answer this."
- Keep responses clear, concise, and factual
- Maintain a helpful and professional tone

### **Answer:**
"""
                
                streaming_response = query_groq_api(full_prompt, model_option, api_key, st.session_state.conversation_history)
                
                if isinstance(streaming_response, str):
                    full_response = streaming_response
                else:
                    for chunk in streaming_response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    "user": prompt,
                    "assistant": full_response
                })
                
                # Keep only last 10 exchanges to manage memory
                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = st.session_state.conversation_history[-10:]
                
            except Exception as e:
                error_message = f"An error occurred while processing your request: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})