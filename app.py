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
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.vector_store = None

session_id = st.session_state.id
groq_client = None

@st.cache_resource
def initialize_groq_client(api_key):
    return groq.Groq(api_key=api_key)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
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

def query_groq_api(prompt, model_name, api_key):
    try:
        groq_client = initialize_groq_client(api_key)
        chat_completion = groq_client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "You are a helpful AI assistant. Answer the user questions based only on the retrieved context."
            }, {
                "role": "user",
                "content": prompt
            }],
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
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(cleaned_chunks, embedding_model)
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

with st.sidebar:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("")
    
    api_key_input = st.text_input("Enter your Groq API Key:", type="password", key="api_key_input")
    model_option = st.selectbox("Select Model", ["meta-llama/llama-4-scout-17b-16e-instruct","llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b"])
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
    
    st.header("Attach Documents Here")
    uploaded_file = st.file_uploader("Insert a .pdf file here", type="pdf")

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
        font-size: 2.2rem;
        font-weight: 600;
        color: #0066cc;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 102, 204, 0.1);
        text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.05);
    }
    .github-link {
        background: linear-gradient(to right, rgba(45, 164, 78, 0.05), rgba(45, 164, 78, 0.02));
        border-radius: 12px;
        padding: 12px 15px;
        transition: all 0.3s ease;
        border: 1px solid rgba(45, 164, 78, 0.15);
    }
    .github-link:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(45, 164, 78, 0.1);
    }
    .clear-btn {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
        transition: all 0.2s ease;
    }
    .clear-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("<h2 class='main-title'>Ask Anything About Your File</h2>", unsafe_allow_html=True)
    
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
                docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                context = format_context(docs)
                
                full_prompt = f"""
You are an AI assistant. Answer the user's question **ONLY** using the retrieved information provided below.

---
### **Retrieved Context:**
{context}
---

### **User Query:**
{prompt}

### **Instructions:**
- **Do not** use any external knowledge.
- If the retrieved context does not contain enough information, politely say:  
  "I don't have enough information to answer this."
- Keep your response clear, concise, and factual.
- Maintain a **helpful and professional** tone.

### **Answer:**
"""
                
                streaming_response = query_groq_api(full_prompt, model_option, api_key)
                
                if isinstance(streaming_response, str):
                    full_response = streaming_response
                else:
                    for chunk in streaming_response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_message = f"An error occurred while processing your request: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})