# RAG Buddy - PDF RAG Streamlit Application

![Lottie Animation](./assets/Dev.gif)

Welcome to **RAG Buddy**, an interactive application built with Streamlit that provides a retrieval-augmented generation (RAG) experience for your PDF documents. This project integrates Groq's API for dynamic chat completions and leverages FAISS for document vector retrieval, offering a smooth and efficient way to ask questions about your documents.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

Exploring RAG : Built a small project called RAG Buddy.

It answers queries based on the content of your uploaded PDF documents. No hallucinations.

The process is simple: upload a document, the app extracts and chunks the text, embeds it into vectors using sentence-transformers/all-MiniLM-L6-v2, and retrieves the most relevant pieces to answer your question using an LLM.

It also keeps the model safe from prompt injection, since it only replies using what’s actually in your document and nothing outside it.

---

## How It Works

1. **PDF Upload & Processing:**
   - The application allows users to upload a PDF file.
   - Uploaded files are temporarily stored and then processed using LangChain's **PyPDFLoader**.
   - The PDF is split into manageable chunks using a **RecursiveCharacterTextSplitter**.

2. **Document Vectorization:**
   - Each chunk is cleaned and vectorized using the **HuggingFaceEmbeddings** model.
   - The vectorized chunks are stored in a **FAISS** vector store for efficient similarity search.

3. **Retrieval-Augmented Generation:**
   - When a question is posed, the application retrieves the most relevant document segments based on similarity search.
   - These segments serve as context for the AI assistant.
   - The full context and user query are passed as a prompt to Groq's API, ensuring the response is built solely from the uploaded document's content.

4. **Dynamic Chat Experience:**
   - The application offers a chat interface built with Streamlit’s chat components.
   - Users can ask questions, receive answers generated using the Groq API.

5. **Clear & Interactive UI:**
   - The user-friendly interface showcases the PDF preview and provides a clear, concise chat interface.
   - A "Clear" button allows users to reset the session and start fresh at any time.

---

## Key Features

- **Interactive Chat Interface:** Ask questions related to your PDF with a chat-based interaction.
- **Efficient Document Processing:** Utilizes LangChain for splitting and cleaning PDFs.
- **Vectorized Search:** Employs FAISS and HuggingFace embeddings for rapid and accurate similarity searches.
- **Dynamic AI Responses:** Integrates with Groq's API ensuring responses are derived strictly from the document content.
- **Clean and Professional UI:** Built with Streamlit, offering a smooth user experience.
- **Developer-Friendly:** Open-source code with an MIT License, welcoming improvements and contributions!

---

## Installation & Setup

### Prerequisites

- Python 3.11 (or compatible version)
- Streamlit
- Groq API Key (you can get one [here](https://console.groq.com/keys))
- Necessary packages as listed in `requirements.txt`

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/yogeshwaran10-pdf-rag-streamlit.git
   cd yogeshwaran10-pdf-rag-streamlit
   ```

2. **Set Up the Environment:**

   It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**

   Create a `.env` file in the root directory and set your Groq API Key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Application:**

   ```bash
   streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
   ```

---

## Usage

1. **Launch the App:**
   - Access the local application through your browser (usually at `http://localhost:8501`).

2. **Upload Your PDF:**
   - Use the sidebar to upload a PDF file. The file will be processed and vectorized.

3. **Ask a Question:**
   - Enter your query in the chat input field. The assistant will respond by retrieving relevant content from the uploaded document.

4. **View PDF Preview:**
   - A preview of the PDF is available within the interface.

5. **Reset Session:**
   - Use the "Clear ↺" button to reset chat history and start a new session.

---

## Demo
Check out the live demo of RAG Buddy here: [Live Demo]([https://your-demo-link.com](https://ragbuddy.streamlit.app/))
![RAG Buddy Animation](./assets/Star.gif)

Hey! ⭐ Shine a star - it helps!

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Contact

For any questions or further guidance, feel free to reach out via GitHub:

- **GitHub:** [@yogeshwaran10](https://github.com/yogeshwaran10)

---

