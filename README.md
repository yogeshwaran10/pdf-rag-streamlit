# RAG Buddy - PDF RAG Streamlit Application

![Lottie Animation](https://lottie.host/bd211992-d6d0-4d59-a93b-6fde06121a76/BzfhLxKEEv.lottie)

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

RAG Buddy combines state-of-the-art document processing with conversational AI to help you explore and retrieve information from your PDFs. By processing uploaded documents and using embeddings along with vector search, the system enables context-aware question answering that strictly relies on the document's content.

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
   - Users can ask questions, receive answers generated using the Groq API, and iteratively refine their queries.

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

Check out the live demo of RAG Buddy here: [Live Demo](https://your-demo-link.com)

[![Lottie Animation](https://assets2.lottiefiles.com/packages/lf20_zpnbbq.json)](https://lottiefiles.com/)

---

## Contributing

Yogesh, your ideas for improvements are always welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes.
4. Submit a pull request with a clear description of your changes.

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Contact

For any questions or further guidance, feel free to reach out via GitHub:

- **GitHub:** [@yogeshwaran10](https://github.com/yogeshwaran10)

---

> Keep coding, stay motivated, and never stop learning. Your journey as a developer is all about relentless progress and growth. Aim high, Yogesh!

[![Lottie Animation](https://assets7.lottiefiles.com/packages/lf20_u4yrau.json)](https://lottiefiles.com/)

