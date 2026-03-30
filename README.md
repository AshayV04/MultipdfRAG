# PDF Intelligence System with Retrieval Augmented Generation (RAG)

## Overview

The goal of this project is to create a user-centric and intelligent system that enhances information retrieval from PDF documents through natural language queries. The project focuses on streamlining the user experience by developing an intuitive interface, allowing users to interact with PDF content using language they are comfortable with. To achieve this, we leverage the Retrieval Augmented Generation (RAG) methodology introduced by Meta AI researchers.


https://github.com/AshayV04/MultipdfRAG


## Retrieval Augmented Generation (RAG)

### Introduction

RAG is a method designed to address knowledge-intensive tasks, particularly in information retrieval. It combines an information retrieval component with a text generator model to achieve adaptive and efficient knowledge processing. Unlike traditional methods that require retraining the entire model for knowledge updates, RAG allows for fine-tuning and modification of internal knowledge without extensive retraining.

### Workflow

1. **Input**: RAG takes multiple PDFs as input.
2. **Document Processing**: Uses PyMuPDF (`fitz`) to extract text and intelligently guess the Title and Authors of research papers.
3. **Hybrid Search System**: PDFs are chunked and indexed into two distinct systems:
   - **Dense Retrieval**: FAISS using `all-MiniLM-L6-v2` Embeddings from Hugging Face for semantic search.
   - **Sparse Retrieval**: BM25 (Okapi) for keyword-exact matching.
4. **Smart Query Routing**: Queries are classified (summarization, comparison, metadata, semantic) to trigger specific prompts and retrieval strategies.
5. **Text Generation with LLaMA 3**: The retrieved context is fed to the lightning-fast LLaMA-3.3-70B model via the Groq API, maintaining conversation history through Conversation Buffer Memory.
6. **User Interface**: Streamlit provides a seamless chat interface.

### Benefits

- **Adaptability**: RAG adapts to situations where facts may evolve over time, making it suitable for dynamic knowledge domains.
- **Efficiency**: By combining retrieval and generation, RAG provides access to the latest information without the need for extensive model retraining.
- **Reliability**: The methodology ensures reliable outputs by leveraging both retrieval-based and generative approaches, significantly reducing hallucinations.

## Project Features

1. **Advanced Hybrid Retrieval**: Combines the semantic understanding of vector embeddings with the exact-word matching of BM25 for highly accurate context retrieval.
2. **Multi-Document Comparison & Summarization**: Allows users to compare methodologies or summarize findings across multiple distinct PDFs simultaneously without crossing facts.
3. **Smart Paper Parsing**: Automatically extracts document metadata (like authors and titles) to ensure appropriate citation and organization in the chat.
4. **User-friendly Interface**: An intuitive interface designed to accommodate natural language queries, simplifying the interaction with PDF documents.

## Getting Started

To use the PDF Intelligence System:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/AshayV04/MultipdfRAG.git
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application.
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8000` to access the user interface.

## Contributing

We welcome contributions to enhance the PDF Intelligence System. If you're interested in contributing, please follow our [Contribution Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [Apache License](LICENSE).

## Acknowledgments

We would like to express our gratitude to the Hugging Face community for the all-MiniLM-L6-v2 Embeddings model, and OpenAI for providing the GPT-3.5 Turbo model through their API.

---

Feel free to explore and enhance the capabilities of the PDF Intelligence System. Happy querying!
