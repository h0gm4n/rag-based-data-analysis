# RAG-Based Data Analysis

A Retrieval-Augmented Generation (RAG) system for analyzing superstore sales data using LLMs. Ask natural language questions about your data and get intelligent insights powered by semantic search and language models.

## Features

- **RAG Pipeline**: Combines vector search with LLM generation for context-aware answers
- **Data Aggregation**: Monthly trends, category analysis, product performance, discount impact
- **Semantic Search**: Uses HuggingFace embeddings for intelligent document retrieval
- **LLM Integration**: Ollama (phi3) for local, privacy-preserving inference
- **Vector Store**: Chroma for efficient similarity search

## Requirements

### System Requirements
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running with phi3 model


### Installation

1. **Install Ollama** (if not already installed):
   - Download from [ollama.ai](https://ollama.ai)
   - After installation, download the phi3 model:
   ```bash
   ollama pull phi3
   ```

2. **Set up Python environment**:
   ```bash
   uv venv
   uv sync
   ```

3. **Download data** :
- Download data from Kaggle
- [Superstore data](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final/data)
    ```bash
   uv run src/init.py
   ```

4. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ```

## Usage

Run the interactive Q&A system:
```bash
uv run python src/main.py
```

Then ask questions about your data:
```
Ask a question (or 'exit'): What is the top selling product?
Ask a question (or 'exit'): How much profit did we make in 2017?
Ask a question (or 'exit'): Which region has the highest sales?
```

## Project Structure

```
src/
├── main.py              # Entry point, main loop
├── load_data.py         # CSV data loading with encoding handling
├── text_transform.py    # Convert rows to natural language text
├── aggregation.py       # Data aggregation (monthly, category, product, discount)
├── chunking.py          # Text splitting for RAG
├── embeddings.py        # HuggingFace embeddings
├── vector_store.py      # Chroma vector database
└── rag_pipeline.py      # RAG query processing

data/
└── superstore.csv       # Superstore sales dataset
```

## How It Works

1. **Data Loading**: CSV data is loaded and converted to natural language text
2. **Aggregation**: Monthly trends, categories, products, and discounts are summarized
3. **Chunking**: Text is split into semantic chunks for efficient storage
4. **Embedding**: Chunks are embedded using HuggingFace's all-MiniLM-L6-v2 model
5. **Vector Store**: Embeddings are stored in Chroma for fast retrieval
6. **Query Processing**: User questions are embedded and similar documents are retrieved
7. **LLM Response**: Retrieved context is sent to Ollama (phi3) for generation

## Key Technologies

- **LangChain**: RAG framework and LLM orchestration
- **Ollama**: Local LLM serving (phi3)
- **Chroma**: Vector database
- **HuggingFace**: Sentence embeddings
- **Polars**: Fast data processing
- **UV**: Python package management

## Configuration

- **Model**: phi3 (via Ollama)
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **Chunk Size**: 1000 characters with 100-char overlap
- **Retrieval**: Top 20 most similar documents per query



