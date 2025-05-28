# RAG LLM Application

A Retrieval-Augmented Generation (RAG) application using LLM models and vector stores.

## Python Version

3.11

## Key Components

- **Embeddings**: Uses `LlamaCppEmbeddings` - requires **GGUF format models** (quantized llama.cpp models)
- **LLM**: Uses `LlamaCpp` - requires **GGUF format models**
- **Vector Store**: FAISS (model-agnostic)
- **RAG Pipeline**: LangChain's RetrievalQA (works with any LangChain-compatible LLM)

## Apple Silicon Support

Not supported.

## Supported File Types

The application supports the following file types:

- PDF (*.pdf)
- CSV (*.csv)
- JSON (*.json)
- HTML (*.html)
- Text (*.txt)

## Configuration

The application can be configured using environment variables:

### Basic Configuration
- `SOURCE_DIR`: Directory containing the source files to process
- `SOURCE_TYPE`: Type of source files to process (pdf, csv, json, html, txt)
- `EMBEDDINGS_MODEL_PATH`: Path to the embeddings model
- `LLM_MODEL_PATH`: Path to the LLM model
- `FLASK_PORT`: Port for the Flask API server

### LangSmith Monitoring (Optional)
- `LANGCHAIN_TRACING_V2`: Set to "true" to enable LangSmith tracing
- `LANGCHAIN_ENDPOINT`: LangSmith API endpoint (default: https://api.smith.langchain.com)
- `LANGCHAIN_API_KEY`: Your LangSmith API key
- `LANGCHAIN_PROJECT`: LangSmith project name for organizing traces

### LLM Model Parameters
- `CPU_THREADS`: Number of CPU threads to use (default: 6, recommended: match your CPU cores)
- `GPU_LAYERS`: Number of layers to offload to GPU (-1 means all)
- `GPU_BATCH_SIZE`: Batch size for processing multiple inputs at once (default: 256)
- `N_CTX`: Context window size/token limit (default: 4096, should match your LLM model's context size)
- `TEMPERATURE`: Controls randomness in generation (0.0 = deterministic, higher = more random)
- `MAX_TOKENS`: Maximum number of tokens to generate in responses (default: 2000)
- `TOP_P`: Nucleus sampling probability threshold (default: 0.9)
- `TOP_K`: Limits vocabulary to top K tokens (default: 40)
- `REPEAT_PENALTY`: Penalizes token repetition (default: 1.3, higher = less repetition)
- `GRAMMAR_PATH`: Optional path to JSON grammar for structured output

## Usage

### Docker Compose

The easiest way to run the application is using Docker Compose:

1. Make sure you have a `.env` file in the project root directory with all required environment variables (see Configuration section above).
```bash
cp .env.example .env
```

2. Run the application:
```bash
docker-compose up
```

The Docker container will use the environment variables from your `.env` file. Make sure this file exists before running the container.

### Running Directly

You can also run the application directly:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file based on `.env.example` and configure your environment variables.

3. Run the application:
```bash
# Initialize and run the application
python -m rag_app.run
```

To force recreation of the vector store:
```bash
python -m rag_app.run --force-create
```

## API

The application provides a simple API to query the vector store:

```
POST /query
```

Request body:
```json
{
  "question": "your question here"
}
```

Example:
```bash
curl -X POST "http://localhost:8080/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Response:
```json
{
  "data": {
    "answer": "string",
    "question": "string"
  }
}
```

## Testing

Run the tests with:

```bash
python -m unittest discover tests
```