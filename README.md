# FastAPI RAG Application
# Multi-layer system for PDF processing, embedding, retrieval, and grounded answer generation

## Project Structure

```
clinictechai/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py          # Pydantic settings (loads from .env)
│   │
│   ├── data_ingest/             # Document ingestion layer
│   │   ├── __init__.py
│   │   └── ingester.py          # Document upload and storage management
│   │
│   ├── pdf_processing/          # PDF type detection and processing
│   │   ├── __init__.py
│   │   ├── processor.py         # PDF type detection (scanned vs text)
│   │   └── ocr_processor.py     # OCR pipeline for scanned PDFs
│   │
│   ├── chunking/                # Text chunking with metadata
│   │   ├── __init__.py
│   │   └── chunker.py           # Semantic chunking with page/bbox metadata
│   │
│   ├── embedding/               # Embedding and vector storage
│   │   ├── __init__.py
│   │   └── embedding_service.py # Embedding generation and vector store
│   │
│   ├── retrieval/               # Retrieval and reranking
│   │   ├── __init__.py
│   │   └── retriever.py         # Vector search and cross-encoder reranking
│   │
│   ├── llm/                     # LLM integration (grounded mode)
│   │   ├── __init__.py
│   │   └── llm_service.py       # Grounded answer generation
│   │
│   ├── verification/            # Post-answer verification
│   │   ├── __init__.py
│   │   └── verifier.py          # Answer grounding and confidence verification
│   │
│   ├── schemas/                 # Pydantic models
│   │   ├── __init__.py
│   │   └── models.py            # Request/response schemas
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py           # Logging, ID generation, file utils
│   │
│   └── api/                     # API endpoints
│       ├── __init__.py
│       └── routes.py            # FastAPI routes
│
├── storage/                     # Document storage
│   ├── raw_documents/           # Original PDF files
│   └── processed_documents/     # Processed content
│
├── logs/                        # Application logs
│
├── main.py                      # FastAPI application entry point
├── .env                         # Environment variables
└── requirements.txt             # Python dependencies
```

## Architecture Layers

### 1. **Data Ingestion Layer** (`data_ingest/`)
- Accepts PDF uploads
- Validates file format and size
- Stores documents in the storage system
- Generates unique document IDs

### 2. **PDF Processing Layer** (`pdf_processing/`)
- **PDF Type Detection**: Classifies PDFs as:
  - `TEXT`: Text-based PDF (direct extraction)
  - `SCANNED`: Image-based PDF (requires OCR)
  - `MIXED`: Contains both
- **Text PDF**: Uses PyMuPDF to extract text and bbox info
- **Scanned PDF**: Uses Tesseract OCR for text extraction

### 3. **Chunking Layer** (`chunking/`)
- Breaks text into semantic chunks
- Preserves metadata:
  - Page number
  - Bounding box coordinates
  - Document reference
  - Extraction type (text/OCR)
- Configurable chunk size and overlap

### 4. **Embedding Layer** (`embedding/`)
- Generates embeddings using SentenceTransformers
- Stores vectors in vector database
- Supports multiple backends:
  - Milvus
  - Pinecone
  - Chroma

### 5. **Retrieval Layer** (`retrieval/`)
- **Vector Search**: Semantic similarity-based retrieval
- **Reranking**: Uses cross-encoder models (mmarco) for relevance scoring
- Returns top-k relevant chunks with similarity scores

### 6. **LLM Layer** (`llm/`)
- Generates answers in STRICT GROUNDED MODE
- Only uses provided context from medical documents
- Refuses to answer off-topic or unrelated questions
- Supports multiple providers (OpenAI, Anthropic)
- Low temperature (0.1) for consistency
- No source citations in answer text (sources provided separately)
- Optimized prompts for medical documents only

### 7. **Safety & Moderation Layer** (`safety/`)
- **Content Moderation**: Uses OpenAI's moderation API
- **Hate Speech Detection**: Filters harmful content in user input
- **Output Validation**: Checks LLM-generated answers for policy violations
- **Domain Enforcement**: Ensures responses stay within medical document context
- **Fail-Safe Mode**: Allows content through on API errors

### 8. **Verification Layer** (`verification/`)
- **Grounding Check**: Verifies answer is supported by context
- **Consistency Check**: Detects contradictions
- **Relevance Check**: Ensures answer addresses query
- **Domain Relevance**: Confirms response stays on topic
- Outputs confidence score and evidence

## Output Format

```json
{
  "success": true,
  "answer": "The clinical findings show...",
  "query": "What are the clinical findings?",
  "page_numbers": [1, 3, 5],
  "evidence": [
    {
      "page_number": 1,
      "document": "patient_report.pdf",
      "exact_chunk": "Clinical findings on imaging reveal...",
      "bbox": [50, 100, 300, 150],
      "highlighted": "Clinical findings on **imaging** reveal..."
    }
  ],
  "sources": [
    {
      "document": "patient_report.pdf",
      "page_number": 1,
      "similarity_score": 0.92,
      "rerank_score": 0.88
    }
  ],
  "verification": {
    "verified": true,
    "confidence_score": 0.85,
    "grounding_score": 0.90,
    "consistency_score": 0.88,
    "relevance_score": 0.78
  }
}
```

## API Endpoints

### 1. Upload Document
```
POST /api/v1/upload
- Accepts: PDF file
- Returns: document_id, chunk_count, pdf_type
```

### 2. Query RAG
```
POST /api/v1/query
- Input: query, document_ids (optional), top_k
- Returns: answer, sources, evidence, verification
```

### 3. Health Check
```
GET /api/v1/health
- Returns: service status
```

## Key Features

✓ Multi-layer architecture for robust processing
✓ Dual PDF processing paths (text and OCR)
✓ Tesseract OCR for scanned PDFs with image preprocessing
✓ Metadata preservation (page, bbox, document)
✓ Semantic chunking with overlap
✓ Cross-encoder reranking
✓ Strict grounding in answer generation
✓ Domain-focused responses (medical documents only)
✓ Content moderation with hate speech filtering
✓ No source citations in answer text (sources provided separately)
✓ Post-answer verification with confidence scores
✓ Comprehensive evidence extraction
✓ Support for multiple LLM providers
✓ Configurable via .env file

## Configuration

All settings are configured via `.env` file:

```bash
# FastAPI
FASTAPI_ENV=development
API_PORT=8000
DEBUG=True

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=your_openai_key
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your_openai_key
EMBEDDING_DIMENSION=1536

# PDF Processing & OCR
OCR_PROVIDER=tesseract
TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.5
RERANK_MODEL=cross-encoder/mmarco-MiniLMv2-L12-H384-v1

# Verification & Safety
VERIFICATION_ENABLED=True
CONFIDENCE_THRESHOLD=0.7

# Vector Database
VECTOR_DB_TYPE=chroma
VECTOR_DB_PATH=./storage/chroma_db
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```
