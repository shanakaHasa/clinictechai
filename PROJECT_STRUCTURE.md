# FastAPI RAG Application - Project Structure Summary

## 📁 Complete Directory Structure

```
clinictechai/
│
├── 📄 main.py                          ⭐ FastAPI application entry point
├── 📄 .env                             ⭐ Environment configuration (UPDATE WITH YOUR KEYS)
├── 📄 .gitignore                       Git ignore file
├── 📄 requirements.txt                 ⭐ Python dependencies
│
├── 📖 README.md                        Project overview & architecture
├── 📖 QUICKSTART.md                    Quick start guide
├── 📖 ARCHITECTURE.md                  Detailed architecture & data flow
├── 📖 EXAMPLES.md                      Usage examples & patterns
│
├── 📁 app/                             Main application package
│   │
│   ├── 📄 __init__.py                  Package initialization
│   │
│   ├── 📁 config/                      Configuration management
│   │   ├── 📄 __init__.py
│   │   └── 📄 settings.py              ⭐ Pydantic settings (loads from .env)
│   │
│   ├── 📁 data_ingest/                 Layer 1: Document Ingestion
│   │   ├── 📄 __init__.py
│   │   └── 📄 ingester.py              Upload, validate, store documents
│   │
│   ├── 📁 pdf_processing/              Layer 2: PDF Type Detection
│   │   ├── 📄 __init__.py
│   │   ├── 📄 processor.py             Detect TEXT vs SCANNED PDFs
│   │   └── 📄 ocr_processor.py         OCR pipeline for scanned PDFs
│   │
│   ├── 📁 chunking/                    Layer 3: Text Chunking
│   │   ├── 📄 __init__.py
│   │   └── 📄 chunker.py               Split text, preserve metadata
│   │
│   ├── 📁 embedding/                   Layer 4: Embedding & Vector Store
│   │   ├── 📄 __init__.py
│   │   └── 📄 embedding_service.py     Embeddings + vector DB operations
│   │
│   ├── 📁 retrieval/                   Layer 5: Retrieval & Reranking
│   │   ├── 📄 __init__.py
│   │   └── 📄 retriever.py             Vector search + cross-encoder reranking
│   │
│   ├── 📁 llm/                         Layer 6: LLM Integration
│   │   ├── 📄 __init__.py
│   │   ├── 📄 llm_service.py           Grounded answer generation (strict mode)
│   │   └── 📄 prompts.py               Optimized prompts for medical documents
│   │
│   ├── 📁 safety/                      Layer 7: Content Moderation & Safety
│   │   ├── 📄 __init__.py
│   │   └── 📄 content_moderator.py     OpenAI moderation API for hate speech filtering
│   │
│   ├── 📁 verification/                Layer 8: Post-Answer Verification
│   │   ├── 📄 __init__.py
│   │   └── 📄 verifier.py              Verify grounding & confidence scoring
│   │
│   ├── 📁 schemas/                     Data Validation
│   │   ├── 📄 __init__.py
│   │   └── 📄 models.py                ⭐ Pydantic request/response models
│   │
│   ├── 📁 utils/                       Utilities
│   │   ├── 📄 __init__.py
│   │   └── 📄 helpers.py               Logging, ID generation, file utilities
│   │
│   └── 📁 api/                         API Routes
│       ├── 📄 __init__.py
│       └── 📄 routes.py                ⭐ FastAPI endpoints (upload, query, health)
│
├── 📁 storage/                         Document storage (auto-created)
│   ├── 📁 raw_documents/               Original PDF files
│   └── 📁 processed_documents/         Processed content
│
└── 📁 logs/                            Application logs (auto-created)
```

## 🔄 Data Flow Pipeline

### Document Upload Flow
```
PDF File Upload
    ↓
Document Ingestion (data_ingest/ingester.py)
    ↓
PDF Type Detection (pdf_processing/processor.py)
    ├─→ TEXT → Direct extraction (PyMuPDF)
    └─→ SCANNED → OCR Pipeline (pdf_processing/ocr_processor.py)
    ↓
Text Chunking (chunking/chunker.py)
    [Preserve: page_number, bbox, document_id]
    ↓
Embedding Generation (embedding/embedding_service.py)
    [1536-dimensional vectors]
    ↓
Vector Store (embedding/embedding_service.py)
    [Milvus/Pinecone/Chroma]
```

### Query Flow
```
User Query
    ↓
Retrieval (retrieval/retriever.py)
    [Vector similarity search]
    ↓
Reranking (retrieval/retriever.py)
    [Cross-encoder relevance scoring]
    ↓
LLM Answer Generation (llm/llm_service.py)
    [STRICT GROUNDED MODE - only use context]
    ↓
Post-Answer Verification (verification/verifier.py)
    [Grounding, consistency, relevance checks]
    ↓
Response with Evidence & Confidence Scores
    [Answer + Page Numbers + Evidence Chunks + Verification]
```

## 🎯 8-Layer Architecture

| Layer | Module | Purpose | Output |
|-------|--------|---------|--------|
| 1 | data_ingest | Upload & store documents | document_id, storage_path |
| 2 | pdf_processing | Detect PDF type | TEXT/SCANNED classification |
| 3 | (ocr_processor) | Extract from scanned PDFs | OCR text + confidence |
| 4 | chunking | Split & preserve metadata | Chunks with page/bbox/doc info |
| 5 | embedding | Generate vectors | 1536-dim embeddings + storage |
| 6 | retrieval | Vector search + rerank | Top-k relevant chunks |
| 7 | llm | Generate grounded answer | Factual answer text (no sources) |
| 8 | safety | Content moderation | Policy violation detection |
| 9 | verification | Verify quality | Confidence scores + evidence |
| - | api | HTTP endpoints | REST interface |

## 📋 Key Files to Update/Configure

### 1. **FIRST: Update `.env`** (REQUIRED)
```bash
# Add your API keys
LLM_API_KEY=your_openai_or_anthropic_key
EMBEDDING_API_KEY=your_api_key_if_needed

# If using Tesseract OCR
TESSERACT_PATH=/path/to/tesseract

# Database URLs
DATABASE_URL=postgresql://...
VECTOR_DB_URL=http://localhost:19530
```

### 2. **Configure `app/config/settings.py`**
Already set up - just update `.env` file

### 3. **Main Entry Point: `main.py`**
Ready to run - no changes needed

### 4. **API Routes: `app/api/routes.py`**
Contains 3 endpoints:
- `POST /api/v1/upload` - Upload PDF
- `POST /api/v1/query` - Query RAG
- `GET /api/v1/health` - Health check

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Update .env with your API keys
# Edit: .env file

# 3. Start application
python main.py

# 4. Open browser
# http://localhost:8000/docs
# (Interactive API documentation)
```

## 📊 Response Structure

Every RAG query returns:
```json
{
  "answer": "Grounded response based on documents",
  "page_numbers": [1, 3, 5],
  "evidence": [
    {
      "page_number": 1,
      "exact_chunk": "Quote from document",
      "bbox": [x0, y0, x1, y1],
      "highlighted": "Key terms highlighted"
    }
  ],
  "verification": {
    "confidence_score": 0.85,
    "grounding_score": 0.90,
    "consistency_score": 0.88,
    "relevance_score": 0.78
  },
  "sources": [
    {
      "document": "filename.pdf",
      "page_number": 1,
      "similarity_score": 0.92
    }
  ]
}
```

## ✨ Features Implemented

✅ **Multi-Layer Architecture** - 7 distinct processing layers  
✅ **PDF Type Detection** - Automatic TEXT/SCANNED classification  
✅ **OCR Support** - Tesseract integration for scanned PDFs  
✅ **Metadata Preservation** - Page numbers, bounding boxes, document refs  
✅ **Semantic Chunking** - Overlapping chunks with context preservation  
✅ **Vector Search** - Multiple backend support (Milvus, Pinecone, Chroma)  
✅ **Reranking** - Cross-encoder relevance scoring  
✅ **Grounded LLM** - Strict mode, no hallucination  
✅ **Answer Verification** - 3-layer verification with confidence  
✅ **Evidence Extraction** - Exact chunks with highlighting  

## 🔧 Configuration Parameters

Key settings in `.env`:

```bash
# Chunking
CHUNK_SIZE=500              # Characters per chunk
CHUNK_OVERLAP=100           # Overlap between chunks

# Retrieval
TOP_K_RESULTS=5             # Results to retrieve
SIMILARITY_THRESHOLD=0.5    # Minimum similarity

# LLM
LLM_TEMPERATURE=0.1         # Low = grounded, high = creative
LLM_MAX_TOKENS=2000         # Max response length

# Verification
VERIFICATION_ENABLED=True
CONFIDENCE_THRESHOLD=0.7    # Minimum confidence

# Embedding
EMBEDDING_DIMENSION=1536    # Vector dimensions
EMBEDDING_MODEL=text-embedding-3-small
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Project overview & architecture |
| QUICKSTART.md | Quick start guide (40+ lines) |
| ARCHITECTURE.md | Detailed data flow & design |
| EXAMPLES.md | 9 usage examples with code |
| QUICKSTART.py | (This file) - Structure reference |

## 🎓 Learning Path

1. **Start**: Read README.md
2. **Setup**: Follow QUICKSTART.md
3. **Understand**: Read ARCHITECTURE.md  
4. **Practice**: Try examples in EXAMPLES.md
5. **Deploy**: Update .env and run `python main.py`

## 🔍 Module Relationships

```
main.py (FastAPI app)
    ↓
app/api/routes.py (Endpoints)
    ├→ data_ingest/ingester.py (Upload)
    ├→ pdf_processing/processor.py (Detect type)
    ├→ pdf_processing/ocr_processor.py (OCR if needed)
    ├→ chunking/chunker.py (Split text)
    ├→ embedding/embedding_service.py (Vectorize)
    ├→ retrieval/retriever.py (Search + rerank)
    ├→ llm/llm_service.py (Generate answer)
    ├→ verification/verifier.py (Verify quality)
    └→ schemas/models.py (Validate data)

All configured via:
    app/config/settings.py ← .env file
```

## 🎯 Ready for:

- ✅ Medical document processing
- ✅ PDF knowledge base
- ✅ Clinical decision support
- ✅ Document Q&A systems
- ✅ Evidence-based information retrieval

## 📝 Next Steps

1. Update `.env` with your API keys
2. Run `pip install -r requirements.txt`
3. Start with `python main.py`
4. Visit `http://localhost:8000/docs` for interactive API testing
5. Upload a PDF and ask questions!
