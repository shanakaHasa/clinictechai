"""
FastAPI RAG Application - Architecture Overview

COMPLETE DATA FLOW PIPELINE
==============================================================================

INPUT: PDF Document
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: DATA INGESTION                                                 │
│ • File upload validation                                                │
│ • PDF storage (storage/raw_documents)                                   │
│ • Document ID generation                                                │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: PDF TYPE DETECTION                                             │
│ • Check first 5 pages for text content                                  │
│ • Classification:                                                       │
│   ├─ TEXT (>10% text) → Direct extraction with PyMuPDF                 │
│   ├─ SCANNED (<10% text) → Route to OCR                               │
│   └─ MIXED → Hybrid approach                                           │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼ (TEXT)                ▼ (SCANNED)
┌──────────────────────┐  ┌──────────────────────────┐
│ PyMuPDF Extraction   │  │ OCR PIPELINE             │
│ • get_text()         │  │ • Convert to images      │
│ • Extract blocks     │  │ • Tesseract OCR         │
│ • Bounding boxes     │  │ • Extract confidence    │
│ • Page numbers       │  │ • Generate bboxes       │
└──────────────┬───────┘  └────────────┬─────────────┘
               │                        │
               └───────────┬────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: TEXT CHUNKING                                                  │
│ • Split text into chunks (default: 500 chars)                          │
│ • Maintain overlap (default: 100 chars)                                │
│ • Preserve metadata:                                                    │
│   ├─ page_number: Which page                                           │
│   ├─ chunk_index: Position on page                                     │
│   ├─ bbox: Bounding box [x0, y0, x1, y1]                             │
│   ├─ document_id: Source document                                      │
│   └─ extraction_type: "text" or "ocr"                                 │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: EMBEDDING GENERATION                                           │
│ • Model: text-embedding-3-small (1536 dims)                           │
│ • Batch encoding                                                        │
│ • Output: Vector per chunk                                             │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: VECTOR STORAGE                                                 │
│ Supported backends:                                                     │
│ • Milvus (recommended for production)                                  │
│ • Pinecone (cloud-hosted)                                              │
│ • Chroma (lightweight)                                                 │
│ Storage: chunk_id → embedding + metadata                               │
└─────────────────────────────────────────────────────────────────────────┘

============================= QUERY TIME ====================================

INPUT: User Query
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: INPUT MODERATION                                               │
│ • Check user input for policy violations                                │
│ • OpenAI Moderation API: Hate speech, violence, harassment              │
│ • Return safety violation if detected                                   │
│ • Log violations for monitoring                                         │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 7: RETRIEVAL                                                      │
│ 1. Embed query using same model as training                            │
│ 2. Vector similarity search (top_k=5)                                  │
│ 3. Filter by similarity_threshold (0.5)                                │
│ Output: [chunk_id, similarity_score, metadata]                         │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 8: RERANKING (Cross-Encoder)                                      │
│ • Model: mmarco-MiniLMv2-L12-H384-v1                                   │
│ • Input: (query, chunk_text) pairs                                     │
│ • Output: Relevance scores                                             │
│ • Re-sort by rerank score                                              │
│ • Keep top_k results                                                   │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 9: LLM - GROUNDED ANSWER GENERATION                              │
│ CONSTRAINTS:                                                             │
│ • Mode: STRICT GROUNDED (temperature=0.1)                             │
│ • Only use provided context (documents only)                          │
│ • Refuse off-topic questions                                          │
│ • Do NOT include source citations in answer                           │
│ • Return "not available" if answer not in context                     │
│ • No hallucination or assumptions                                     │
│ Providers: OpenAI, Anthropic                                           │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 10: OUTPUT MODERATION                                             │
│ • Check LLM-generated answer for policy violations                      │
│ • Verify domain relevance (medical documents only)                     │
│ • Detect hate speech/violence in output                                │
│ • Replace with safety message if violation detected                    │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 11: ANSWER VERIFICATION                                           │
│ Grounding Check:                                                        │
│   • Match answer sentences to context                                  │
│   • Percentage of sentences supported                                  │
│ Consistency Check:                                                      │
│   • Look for contradictions                                            │
│   • Verify statement alignment                                         │
│ Relevance Check:                                                        │
│   • Answer addresses the query                                         │
│   • Semantic similarity (query ↔ answer)                              │
│ Domain Check:                                                           │
│   • Response stays within medical documents                            │
│ Output:                                                                 │
│   • confidence_score (0-1)                                             │
│   • meets_threshold (bool)                                             │
│   • component scores (grounding, consistency, relevance, domain)      │
│   • evidence extraction                                                │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
OUTPUT: Comprehensive RAG Response
{
  "answer": "The clinical findings show...",
  "page_numbers": [1, 3, 5],
  "evidence": [
    {
      "page_number": 1,
      "exact_chunk": "...",
      "bbox": [50, 100, 300, 150],
      "highlighted": "Key terms highlighted in context"
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
      "page": 1,
      "similarity": 0.92,
      "rerank": 0.88
    }
  ]
}

==============================================================================
KEY FEATURES
==============================================================================

✓ PDF Type Detection
  - Automatic detection of scanned vs text PDFs
  - Dual processing pipelines (OCR + direct extraction)

✓ Metadata Preservation
  - Page numbers attached to every chunk
  - Bounding box coordinates for highlighting
  - Document and extraction type tracking

✓ Semantic Search + Reranking
  - Vector similarity for broad relevance
  - Cross-encoder reranking for precision

✓ Grounded Answer Generation
  - Strict mode: Only uses provided context
  - No hallucination, no assumptions
  - Source attribution built-in

✓ Multi-layer Verification
  - Grounding verification
  - Consistency checking
  - Confidence scoring

✓ Evidence Extraction
  - Exact chunks from source
  - Page/bbox coordinates
  - Text highlighting

==============================================================================
CONFIGURATION PARAMETERS (from .env)
==============================================================================

PDF_PROCESSING:
  - OCR_PROVIDER: tesseract
  - TESSERACT_PATH: Path to OCR engine (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe)

CHUNKING:
  - CHUNK_SIZE: 500 (characters per chunk)
  - CHUNK_OVERLAP: 100 (overlap between chunks)

RETRIEVAL:
  - TOP_K_RESULTS: 5 (chunks to retrieve)
  - RERANK_MODEL: cross-encoder model name
  - SIMILARITY_THRESHOLD: 0.5 (minimum score)

LLM:
  - LLM_PROVIDER: openai|anthropic
  - LLM_MODEL: gpt-4-turbo|claude-3-opus
  - LLM_TEMPERATURE: 0.1 (grounded mode)
  - LLM_MAX_TOKENS: 2000

MODERATION & SAFETY:
  - Moderation enabled automatically when API key available
  - Uses OpenAI Moderation API for input/output validation
  - Checks for: hate speech, violence, harassment
  - Fail-safe mode: allows content if API error occurs
  - Domain enforcement: medical documents only

VERIFICATION:
  - VERIFICATION_ENABLED: true
  - CONFIDENCE_THRESHOLD: 0.7 (minimum confidence)

EMBEDDING:
  - EMBEDDING_MODEL: text-embedding-3-small
  - EMBEDDING_DIMENSION: 1536
  - VECTOR_DB_TYPE: milvus|pinecone|chroma

==============================================================================
API ENDPOINTS
==============================================================================

POST /api/v1/upload
  Upload PDF → Document indexed → Chunks embedded
  Response: document_id, total_chunks, pdf_type

POST /api/v1/query
  Query → Retrieve → Rerank → LLM → Verify → Response
  Response: answer, page_numbers, evidence, verification

GET /api/v1/health
  Check service status

GET /docs
  Interactive API documentation (Swagger UI)

==============================================================================
"""
