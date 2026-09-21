# Research Intelligence RAG

A production-minded, multi-document research assistant for papers, technical PDFs, reports, documentation and whitepapers.

## Pipeline

```text
PDF
 -> PyMuPDF extraction (+ Tesseract OCR for scanned pages)
 -> document metadata + SHA-256 deduplication
 -> section-aware parent chunks
 -> page-aware child chunks
 -> embeddings
 -> PostgreSQL + pgvector HNSW
 -> query rewriting / multi-query
 -> optional HyDE (retrieval only)
 -> dense retrieval + BM25
 -> Reciprocal Rank Fusion
 -> cross-encoder reranking
 -> parent-context expansion
 -> exact-span contextual compression
 -> grounded answer + [S1] citations
 -> deterministic evidence-span validation
 -> LLM claim-level citation audit
 -> one bounded citation-repair pass
```

## What was fixed

### 1. Exact child page ranges
Child chunks now retain `page_start` and `page_end`. A chunk spanning pages 4–5 is no longer incorrectly reported as page 4. Parent expansion also preserves its true page range.

### 2. Deterministic citation/evidence verification
The compression model is not trusted blindly. Its extracted evidence must be found verbatim in the retrieved source text (with whitespace-normalized matching as a fallback). Citation IDs are validated, and the semantic claim audit is run only after span validation.

### 3. OCR works in Docker
The API image installs the actual `tesseract-ocr` executable in addition to the Python `pytesseract` wrapper.

### 4. Configurable expensive stages
Use environment flags to run controlled experiments:

```env
ENABLE_MULTI_QUERY=true
ENABLE_COMPRESSION=true
ENABLE_CITATION_AUDIT=true
ENABLE_OCR=false
RETRIEVAL_CANDIDATE_K=24
```

### 5. Metadata filtering
The API supports `year_from` and `year_to` filters and document-ID filtering. Metadata includes title, authors, publication year only when explicitly supported by first-page publication cues or an arXiv identifier, abstract and DOI when discoverable.

### 6. Better evaluation
The benchmark runner now reports:

- Precision@K
- Recall@K
- MRR
- Hit Rate@K
- nDCG@K

The benchmark runner can compare vector, BM25, hybrid/RRF and HyDE configurations on a fixed labelled corpus. It reports retrieval metrics only; answer-quality evaluation requires reference answers and a separate judge/evaluation protocol.

## Important engineering boundaries

- BM25 is an in-process cache. It is suitable for a portfolio/small deployment; for very large corpora or multiple API workers, replace it with PostgreSQL full-text search or a dedicated search service.
- PDF extraction is text-first. Tables, figures and equations are not yet reconstructed as multimodal structured objects.
- Authentication, RBAC, multi-tenancy, object storage, distributed rate limiting and OpenTelemetry are not falsely advertised as implemented.
- Citation verification is deterministic for evidence spans and LLM-assisted for semantic claim support. The LLM audit is never the only check.

## Setup

### Local

```bash
cp .env.example .env
# Put your OPENAI_API_KEY in .env
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL/pgvector
Docker compose up -d db

# Initialize/migrate the database
python scripts/init_db.py

# API
uvicorn app.main:app --reload

# Frontend, in another terminal
streamlit run frontend/streamlit_app.py
```

### Docker

```bash
docker compose up --build
```

With `ENABLE_OCR=true`, the API container already contains Tesseract.

## Migrations

Alembic contains an incremental migration for the new child page-range columns:

```bash
alembic upgrade head
```

`scripts/init_db.py` runs `alembic upgrade head`, so local and deployment environments use the same migration path. Do not use `Base.metadata.create_all()` for application schema management.

## API

### Upload

`POST /documents/upload`

Accepts PDF only, validates size/page count and PDF structure, computes SHA-256, extracts metadata, chunks, embeds and stores the document with a database transaction plus crash-recoverable staging.

### Query

`POST /query`

Example:

```json
{
  "question": "Compare the limitations of the proposed approaches.",
  "document_ids": [],
  "top_k": 8,
  "use_hyde": false,
  "year_from": 2020,
  "year_to": 2026
}
```

### Response highlights

```text
answer
sources[]
  citation_id
  document_id
  page_start
  page_end
  section
  evidence
  evidence_verified
citation_verification
queries
 timings_ms
```

## Evaluation workflow

1. Index a fixed benchmark corpus.
2. Label relevant child chunk IDs.
3. Run `python scripts/evaluate.py`.
4. Compare configurations.
5. Record retrieval and answer-quality metrics.

Do not use placeholder labels to claim performance. The sample benchmark intentionally skips unlabelled examples.

## Suggested learning order

1. Understand PDF extraction and page-aware chunking.
2. Understand embeddings and pgvector.
3. Compare vector retrieval vs BM25.
4. Understand RRF mathematically and experimentally.
5. Study cross-encoder reranking.
6. Study parent-child retrieval.
7. Add query rewriting and HyDE one at a time.
8. Understand contextual compression and why exact-span validation is necessary.
9. Build a labelled evaluation set.
10. Run ablations and measure retrieval quality.
11. Add tracing/observability when moving toward deployment.

## Operational notes

- `/health` means the API process is alive; `/ready` verifies PostgreSQL, pgvector, required tables, and that the recorded Alembic revisions exactly match the migration graph heads (without hard-coding a revision).
- Embeddings are sent in bounded batches and their dimension is validated against the current fixed 1536-dimensional database schema; unsupported `EMBEDDING_DIMENSION` overrides fail fast.
- Uploads are limited by both byte size and page count.
- BM25 is rebuilt only after document ingestion/deletion in a given API process; it remains process-local and is therefore not a distributed search index.
- Child chunks are ranked, then expanded to their parent context. The original child text is retained as `retrieval_text` for traceability.
- Exact evidence validation is deterministic; semantic citation auditing is LLM-assisted and is not treated as a mathematical guarantee.


## Reliability and readiness behavior

- `/ready` verifies PostgreSQL, pgvector, the required application tables, and a populated `alembic_version` row before reporting `ready`.
- PDF validation and extraction happen in one extraction pass; the page-count limit is checked before page text/OCR work.
- Uploads are first written to a unique hidden `.uploading` staging file. After the indexing transaction commits, the file is atomically renamed to final storage. Startup reconciliation repairs a crash between the DB commit and rename, removes orphan staging files, and only removes orphan final PDFs matching the application-owned 16-hex-hash filename convention.
- Multi-query and HyDE are optional retrieval enhancements. If either generation step fails, retrieval falls back to the original query instead of failing the entire request.
- Cross-encoder reranking falls back to RRF ordering when the reranker is unavailable; source records retain `rrf_rank`, `rrf_score`, `rerank_score` (when available), and `ranking_source`.
- Compression falls back to full retrieved text and reports a structured `pipeline_status.compression` object with `configured`, `used`, and `status`; each source also reports `source_representation`.
- Citation verification reports explicit `passed`, `failed`, `unavailable`, or `disabled` status.
- Embeddings are batched and retried using one reusable API client per operation.


## Testing

Install the project dependencies in a clean virtual environment, then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

GitHub Actions runs the same dependency installation and test command on pushes and pull requests.
