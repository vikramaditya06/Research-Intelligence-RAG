from pathlib import Path
import hashlib
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ingestion.pdf import extract_pdf
from app.ingestion.chunker import chunk_pages
from app.core.config import settings
from app.core.logging import logger
from app.api.schemas import QueryRequest


router = APIRouter()
DATA_DIR = Path(settings.data_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        vector_extension = db.execute(
            text(
                "SELECT extname FROM pg_extension "
                "WHERE extname = 'vector'"
            )
        ).scalar_one_or_none()

        if vector_extension != "vector":
            raise RuntimeError("pgvector extension is unavailable")

        required_columns = {
            "documents": {
                "id",
                "filename",
                "path",
                "file_hash",
                "created_at",
            },
            "parent_chunks": {
                "id",
                "document_id",
                "page_start",
                "page_end",
                "text",
            },
            "chunks": {
                "id",
                "document_id",
                "parent_id",
                "page",
                "page_start",
                "page_end",
                "text",
                "embedding",
            },
        }

        for table, columns in required_columns.items():
            actual = set(
                db.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = :table
                        """
                    ),
                    {"table": table},
                ).scalars().all()
            )

            missing = columns - actual

            if missing:
                raise RuntimeError(
                    f"Required schema columns missing from "
                    f"{table}: {sorted(missing)!r}"
                )

        vector_type = db.execute(
            text(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = 'chunks'
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            )
        ).scalar_one_or_none()

        if vector_type != f"vector({settings.embedding_dimension})":
            raise RuntimeError(
                f"chunks.embedding has type {vector_type!r}; "
                f"expected vector({settings.embedding_dimension})"
            )

        db_versions = set(
            db.scalars(
                text("SELECT version_num FROM alembic_version")
            ).all()
        )

        if not db_versions:
            raise RuntimeError(
                "Database has no recorded Alembic migration revision"
            )

        alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
        alembic_config = Config(str(alembic_ini))
        alembic_config.set_main_option(
            "script_location",
            str(alembic_ini.parent / "alembic"),
        )

        heads = set(
            ScriptDirectory.from_config(
                alembic_config
            ).get_heads()
        )

        if db_versions != heads:
            raise RuntimeError(
                f"Database schema revisions are {sorted(db_versions)!r}; "
                f"expected Alembic heads {sorted(heads)!r}"
            )

        return {
            "status": "ready",
            "migration": sorted(db_versions),
        }

    except Exception as exc:
        logger.warning(
            "Readiness check failed: %s",
            type(exc).__name__,
        )
        raise HTTPException(503, "Application database/schema is not ready") from exc


@router.post("/documents/upload")
def upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    from app.db.models import Document, Chunk, ParentChunk
    from app.services.llm import embed_batches
    from app.services.pipeline import invalidate_bm25

    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            400,
            "Only PDF files are supported",
        )

    content = file.file.read(settings.max_upload_bytes + 1)

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            413,
            f"PDF exceeds maximum size of "
            f"{settings.max_upload_bytes // (1024 * 1024)} MB",
        )

    file_hash = hashlib.sha256(content).hexdigest()

    existing = db.scalar(
        select(Document).where(
            Document.file_hash == file_hash
        )
    )

    if existing:
        return {
            "document_id": existing.id,
            "filename": existing.filename,
            "duplicate": True,
            "pages": 0,
            "parents": 0,
            "chunks": 0,
        }

    final_path = DATA_DIR / f"{file_hash[:16]}_{filename}"

    staging_path = (
        DATA_DIR
        / f".{file_hash}_{uuid.uuid4().hex}_{filename}.uploading"
    )

    path = staging_path

    try:
        staging_path.write_bytes(content)

        pages, pdf_meta = extract_pdf(
            str(path),
            enable_ocr=settings.enable_ocr,
            ocr_min_chars=settings.ocr_min_chars,
            max_pages=settings.max_pages,
        )

        if not pages:
            raise HTTPException(
                422,
                "No extractable text was found in the PDF. "
                "Enable OCR for scanned PDFs.",
            )

        parents, chunks = chunk_pages(
            pages,
            settings.parent_size,
            settings.chunk_size,
            settings.chunk_overlap,
        )

        if not chunks:
            raise HTTPException(
                422,
                "PDF produced no usable text chunks",
            )

        doc = Document(
            filename=filename,
            title=pdf_meta.get("title") or Path(filename).stem,
            path=str(path),
            file_hash=file_hash,
            authors=pdf_meta.get("authors"),
            year=pdf_meta.get("year"),
            abstract=pdf_meta.get("abstract"),
            doi=pdf_meta.get("doi"),
        )

        db.add(doc)

        try:
            db.flush()

        except IntegrityError:
            # Another request may have indexed the same SHA-256 concurrently.
            # Roll back this transaction, then return the committed winner.
            db.rollback()

            existing = db.scalar(select(Document).where(Document.file_hash == file_hash))

            if existing is not None:
                # If this request used a different filename, its temporary file
                # is not the committed winner and must not become an orphan.
                if path != Path(existing.path):
                    path.unlink(missing_ok=True)

                return {
                    "document_id": existing.id,
                    "filename": existing.filename,
                    "duplicate": True,
                    "pages": 0,
                    "parents": 0,
                    "chunks": 0,
                }

            raise

        for p in parents:
            db.add(
                ParentChunk(
                    id=p.id,
                    document_id=doc.id,
                    page_start=p.page_start,
                    page_end=p.page_end,
                    section=p.section,
                    text=p.text,
                    metadata_json=p.metadata,
                )
            )

        chunk_offset = 0

        for vector_batch in embed_batches((c.text for c in chunks)):
            batch_chunks = chunks[
                chunk_offset:chunk_offset + len(vector_batch)
            ]

            if len(batch_chunks) != len(vector_batch):
                raise RuntimeError(
                    "Embedding count does not match chunk batch"
                )

            for c, vec in zip(batch_chunks, vector_batch):
                db.add(
                    Chunk(
                        document_id=doc.id,
                        parent_id=c.parent_id,
                        page=c.page,
                        page_start=c.page_start,
                        page_end=c.page_end,
                        section=c.section,
                        text=c.text,
                        metadata_json=c.metadata,
                        embedding=vec,
                    )
                )

            chunk_offset += len(vector_batch)

        if chunk_offset != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count"
            )

        db.commit()

        # The DB commit is the source of truth. Move the successfully indexed
        # staging file into its final location atomically. If the process dies
        # between these steps, the DB still points at the staging file and a
        # reconciliation job can safely finish the rename later.
        try:
            staging_path.replace(final_path)

            db.execute(
                update(Document)
                .where(Document.id == doc.id)
                .values(path=str(final_path))
            )

            db.commit()
            doc.path = str(final_path)

        except Exception:
            db.rollback()

            # Compensate when possible: if the rename succeeded but the DB
            # update did not, move the file back so the committed DB path
            # remains valid and reconciliation can retry safely later.
            if final_path.exists() and not staging_path.exists():
                try:
                    final_path.replace(staging_path)
                except OSError:
                    logger.exception(
                        "Could not restore staging file for document %s",
                        doc.id,
                    )

            logger.exception(
                "Document %s indexed but final storage rename is pending",
                doc.id,
            )

        invalidate_bm25()

        return {
            "document_id": doc.id,
            "filename": filename,
            "duplicate": False,
            "pages": len(pages),
            "parents": len(parents),
            "chunks": len(chunks),
        }

    except HTTPException:
        db.rollback()
        staging_path.unlink(missing_ok=True)

        if final_path != staging_path:
            final_path.unlink(missing_ok=True)

        raise

    except ValueError as exc:
        db.rollback()
        staging_path.unlink(missing_ok=True)

        if final_path != staging_path:
            final_path.unlink(missing_ok=True)

        raise HTTPException(422, str(exc)) from exc

    except Exception as exc:
        db.rollback()
        staging_path.unlink(missing_ok=True)

        logger.exception(
            "Document indexing failed: %s",
            exc,
        )

        raise HTTPException(
            500,
            "Document indexing failed. Check server logs for details.",
        ) from exc


@router.get("/documents")
def documents(db: Session = Depends(get_db)):
    from app.db.models import Document

    docs = db.scalars(
        select(Document).order_by(
            Document.created_at.desc()
        )
    ).all()

    return [
        {
            "id": d.id,
            "filename": d.filename,
            "title": d.title,
            "authors": d.authors,
            "year": d.year,
            "doi": d.doi,
        }
        for d in docs
    ]


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    from app.db.models import Document
    from app.services.pipeline import invalidate_bm25

    doc = db.get(Document, document_id)

    if not doc:
        raise HTTPException(
            404,
            "Document not found",
        )

    path = Path(doc.path)

    try:
        # The database is the source of truth. Commit the deletion, then clean storage.
        db.delete(doc)
        db.commit()

        invalidate_bm25()

        cleanup_pending = False

        try:
            path.unlink(missing_ok=True)
        except OSError:
            cleanup_pending = True
            logger.exception(
                "Document %s deleted from DB but file cleanup failed: %s",
                document_id,
                path,
            )

        return {
            "deleted": document_id,
            "cleanup_pending": cleanup_pending,
        }

    except Exception:
        db.rollback()
        raise HTTPException(
            500,
            "Document deletion failed. Check server logs for details.",
        )


@router.post("/query")
def query(
    req: QueryRequest,
    db: Session = Depends(get_db),
):
    if not req.question.strip():
        raise HTTPException(
            400,
            "question must not be blank",
        )

    from app.db.models import Document
    from app.services.pipeline import answer_question

    if (
        req.year_from is not None
        and req.year_to is not None
        and req.year_from > req.year_to
    ):
        raise HTTPException(
            400,
            "year_from must be <= year_to",
        )

    if req.document_ids:
        found = set(
            db.scalars(
                select(Document.id).where(
                    Document.id.in_(req.document_ids)
                )
            ).all()
        )

        missing = sorted(
            set(req.document_ids) - found
        )

        if missing:
            raise HTTPException(
                400,
                f"Unknown document_ids: {missing}",
            )

    if req.year_from is not None or req.year_to is not None:
        stmt = select(Document.id)

        if req.year_from is not None:
            stmt = stmt.where(
                Document.year >= req.year_from
            )

        if req.year_to is not None:
            stmt = stmt.where(
                Document.year <= req.year_to
            )

        year_ids = set(
            db.scalars(stmt).all()
        )

        document_ids = sorted(
            year_ids
            if not req.document_ids
            else set(req.document_ids) & year_ids
        )

        if not document_ids:
            return {
                "answer": (
                    "No indexed documents match the "
                    "requested metadata filters."
                ),
                "sources": [],
                "citation_verification": {
                    "valid": False,
                    "reason": "no_matching_documents",
                },
                "queries": [],
                "timings_ms": {},
            }

    else:
        document_ids = req.document_ids or None

    try:
        return answer_question(
            db,
            req.question.strip(),
            document_ids,
            req.top_k,
            req.use_hyde,
        )

    except RuntimeError as exc:
        logger.exception(
            "Query dependency failure: %s",
            exc,
        )
        raise HTTPException(
            503,
            "Query dependencies are unavailable. Check server logs for details.",
        )

    except Exception as exc:
        logger.exception(
            "Query failed: %s",
            exc,
        )
        raise HTTPException(
            500,
            "Query failed. Check server logs for details.",
        ) from exc