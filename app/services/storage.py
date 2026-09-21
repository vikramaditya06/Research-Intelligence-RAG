from pathlib import Path

from sqlalchemy import select, update

from app.core.logging import logger
from app.db.models import Document
from app.core.config import settings


def reconcile_document_storage(db) -> dict[str, int]:
    """Repair DB/file mismatches left by crashes around the atomic rename.

    A committed document may temporarily point at a hidden .uploading file.
    On startup, finish that rename when possible. Orphan staging files that
    have no corresponding committed document are removed.
    """
    data_dir = Path(settings.data_dir)
    repaired = 0
    removed_orphans = 0
    missing_references = 0
    docs = db.scalars(select(Document)).all()
    # Normalize paths so absolute/relative DB values compare consistently with
    # files discovered under the application-owned data directory.
    by_path = {Path(d.path).resolve(): d for d in docs}

    for doc in docs:
        current = Path(doc.path)
        if not current.name.startswith(f".{doc.file_hash}_") or not current.name.endswith(".uploading"):
            continue
        final_path = data_dir / f"{doc.file_hash[:16]}_{doc.filename}"
        try:
            if final_path.exists():
                current.unlink(missing_ok=True)
            else:
                current.replace(final_path)
            db.execute(update(Document).where(Document.id == doc.id).values(path=str(final_path)))
            db.commit()
            doc.path = str(final_path)
            # Keep the in-memory reference map in sync so the repaired final
            # file is not mistaken for an orphan later in this same pass.
            by_path.pop(current.resolve(), None)
            by_path[final_path.resolve()] = doc
            repaired += 1
        except Exception:
            db.rollback()
            logger.exception("Storage reconciliation failed for document %s", doc.id)

    # Detect committed DB rows whose referenced file disappeared.  Do not
    # delete the document automatically: its chunks/metadata may still be
    # valuable, and an operator can restore the file and retry.
    for doc in docs:
        current = Path(doc.path)
        if not current.exists():
            missing_references += 1
            logger.error("Document %s references missing storage file %s", doc.id, current)

    for staging in data_dir.iterdir():
        if not staging.is_file() or not staging.name.endswith(".uploading"):
            continue
        if staging.resolve() not in by_path:
            try:
                staging.unlink(missing_ok=True)
                removed_orphans += 1
            except OSError:
                logger.exception("Could not remove orphan staging file %s", staging)

    # A document deletion is intentionally committed before filesystem cleanup.
    # If the process crashes or unlink() fails, the final PDF can survive after
    # its DB row is gone. Since this directory is owned by the application, any
    # non-staging PDF that is not referenced by a committed Document is safe to
    # treat as an orphan and remove during startup reconciliation.
    for pdf in data_dir.iterdir():
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            continue
        # Only delete files produced by this application.  User-managed PDFs
        # placed in the directory are not implicitly application-owned.
        prefix = pdf.name.split("_", 1)[0]
        if len(prefix) != 16 or any(ch not in "0123456789abcdef" for ch in prefix.lower()):
            continue
        if pdf.resolve() in by_path:
            continue
        try:
            pdf.unlink(missing_ok=True)
            removed_orphans += 1
            logger.warning("Removed orphan document file %s", pdf)
        except OSError:
            logger.exception("Could not remove orphan document file %s", pdf)

    return {"repaired": repaired, "removed_orphans": removed_orphans, "missing_references": missing_references}
