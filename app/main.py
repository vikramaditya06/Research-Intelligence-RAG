from fastapi import FastAPI
from app.api.routes import router
from app.core.logging import request_logger, logger
from app.db.session import SessionLocal, engine

app = FastAPI(title="Research Intelligence RAG", version="1.0.0")
app.middleware("http")(request_logger)
app.include_router(router)


@app.on_event("startup")
def reconcile_storage_on_startup():
    if engine is None:
        logger.warning("Skipping storage reconciliation because the database driver is unavailable")
        return
    from app.services.storage import reconcile_document_storage
    db = SessionLocal()
    try:
        result = reconcile_document_storage(db)
        if result["repaired"] or result["removed_orphans"]:
            logger.info("Storage reconciliation: %s", result)
    except Exception:
        logger.exception("Storage reconciliation could not complete")
    finally:
        db.close()
