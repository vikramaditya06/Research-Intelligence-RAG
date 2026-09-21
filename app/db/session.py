from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

# Keep imports of API schemas and models usable in lightweight environments
# (for example unit-test collection) even when the optional PostgreSQL driver
# has not been installed yet. Production startup still fails clearly when a
# database session is actually requested.
try:
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    _engine_error = None
except (ImportError, ModuleNotFoundError) as exc:
    engine = None
    SessionLocal = sessionmaker(autoflush=False, autocommit=False)
    _engine_error = exc


class Base(DeclarativeBase):
    pass


def get_db():
    if engine is None:
        raise RuntimeError(
            "Database driver is unavailable. Install project dependencies with "
            "'pip install -r requirements.txt'."
        ) from _engine_error
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
