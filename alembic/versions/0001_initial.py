"""initial research rag schema"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    print(">>> 0001_INITIAL UPGRADE IS RUNNING <<<")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table("documents",
        sa.Column("id", sa.String(36), primary_key=True), sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("title", sa.String(500)), sa.Column("path", sa.String(1000), nullable=False), sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("authors", sa.JSON()), sa.Column("abstract", sa.Text()), sa.Column("year", sa.Integer()), sa.Column("doi", sa.String(500)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"], unique=True)
    op.create_index("ix_documents_year", "documents", ["year"])
    op.create_table("parent_chunks",
        sa.Column("id", sa.String(36), primary_key=True), sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False), sa.Column("page_end", sa.Integer(), nullable=False), sa.Column("section", sa.String(500)),
        sa.Column("text", sa.Text(), nullable=False), sa.Column("metadata_json", sa.JSON()),
    )
    op.create_index("ix_parent_chunks_document_id", "parent_chunks", ["document_id"])
    op.create_table("chunks",
        sa.Column("id", sa.String(36), primary_key=True), sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_id", sa.String(36), sa.ForeignKey("parent_chunks.id", ondelete="CASCADE"), nullable=False), sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(500)), sa.Column("text", sa.Text(), nullable=False), sa.Column("metadata_json", sa.JSON()), sa.Column("embedding", Vector(1536)),
    )
    for name, cols in [("ix_chunks_document_id", ["document_id"]),("ix_chunks_parent_id", ["parent_id"]),("ix_chunks_page", ["page"]),("ix_chunks_section", ["section"])]: op.create_index(name, "chunks", cols)
    op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")

def downgrade():
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_table("chunks"); op.drop_table("parent_chunks"); op.drop_index("ix_documents_year", table_name="documents"); op.drop_index("ix_documents_file_hash", table_name="documents"); op.drop_table("documents")
