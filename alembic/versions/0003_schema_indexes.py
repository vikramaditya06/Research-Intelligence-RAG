"""add indexes present in current ORM metadata"""
from alembic import op

revision = "0003_schema_indexes"
down_revision = "0002_chunk_page_ranges"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index("ix_documents_created_at", "documents", ["created_at"])
    op.create_index("ix_parent_chunks_section", "parent_chunks", ["section"])


def downgrade():
    op.drop_index("ix_parent_chunks_section", table_name="parent_chunks")
    op.drop_index("ix_documents_created_at", table_name="documents")
