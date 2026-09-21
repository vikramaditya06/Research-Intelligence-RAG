"""add exact child chunk page ranges"""
from alembic import op
import sqlalchemy as sa

revision = "0002_chunk_page_ranges"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.execute("UPDATE chunks SET page_start = page, page_end = page")
    op.alter_column("chunks", "page_start", nullable=False)
    op.alter_column("chunks", "page_end", nullable=False)

def downgrade():
    op.drop_column("chunks", "page_end")
    op.drop_column("chunks", "page_start")
