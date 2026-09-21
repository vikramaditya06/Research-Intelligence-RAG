"""make page-range coordinate systems explicit in chunk metadata"""
from alembic import op

revision = "0004_page_coords"
down_revision = "0003_schema_indexes"
branch_labels = None
depends_on = None


def upgrade():
    # metadata_json is SQLAlchemy JSON (JSON on PostgreSQL). Convert only rows
    # produced by older builds, preserving all other metadata keys.
    op.execute("""
        UPDATE parent_chunks
        SET metadata_json = (metadata_json::jsonb - 'page_ranges' ||
            jsonb_build_object('parent_page_ranges', metadata_json::jsonb -> 'page_ranges'))::json
        WHERE metadata_json IS NOT NULL
          AND metadata_json::jsonb ? 'page_ranges'
          AND NOT (metadata_json::jsonb ? 'parent_page_ranges')
    """)
    op.execute("""
        UPDATE chunks
        SET metadata_json = (metadata_json::jsonb - 'page_ranges' ||
            jsonb_build_object('child_page_ranges', metadata_json::jsonb -> 'page_ranges'))::json
        WHERE metadata_json IS NOT NULL
          AND metadata_json::jsonb ? 'page_ranges'
          AND NOT (metadata_json::jsonb ? 'child_page_ranges')
    """)


def downgrade():
    op.execute("""
        UPDATE parent_chunks
        SET metadata_json = (metadata_json::jsonb - 'parent_page_ranges' ||
            jsonb_build_object('page_ranges', metadata_json::jsonb -> 'parent_page_ranges'))::json
        WHERE metadata_json IS NOT NULL
          AND metadata_json::jsonb ? 'parent_page_ranges'
    """)
    op.execute("""
        UPDATE chunks
        SET metadata_json = (metadata_json::jsonb - 'child_page_ranges' ||
            jsonb_build_object('page_ranges', metadata_json::jsonb -> 'child_page_ranges'))::json
        WHERE metadata_json IS NOT NULL
          AND metadata_json::jsonb ? 'child_page_ranges'
    """)
