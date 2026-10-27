"""initial legal agent schema"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("cases", sa.Column("id", sa.String(64), primary_key=True), sa.Column("tenant_id", sa.String(128), nullable=False), sa.Column("title", sa.String(200)), sa.Column("created_at", sa.DateTime()))
    op.create_table("analysis_records", sa.Column("id", sa.String(64), primary_key=True), sa.Column("case_id", sa.String(64)), sa.Column("tenant_id", sa.String(128)), sa.Column("request_id", sa.String(64)), sa.Column("status", sa.String(32)), sa.Column("output_json", sa.JSON()), sa.Column("created_at", sa.DateTime()))
    op.create_table("review_records", sa.Column("id", sa.String(64), primary_key=True), sa.Column("analysis_id", sa.String(64)), sa.Column("tenant_id", sa.String(128)), sa.Column("reviewer", sa.String(128)), sa.Column("decision", sa.String(32)), sa.Column("comment", sa.Text()), sa.Column("created_at", sa.DateTime()))

def downgrade():
    op.drop_table("review_records")
    op.drop_table("analysis_records")
    op.drop_table("cases")
