"""Add User table and user_id

Revision ID: 2fc7c8c1c46d
Revises: 94fa724aefca
Create Date: 2026-03-27 14:45:17.436672

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import hashlib

# revision identifiers, used by Alembic.
revision: str = '2fc7c8c1c46d'
down_revision: Union[str, Sequence[str], None] = '94fa724aefca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_users_table(existing_tables: list[str]) -> None:
    """Create users table if not exists."""
    if 'users' not in existing_tables:
        op.create_table(
            'users',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('name', sa.String(), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_users_name'), 'users', ['name'], unique=True)


def _add_user_id_columns(inspector) -> None:
    """Add columns and FKs to existing tables using batch_alter_table."""
    for table_name in ['edges', 'embeddings', 'entities']:
        with op.batch_alter_table(table_name, schema=None) as batch_op:
            columns = [c['name'] for c in inspector.get_columns(table_name)]
            if 'user_id' not in columns:
                batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=True))
                batch_op.create_index(batch_op.f(f'ix_{table_name}_user_id'), ['user_id'], unique=False)
                batch_op.create_foreign_key(
                    f'fk_{table_name}_user_id', 'users',
                    ['user_id'], ['id'], ondelete='CASCADE'
                )


def _migrate_user_data(conn, inspector) -> None:
    """Data Migration: Populate users table, set user_id, and SALT IDs."""
    def salt_fact_id(text: str, user_id: int = None) -> str:
        salt = f"{user_id}_" if user_id is not None else ""
        hashed = hashlib.sha256((salt + text).encode("utf-8")).hexdigest()
        return "f_" + hashed[:8]

    entities_cols = [c['name'] for c in inspector.get_columns('entities')]
    edges_cols = [c['name'] for c in inspector.get_columns('edges')]
    embeddings_cols = [c['name'] for c in inspector.get_columns('embeddings')]

    user_names = []
    if 'user_name' in entities_cols or 'user_name' in edges_cols or 'user_name' in embeddings_cols:
        # Get unique user names from all tables that have the column
        queries = []
        if 'user_name' in entities_cols:
            queries.append("SELECT DISTINCT user_name FROM entities WHERE user_name IS NOT NULL")
        if 'user_name' in edges_cols:
            queries.append("SELECT DISTINCT user_name FROM edges WHERE user_name IS NOT NULL")
        if 'user_name' in embeddings_cols:
            queries.append("SELECT DISTINCT user_name FROM embeddings WHERE user_name IS NOT NULL")

        if queries:  # Only execute if there are tables with user_name
            res = conn.execute(sa.text(" UNION ".join(queries)))
            user_names = [row[0] for row in res]

    for name in user_names:
        # Insert user if not exists
        conn.execute(sa.text("INSERT INTO users (name) VALUES (:name)"), {"name": name})
        user_id = conn.execute(sa.text("SELECT id FROM users WHERE name = :name"), {"name": name}).scalar()

        # Update initial user_id columns
        conn.execute(sa.text(
            "UPDATE entities SET user_id = :uid WHERE user_name = :uname"),
            {"uid": user_id, "uname": name})
        conn.execute(sa.text(
            "UPDATE edges SET user_id = :uid WHERE user_name = :uname"),
            {"uid": user_id, "uname": name})
        conn.execute(sa.text(
            "UPDATE embeddings SET user_id = :uid WHERE user_name = :uname"),
            {"uid": user_id, "uname": name})

        # NOW SALT THE IDs for this user to ensure isolation and consistency with runtime logic
        entities = conn.execute(
            sa.text("SELECT id, name FROM entities WHERE user_id = :uid"),
            {"uid": user_id}
        ).fetchall()
        for old_id, ent_name in entities:
            new_id = salt_fact_id(ent_name, user_id)
            if old_id != new_id:
                # Update references in edges
                conn.execute(sa.text(
                    "UPDATE edges SET source_id = :nid WHERE source_id = :oid AND user_id = :uid"),
                    {"nid": new_id, "oid": old_id, "uid": user_id})
                conn.execute(sa.text(
                    "UPDATE edges SET target_id = :nid WHERE target_id = :oid AND user_id = :uid"),
                    {"nid": new_id, "oid": old_id, "uid": user_id})
                conn.execute(sa.text(
                    "UPDATE embeddings SET source_id = :nid WHERE source_id = :oid "
                    "AND user_id = :uid AND collection = 'entity'"),
                    {"nid": new_id, "oid": old_id, "uid": user_id})
                # Finally update the entity ID itself
                conn.execute(sa.text("UPDATE entities SET id = :nid WHERE id = :oid"), {"nid": new_id, "oid": old_id})


def _cleanup_user_name_columns(inspector) -> None:
    """Final Cleanup: Drop redundant user_name columns."""
    entities_cols = [c['name'] for c in inspector.get_columns('entities')]
    edges_cols = [c['name'] for c in inspector.get_columns('edges')]
    embeddings_cols = [c['name'] for c in inspector.get_columns('embeddings')]

    for table_name, cols in [('entities', entities_cols), ('edges', edges_cols), ('embeddings', embeddings_cols)]:
        with op.batch_alter_table(table_name, schema=None) as batch_op:
            if 'user_name' in cols:
                batch_op.drop_index(batch_op.f(f'ix_{table_name}_user_name'))
                batch_op.drop_column('user_name')


def upgrade() -> None:
    """Upgrade schema using batch mode for SQLite support."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    _create_users_table(existing_tables)
    _add_user_id_columns(inspector)
    _migrate_user_data(conn, inspector)
    _cleanup_user_name_columns(inspector)


def downgrade() -> None:
    """Downgrade schema using batch mode."""
    # Add back the user_name columns if needed
    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_name', sa.VARCHAR(), nullable=True))
    with op.batch_alter_table('embeddings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_name', sa.VARCHAR(), nullable=True))
    with op.batch_alter_table('edges', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_name', sa.VARCHAR(), nullable=True))

    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.drop_constraint('fk_entities_user_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_entities_user_id'))
        batch_op.drop_column('user_id')

    with op.batch_alter_table('embeddings', schema=None) as batch_op:
        batch_op.drop_constraint('fk_embeddings_user_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_embeddings_user_id'))
        batch_op.drop_column('user_id')

    with op.batch_alter_table('edges', schema=None) as batch_op:
        batch_op.drop_constraint('fk_edges_user_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_edges_user_id'))
        batch_op.drop_column('user_id')

    op.drop_index(op.f('ix_users_name'), table_name='users')
    op.drop_table('users')
