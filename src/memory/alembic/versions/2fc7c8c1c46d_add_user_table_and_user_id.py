"""Add User table and user_id

Revision ID: 2fc7c8c1c46d
Revises: 94fa724aefca
Create Date: 2026-03-27 14:45:17.436672

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2fc7c8c1c46d'
down_revision: Union[str, Sequence[str], None] = '94fa724aefca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema using batch mode for SQLite support."""
    # 1. Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_name'), 'users', ['name'], unique=True)

    # 2. Add columns and FKs to existing tables using batch_alter_table
    with op.batch_alter_table('edges', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_edges_user_id'), ['user_id'], unique=False)
        batch_op.create_foreign_key('fk_edges_user_id', 'users', ['user_id'], ['id'], ondelete='CASCADE')

    with op.batch_alter_table('embeddings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_embeddings_user_id'), ['user_id'], unique=False)
        batch_op.create_foreign_key('fk_embeddings_user_id', 'users', ['user_id'], ['id'], ondelete='CASCADE')

    with op.batch_alter_table('entities', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_entities_user_id'), ['user_id'], unique=False)
        batch_op.create_foreign_key('fk_entities_user_id', 'users', ['user_id'], ['id'], ondelete='CASCADE')

    # 3. Data Migration: Populate users table and set user_id in other tables
    conn = op.get_bind()

    # Get unique user names from all tables
    res = conn.execute(sa.text(
        "SELECT DISTINCT user_name FROM entities WHERE user_name IS NOT NULL "
        "UNION "
        "SELECT DISTINCT user_name FROM edges WHERE user_name IS NOT NULL "
        "UNION "
        "SELECT DISTINCT user_name FROM embeddings WHERE user_name IS NOT NULL"
    ))
    user_names = [row[0] for row in res]

    for name in user_names:
        # Insert user if not exists
        conn.execute(sa.text("INSERT INTO users (name) VALUES (:name)"), {"name": name})

        # Get the ID of the newly created user
        user_id = conn.execute(sa.text("SELECT id FROM users WHERE name = :name"), {"name": name}).scalar()

        # Update related tables
        conn.execute(sa.text("UPDATE entities SET user_id = :uid WHERE user_name = :uname"), {"uid": user_id, "uname": name})
        conn.execute(sa.text("UPDATE edges SET user_id = :uid WHERE user_name = :uname"), {"uid": user_id, "uname": name})
        conn.execute(sa.text("UPDATE embeddings SET user_id = :uid WHERE user_name = :uname"), {"uid": user_id, "uname": name})


def downgrade() -> None:
    """Downgrade schema using batch mode."""
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