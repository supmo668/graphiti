"""Postgres-backed API key store for bearer token authentication."""

import hashlib
import logging
import os
import secrets

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          SERIAL PRIMARY KEY,
    key_hash    VARCHAR(64) NOT NULL UNIQUE,
    user_id     VARCHAR(255) NOT NULL,
    group_id    VARCHAR(255) NOT NULL,
    label       VARCHAR(255) DEFAULT '',
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash) WHERE is_active;
"""


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of a bearer token for safe storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        dsn = os.environ.get('AUTH_DATABASE_URL') or os.environ.get('DATABASE_URL')
        if not dsn:
            raise RuntimeError(
                'AUTH_DATABASE_URL or DATABASE_URL must be set for bearer auth'
            )
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        logger.info('Auth database pool created')
    return _pool


async def ensure_schema() -> None:
    """Create the api_keys table if it does not exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    logger.info('Auth schema ensured')


async def lookup_key(raw_key: str) -> dict | None:
    """Resolve a bearer token to (user_id, group_id) or None."""
    pool = await get_pool()
    h = hash_key(raw_key)
    row = await pool.fetchrow(
        'SELECT user_id, group_id FROM api_keys WHERE key_hash = $1 AND is_active',
        h,
    )
    if row is None:
        return None
    return dict(row)


async def create_api_key(user_id: str, group_id: str, label: str = '') -> str:
    """Provision a new API key. Returns the raw key (shown once)."""
    raw_key = f'gm_{secrets.token_urlsafe(32)}'
    pool = await get_pool()
    await pool.execute(
        'INSERT INTO api_keys (key_hash, user_id, group_id, label) VALUES ($1, $2, $3, $4)',
        hash_key(raw_key),
        user_id,
        group_id,
        label,
    )
    logger.info(f'Created API key for user={user_id} group={group_id} label={label}')
    return raw_key


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
