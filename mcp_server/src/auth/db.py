"""API key store for bearer token authentication.

Supports two backends (checked in order):
  1. Environment variable keys (always available, zero dependencies)
  2. Postgres-backed keys (optional, requires asyncpg + AUTH_DATABASE_URL)

Environment variable format (AUTH_KEYS):
  AUTH_KEYS="token1:user1:group1,token2:user2:group2"

Each entry is "raw_key:user_id:group_id" separated by commas.
"""

import hashlib
import logging
import os
import secrets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory key store populated from AUTH_KEYS env var
# ---------------------------------------------------------------------------
_static_keys: dict[str, dict[str, str]] = {}  # hash -> {user_id, group_id}


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of a bearer token for safe storage/comparison."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _load_static_keys() -> int:
    """Parse AUTH_KEYS env var into the in-memory store. Returns count loaded."""
    raw = os.environ.get('AUTH_KEYS', '')
    if not raw:
        return 0
    count = 0
    for entry in raw.split(','):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(':')
        if len(parts) != 3:
            logger.warning(f'Skipping malformed AUTH_KEYS entry (expected key:user:group)')
            continue
        key, user_id, group_id = parts
        _static_keys[hash_key(key)] = {'user_id': user_id, 'group_id': group_id}
        count += 1
    return count


# ---------------------------------------------------------------------------
# Optional Postgres backend
# ---------------------------------------------------------------------------
_pool = None  # asyncpg.Pool | None — lazy import to avoid hard dep

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


async def _get_pool():
    """Get or create the Postgres connection pool (lazy)."""
    global _pool
    if _pool is None:
        import asyncpg

        dsn = os.environ.get('AUTH_DATABASE_URL') or os.environ.get('DATABASE_URL')
        if not dsn:
            return None
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        logger.info('Auth Postgres pool created')
    return _pool


async def ensure_schema() -> None:
    """Initialise key stores. Static keys are always loaded; Postgres is optional."""
    count = _load_static_keys()
    if count:
        logger.info(f'Loaded {count} static API key(s) from AUTH_KEYS')

    dsn = os.environ.get('AUTH_DATABASE_URL') or os.environ.get('DATABASE_URL')
    if dsn:
        try:
            pool = await _get_pool()
            if pool:
                async with pool.acquire() as conn:
                    await conn.execute(SCHEMA_SQL)
                logger.info('Auth Postgres schema ensured')
        except Exception as e:
            logger.warning(f'Postgres auth unavailable (static keys still active): {e}')

    if not count and not dsn:
        raise RuntimeError(
            'No auth backend configured. Set AUTH_KEYS env var or AUTH_DATABASE_URL.'
        )


async def lookup_key(raw_key: str) -> dict | None:
    """Resolve a bearer token to {user_id, group_id} or None.

    Checks static keys first, then Postgres if available.
    """
    h = hash_key(raw_key)

    # 1. Static keys (env var)
    if h in _static_keys:
        return _static_keys[h]

    # 2. Postgres (if configured)
    try:
        pool = await _get_pool()
        if pool:
            row = await pool.fetchrow(
                'SELECT user_id, group_id FROM api_keys WHERE key_hash = $1 AND is_active',
                h,
            )
            if row:
                return dict(row)
    except Exception:
        pass  # Postgres down — fall through to None

    return None


async def create_api_key(user_id: str, group_id: str, label: str = '') -> str:
    """Provision a new API key in Postgres. Returns the raw key (shown once)."""
    import asyncpg  # noqa: F811

    raw_key = f'gm_{secrets.token_urlsafe(32)}'
    pool = await _get_pool()
    if pool is None:
        raise RuntimeError('Postgres not configured — cannot create persistent keys')
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
