"""Request-scoped authentication context using contextvars."""

from contextvars import ContextVar

# Per-request auth state — set by BearerAuthMiddleware, read by MCP tools
_auth_user_id: ContextVar[str | None] = ContextVar('auth_user_id', default=None)
_auth_group_id: ContextVar[str | None] = ContextVar('auth_group_id', default=None)


def set_auth_context(user_id: str, group_id: str) -> None:
    _auth_user_id.set(user_id)
    _auth_group_id.set(group_id)


def get_auth_user_id() -> str | None:
    return _auth_user_id.get()


def get_auth_group_id() -> str | None:
    return _auth_group_id.get()
