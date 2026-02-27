"""Starlette middleware for Bearer token authentication."""

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from auth.context import set_auth_context
from auth.db import lookup_key

logger = logging.getLogger(__name__)

# Paths that bypass authentication
PUBLIC_PATHS = frozenset({'/health', '/health/'})


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validate Authorization: Bearer <token> and set auth context.

    - Skips /health so load-balancers work without credentials.
    - Returns 401 on missing/invalid tokens.
    - On success, sets per-request contextvars (user_id, group_id).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path.rstrip('/')
        if path in {p.rstrip('/') for p in PUBLIC_PATHS}:
            return await call_next(request)

        auth_header = request.headers.get('authorization', '')
        if not auth_header.lower().startswith('bearer '):
            return JSONResponse(
                {'error': 'Missing or malformed Authorization header'},
                status_code=401,
                headers={'WWW-Authenticate': 'Bearer'},
            )

        token = auth_header[7:].strip()
        if not token:
            return JSONResponse(
                {'error': 'Empty bearer token'},
                status_code=401,
                headers={'WWW-Authenticate': 'Bearer'},
            )

        result = await lookup_key(token)
        if result is None:
            logger.warning('Invalid bearer token attempted')
            return JSONResponse(
                {'error': 'Invalid or revoked API key'},
                status_code=401,
                headers={'WWW-Authenticate': 'Bearer'},
            )

        # Set per-request auth context
        set_auth_context(user_id=result['user_id'], group_id=result['group_id'])
        logger.debug(f'Authenticated user={result["user_id"]} group={result["group_id"]}')

        return await call_next(request)
