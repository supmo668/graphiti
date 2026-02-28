#!/usr/bin/env python3
"""CLI utility to manage API keys for the Graphiti MCP auth gateway.

Usage:
    # Create a key (prints the raw key once)
    python -m auth.manage_keys create --user alice --group alice_memory

    # List active keys
    python -m auth.manage_keys list

    # Revoke a key by id
    python -m auth.manage_keys revoke --id 3
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure the src directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load .env from mcp_server directory
env_file = Path(__file__).parent.parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)


async def cmd_create(args):
    from auth.db import create_api_key, ensure_schema

    await ensure_schema()
    raw_key = await create_api_key(
        user_id=args.user,
        group_id=args.group,
        label=args.label or '',
    )
    print(f'API key created (save this — it will not be shown again):\n\n  {raw_key}\n')
    print(f'User: {args.user}  Group: {args.group}')


async def cmd_list(args):
    from auth.db import ensure_schema, get_pool

    await ensure_schema()
    pool = await get_pool()
    rows = await pool.fetch(
        'SELECT id, user_id, group_id, label, is_active, created_at FROM api_keys ORDER BY id'
    )
    if not rows:
        print('No API keys found.')
        return
    print(f'{"ID":>4}  {"User":<20}  {"Group":<20}  {"Label":<20}  {"Active":<6}  Created')
    print('-' * 100)
    for r in rows:
        print(
            f'{r["id"]:>4}  {r["user_id"]:<20}  {r["group_id"]:<20}  '
            f'{r["label"]:<20}  {"yes" if r["is_active"] else "no":<6}  {r["created_at"]}'
        )


async def cmd_revoke(args):
    from auth.db import ensure_schema, get_pool

    await ensure_schema()
    pool = await get_pool()
    result = await pool.execute('UPDATE api_keys SET is_active = FALSE WHERE id = $1', args.id)
    print(f'Revoked key id={args.id} ({result})')


def main():
    parser = argparse.ArgumentParser(description='Manage Graphiti MCP API keys')
    sub = parser.add_subparsers(dest='command', required=True)

    p_create = sub.add_parser('create', help='Create a new API key')
    p_create.add_argument('--user', required=True, help='User ID')
    p_create.add_argument('--group', required=True, help='Group ID (memory tenant)')
    p_create.add_argument('--label', help='Human-readable label')

    sub.add_parser('list', help='List all API keys')

    p_revoke = sub.add_parser('revoke', help='Revoke an API key')
    p_revoke.add_argument('--id', type=int, required=True, help='Key ID to revoke')

    args = parser.parse_args()

    # Ensure DATABASE_URL is set
    if not (os.environ.get('AUTH_DATABASE_URL') or os.environ.get('DATABASE_URL')):
        print('ERROR: Set AUTH_DATABASE_URL or DATABASE_URL environment variable', file=sys.stderr)
        sys.exit(1)

    coro = {'create': cmd_create, 'list': cmd_list, 'revoke': cmd_revoke}[args.command]
    asyncio.run(coro(args))


if __name__ == '__main__':
    main()
