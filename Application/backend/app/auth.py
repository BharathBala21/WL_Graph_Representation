"""Password hashing (PBKDF2-SHA256, standard library) and JWT access tokens."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from .db import DATA_DIR

ITERATIONS = 240_000
TOKEN_DAYS = 7


def _secret() -> str:
    """JWT signing key: PLANTCARE_SECRET env var, else a random key kept in data/secret.key."""
    env = os.environ.get('PLANTCARE_SECRET')
    if env:
        return env
    path = os.path.join(DATA_DIR, 'secret.key')
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, 'w') as f:
            f.write(secrets.token_urlsafe(48))
    with open(path) as f:
        return f.read().strip()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, ITERATIONS)
    return f'pbkdf2_sha256${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}'


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt, digest = stored.split('$')
        test = hashlib.pbkdf2_hmac('sha256', password.encode(), base64.b64decode(salt), int(iters))
        return hmac.compare_digest(test, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def create_token(user_id: int) -> str:
    payload = {'sub': str(user_id), 'exp': datetime.now(timezone.utc) + timedelta(days=TOKEN_DAYS)}
    return jwt.encode(payload, _secret(), algorithm='HS256')


def read_token(token: str) -> int | None:
    try:
        return int(jwt.decode(token, _secret(), algorithms=['HS256'])['sub'])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
