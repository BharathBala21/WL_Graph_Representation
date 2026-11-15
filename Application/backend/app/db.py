"""SQLite storage for users and their diagnoses (Python's built-in sqlite3)."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DATA_DIR = os.environ.get('PLANTCARE_DATA_DIR',
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
DB_PATH = os.path.join(DATA_DIR, 'plantcare.db')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL,
    filename      TEXT,
    image_file    TEXT NOT NULL,
    overlay_file  TEXT NOT NULL,
    crop          TEXT NOT NULL,
    stress_status TEXT NOT NULL,
    top_condition TEXT NOT NULL,
    note          TEXT,
    result_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id, created_at DESC);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys = ON')
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with connect() as con:
        con.executescript(SCHEMA)


# ---------------------------------------------------------------- users
def create_user(name: str, email: str, password_hash: str) -> dict:
    with connect() as con:
        cur = con.execute('INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
                          (name, email, password_hash, now()))
        return get_user(cur.lastrowid, con)


def get_user(user_id: int, con=None) -> dict | None:
    if con is None:
        with connect() as c:
            return get_user(user_id, c)
    row = con.execute('SELECT id, name, email, created_at FROM users WHERE id = ?', (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    with connect() as con:
        row = con.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------- predictions
def _out(row) -> dict:
    d = dict(row)
    d['result'] = json.loads(d.pop('result_json'))
    d['image_url'] = f"/files/{d.pop('image_file')}"
    d['overlay_url'] = f"/files/{d.pop('overlay_file')}"
    d.pop('user_id', None)
    return d


def add_prediction(pid: str, user_id: int, filename: str, image_file: str, overlay_file: str, result: dict) -> dict:
    with connect() as con:
        con.execute(
            'INSERT INTO predictions (id, user_id, created_at, filename, image_file, overlay_file, crop, '
            'stress_status, top_condition, result_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (pid, user_id, now(), filename, image_file, overlay_file, result['crop']['code'],
             result['stress']['status'], result['conditions'][0]['code'], json.dumps(result)))
        return _out(con.execute('SELECT * FROM predictions WHERE id = ?', (pid,)).fetchone())


def list_predictions(user_id: int) -> list[dict]:
    with connect() as con:
        rows = con.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
        return [_out(r) for r in rows]


def get_prediction(user_id: int, pid: str) -> dict | None:
    with connect() as con:
        row = con.execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (pid, user_id)).fetchone()
        return _out(row) if row else None


def update_note(user_id: int, pid: str, note: str) -> dict | None:
    with connect() as con:
        con.execute('UPDATE predictions SET note = ? WHERE id = ? AND user_id = ?', (note, pid, user_id))
    return get_prediction(user_id, pid)


def delete_prediction(user_id: int, pid: str) -> dict | None:
    """Deletes the record and returns it (so the caller can remove its files)."""
    rec = get_prediction(user_id, pid)
    if rec:
        with connect() as con:
            con.execute('DELETE FROM predictions WHERE id = ? AND user_id = ?', (pid, user_id))
    return rec


def stats(user_id: int) -> dict:
    with connect() as con:
        total = con.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,)).fetchone()[0]
        stressed = con.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND stress_status = 'stressed'",
                               (user_id,)).fetchone()[0]
        by_crop = {r[0]: r[1] for r in con.execute(
            'SELECT crop, COUNT(*) FROM predictions WHERE user_id = ? GROUP BY crop', (user_id,))}
    return {'total': total, 'stressed': stressed, 'healthy': total - stressed, 'by_crop': by_crop}
