"""
database/database.py
====================
SQLite persistence layer for StoryWeave multi-user support.

Responsibilities:
- Create / migrate the database schema on first run.
- Provide low-level CRUD helpers used by auth_service and learner_service.
- Enforce user-scoped queries so no cross-user data leaks.
- Never raise raw sqlite3 exceptions to callers — wraps them in RuntimeError.

Database file:  StoryWeave/database/storyweave.db
"""

import os
import sqlite3
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── Path ───────────────────────────────────────────────────────────────────────
_DB_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_DB_DIR, "storyweave.db")


# ── Schema ─────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS learner_profiles (
    user_id            TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    estimated_level    TEXT NOT NULL DEFAULT 'Intermediate',
    overall_mastery    REAL NOT NULL DEFAULT 0.0,
    preferred_modality TEXT NOT NULL DEFAULT 'Not enough data',
    updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concept_mastery (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id        TEXT NOT NULL,
    concept       TEXT NOT NULL,
    mastery_score REAL NOT NULL DEFAULT 0.0,
    last_updated  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, doc_id, concept)
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id          TEXT NOT NULL,
    score           INTEGER NOT NULL DEFAULT 0,
    total_questions INTEGER NOT NULL DEFAULT 0,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS interaction_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    interaction_type TEXT NOT NULL,
    doc_id           TEXT NOT NULL DEFAULT '',
    timestamp        TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ── Connection ─────────────────────────────────────────────────────────────────

@contextmanager
def _get_connection(db_path: str = DB_PATH):
    """Yield a SQLite connection; commit on clean exit, rollback on error."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Initialise the database: create directory and all tables if absent. Idempotent."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        with _get_connection(db_path) as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.info("StoryWeave database initialised at %s", db_path)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database initialisation failed: {exc}") from exc


# ── Users ──────────────────────────────────────────────────────────────────────

def insert_user(user_id: str, username: str, password_hash: str,
                db_path: str = DB_PATH) -> None:
    """Insert a new user row. Raises RuntimeError on duplicate username."""
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO users (user_id, username, password_hash) VALUES (?, ?, ?)",
                (user_id, username, password_hash),
            )
    except sqlite3.IntegrityError as exc:
        raise RuntimeError(f"Username '{username}' already exists.") from exc
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not create user: {exc}") from exc


def get_user_by_username(username: str, db_path: str = DB_PATH) -> dict | None:
    """Return the user row as a dict, or None if not found."""
    try:
        with _get_connection(db_path) as conn:
            row = conn.execute(
                "SELECT user_id, username, password_hash, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error looking up user: {exc}") from exc


def get_user_by_id(user_id: str, db_path: str = DB_PATH) -> dict | None:
    """Return user row by user_id, or None."""
    try:
        with _get_connection(db_path) as conn:
            row = conn.execute(
                "SELECT user_id, username, created_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error looking up user by id: {exc}") from exc


# ── Learner Profiles ───────────────────────────────────────────────────────────

def insert_learner_profile(user_id: str, db_path: str = DB_PATH) -> None:
    """Create default learner profile for a newly registered user."""
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO learner_profiles
                   (user_id, estimated_level, overall_mastery, preferred_modality)
                   VALUES (?, 'Intermediate', 0.0, 'Not enough data')""",
                (user_id,),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not create learner profile: {exc}") from exc


def get_learner_profile(user_id: str, db_path: str = DB_PATH) -> dict | None:
    """Return learner profile dict for user_id, or None if absent."""
    try:
        with _get_connection(db_path) as conn:
            row = conn.execute(
                """SELECT user_id, estimated_level, overall_mastery,
                          preferred_modality, updated_at
                   FROM learner_profiles WHERE user_id = ?""",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error loading learner profile: {exc}") from exc


def update_learner_profile(user_id: str, estimated_level: str,
                           overall_mastery: float, preferred_modality: str,
                           db_path: str = DB_PATH) -> None:
    """Upsert learner profile for user_id."""
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO learner_profiles
                       (user_id, estimated_level, overall_mastery, preferred_modality, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(user_id) DO UPDATE SET
                       estimated_level    = excluded.estimated_level,
                       overall_mastery    = excluded.overall_mastery,
                       preferred_modality = excluded.preferred_modality,
                       updated_at         = excluded.updated_at""",
                (user_id, estimated_level, overall_mastery, preferred_modality),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not update learner profile: {exc}") from exc


# ── Concept Mastery ────────────────────────────────────────────────────────────

def upsert_concept_mastery(user_id: str, doc_id: str, concept: str,
                           mastery_score: float, db_path: str = DB_PATH) -> None:
    """Insert or update mastery score for a specific concept/document/user."""
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO concept_mastery
                       (user_id, doc_id, concept, mastery_score, last_updated)
                   VALUES (?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(user_id, doc_id, concept) DO UPDATE SET
                       mastery_score = excluded.mastery_score,
                       last_updated  = excluded.last_updated""",
                (user_id, doc_id, concept, mastery_score),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not save concept mastery: {exc}") from exc


def get_concept_mastery(user_id: str, doc_id: str,
                        db_path: str = DB_PATH) -> list:
    """
    Return all concept mastery rows for (user_id, doc_id).
    Scoped strictly to the given user — no cross-user access possible.
    """
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """SELECT concept, mastery_score, last_updated
                   FROM concept_mastery
                   WHERE user_id = ? AND doc_id = ?
                   ORDER BY mastery_score DESC""",
                (user_id, doc_id),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error loading concept mastery: {exc}") from exc


def get_all_doc_mastery_for_user(user_id: str, db_path: str = DB_PATH) -> list:
    """Return avg mastery per document for a user."""
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """SELECT doc_id, ROUND(AVG(mastery_score), 1) AS avg_mastery
                   FROM concept_mastery
                   WHERE user_id = ?
                   GROUP BY doc_id
                   ORDER BY avg_mastery DESC""",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error loading document mastery summary: {exc}") from exc


# ── Quiz Attempts ──────────────────────────────────────────────────────────────

def insert_quiz_attempt(user_id: str, doc_id: str, score: int,
                        total_questions: int, db_path: str = DB_PATH) -> None:
    """Record a completed quiz attempt."""
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO quiz_attempts (user_id, doc_id, score, total_questions)
                   VALUES (?, ?, ?, ?)""",
                (user_id, doc_id, score, total_questions),
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not save quiz attempt: {exc}") from exc


def get_quiz_history(user_id: str, doc_id: str,
                     db_path: str = DB_PATH) -> list:
    """Return quiz history for (user_id, doc_id), newest first. Scoped to user."""
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """SELECT score, total_questions, timestamp
                   FROM quiz_attempts
                   WHERE user_id = ? AND doc_id = ?
                   ORDER BY timestamp DESC""",
                (user_id, doc_id),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error loading quiz history: {exc}") from exc


def get_recent_quiz_score(user_id: str, db_path: str = DB_PATH):
    """Return the percentage score of the most recent quiz attempt, or None."""
    try:
        with _get_connection(db_path) as conn:
            row = conn.execute(
                """SELECT score, total_questions FROM quiz_attempts
                   WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (user_id,),
            ).fetchone()
        if row and row["total_questions"] > 0:
            return round(row["score"] / row["total_questions"] * 100, 1)
        return None
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error loading recent quiz score: {exc}") from exc


# ── Interaction History ────────────────────────────────────────────────────────

def log_interaction(user_id: str, interaction_type: str, doc_id: str = "",
                    db_path: str = DB_PATH) -> None:
    """
    Record a user interaction event (non-fatal — failures are logged but not raised).
    Types: 'pdf_upload', 'generate_content', 'quiz_submitted', 'retest', 'assistant_query'
    """
    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO interaction_history (user_id, interaction_type, doc_id)
                   VALUES (?, ?, ?)""",
                (user_id, interaction_type, doc_id),
            )
    except sqlite3.Error as exc:
        logger.warning("Could not log interaction (%s): %s", interaction_type, exc)
