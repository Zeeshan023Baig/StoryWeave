"""
services/auth_service.py
========================
Authentication service for StoryWeave.

Responsibilities:
- Register new users with hashed passwords.
- Authenticate existing users (login).
- Never store or log plain-text passwords.
- Provide the unique user_id used throughout the system.

Uses bcrypt for password hashing (strong, salted, future-proof).
"""

import uuid
import bcrypt
from database.database import (
    insert_user,
    insert_learner_profile,
    get_user_by_username,
)


def _hash_password(plain_password: str) -> str:
    """Hash a plain-text password using bcrypt. Returns the hash as a UTF-8 string."""
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def _verify_password(plain_password: str, stored_hash: str) -> bool:
    """Return True if plain_password matches stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            stored_hash.encode("utf-8"),
        )
    except Exception:
        return False


def register_user(username: str, password: str,
                  db_path: str | None = None) -> dict:
    """
    Register a new user.

    Validates:
    - username is non-empty
    - password is non-empty
    - username is unique (DB constraint)

    Creates:
    - User row with hashed password
    - Default learner profile (Intermediate, 0% mastery)

    Returns: dict with user_id and username.
    Raises:  ValueError for validation failures, RuntimeError for DB errors.
    """
    username = (username or "").strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")

    password_hash = _hash_password(password)
    user_id = str(uuid.uuid4())

    kwargs = {"db_path": db_path} if db_path else {}
    insert_user(user_id, username, password_hash, **kwargs)
    insert_learner_profile(user_id, **kwargs)

    return {"user_id": user_id, "username": username}


def login_user(username: str, password: str,
               db_path: str | None = None) -> dict:
    """
    Authenticate a user.

    Returns: dict with user_id and username on success.
    Raises:  ValueError with a safe, generic message on failure.
             (Never reveals whether username vs password was wrong.)
    """
    username = (username or "").strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    kwargs = {"db_path": db_path} if db_path else {}
    user = get_user_by_username(username, **kwargs)

    if user is None or not _verify_password(password, user["password_hash"]):
        raise ValueError("Incorrect username or password.")

    return {"user_id": user["user_id"], "username": user["username"]}
