"""
Security, Authentication, and Account Privacy Masking (Sprint 6).
"""

from __future__ import annotations

import datetime
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from passlib.context import CryptContext

SECRET_KEY = "aml-ledger-trace-jwt-secret-key-production-change-in-env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12

import bcrypt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    # bcrypt max length is 72 bytes
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def mask_account_id(account_id: str, is_admin: bool = False) -> str:
    """
    Privacy masking for account identifiers:
    If not admin: ACC_00123 -> ACC_***23
    """
    if is_admin:
        return account_id
    if len(account_id) <= 6:
        return account_id[:2] + "****"
    prefix = account_id[:4]
    suffix = account_id[-2:]
    return f"{prefix}****{suffix}"
