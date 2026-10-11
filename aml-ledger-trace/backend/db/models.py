"""
Database Schemas for Users, Cases, Notes, AuditLog, Feedback (Sprint 3).
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from backend.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="ANALYST")  # "ADMIN" or "ANALYST"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(50), index=True, nullable=False)
    title = Column(String(200), nullable=False)
    status = Column(String(30), default="OPEN")  # OPEN, IN_REVIEW, ESCALATED, CLEARED
    priority = Column(String(20), default="HIGH")  # LOW, MEDIUM, HIGH, CRITICAL
    assigned_to = Column(String(50), default="analyst")
    created_by = Column(String(50), default="system")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    notes = relationship("CaseNote", back_populates="case", cascade="all, delete-orphan")


class CaseNote(Base):
    __tablename__ = "case_notes"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    author_username = Column(String(50), nullable=False)
    note_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    case = relationship("Case", back_populates="notes")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    entity_type = Column(String(30), nullable=False)
    entity_id = Column(String(50), nullable=False)
    details = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String(50), index=True, nullable=False)
    username = Column(String(50), nullable=False)
    label = Column(String(50), nullable=False)  # CONFIRMED_SUSPICIOUS or NOT_SUSPICIOUS
    comments = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
