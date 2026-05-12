"""
SERS Clinical Webapp - Database Layer (SQLite)
환자 세션, 예측 결과, 감사 추적을 위한 데이터베이스
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).parent / "clinical_data.db"


def get_db_path() -> Path:
    return DEFAULT_DB_PATH


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """SQLite connection with WAL mode for concurrent access."""
    conn = sqlite3.connect(str(db_path or get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None):
    """Create all tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('technician', 'clinician', 'admin')),
    display_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    age REAL NOT NULL,
    sex TEXT NOT NULL CHECK(sex IN ('M', 'F')),
    bmi REAL,
    -- Legacy compatibility column. New UI treats this as the model decision
    -- profile and does not expose per-patient operating-mode selection.
    operating_mode TEXT NOT NULL CHECK(operating_mode IN ('screening', 'balanced', 'confirmatory')),
    model_variant TEXT,
    threshold REAL,
    status TEXT DEFAULT 'created' CHECK(status IN ('created', 'uploaded', 'qc_done', 'analyzed', 'reported')),
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS spectra (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    filename TEXT NOT NULL,
    qc_pass BOOLEAN,
    qc_flags TEXT,
    replicate_correlation REAL,
    fp_mean REAL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL REFERENCES sessions(id),
    cancer_detected BOOLEAN NOT NULL,
    screening_index REAL NOT NULL,
    majority_vote TEXT,
    cancer_type_prediction TEXT,
    cancer_type_confidence REAL,
    cancer_type_probabilities TEXT,
    per_replicate_results TEXT,
    result_json TEXT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    generated_by INTEGER REFERENCES users(id),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    session_id TEXT,
    action TEXT NOT NULL,
    detail TEXT
);
"""


# --- User CRUD ---

def create_user(username: str, password_hash: str, role: str, display_name: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role, display_name) VALUES (?, ?, ?, ?)",
            (username, password_hash, role, display_name),
        )
        return cur.lastrowid


def get_user_by_username(username: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# --- Session CRUD ---

def create_session(
    patient_id: str,
    age: float,
    sex: str,
    bmi: Optional[float],
    operating_mode: str,
    created_by: int,
    threshold: Optional[float] = None,
) -> str:
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO sessions (id, patient_id, age, sex, bmi, operating_mode, threshold, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, patient_id, age, sex, bmi, operating_mode, threshold, created_by),
        )
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def update_session_status(session_id: str, status: str, **kwargs):
    with get_connection() as conn:
        sets = ["status = ?"]
        vals = [status]
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        if status in ("analyzed", "reported"):
            sets.append("completed_at = ?")
            vals.append(datetime.now().isoformat())
        vals.append(session_id)
        conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", vals)


# --- Spectra CRUD ---

def add_spectrum(session_id: str, filename: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO spectra (session_id, filename) VALUES (?, ?)",
            (session_id, filename),
        )
        return cur.lastrowid


def update_spectrum_qc(spectrum_id: int, qc_pass: bool, qc_flags: list, correlation: Optional[float] = None, fp_mean: Optional[float] = None):
    with get_connection() as conn:
        conn.execute(
            "UPDATE spectra SET qc_pass = ?, qc_flags = ?, replicate_correlation = ?, fp_mean = ? WHERE id = ?",
            (qc_pass, json.dumps(qc_flags, ensure_ascii=False), correlation, fp_mean, spectrum_id),
        )


def get_spectra_for_session(session_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM spectra WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
        return [dict(r) for r in rows]


# --- Prediction CRUD ---

def save_prediction(session_id: str, result: dict):
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO predictions
               (session_id, cancer_detected, screening_index, majority_vote,
                cancer_type_prediction, cancer_type_confidence,
                cancer_type_probabilities, per_replicate_results, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                result.get("cancer_detected", False),
                result.get("screening_index", result.get("cancer_signal_score", 0.0)),
                result.get("majority_vote"),
                result.get("cancer_type_prediction"),
                result.get("cancer_type_confidence"),
                json.dumps(result.get("cancer_type_probabilities", {})),
                json.dumps(result.get("per_replicate", []), ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            ),
        )


def get_prediction(session_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            d = dict(row)
            d["cancer_type_probabilities"] = json.loads(d["cancer_type_probabilities"] or "{}")
            d["per_replicate_results"] = json.loads(d["per_replicate_results"] or "[]")
            d["result_json"] = json.loads(d["result_json"] or "{}")
            return d
        return None


# --- Report CRUD ---

def create_report(session_id: str, generated_by: int) -> str:
    report_id = str(uuid.uuid4())[:8].upper()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO reports (id, session_id, generated_by) VALUES (?, ?, ?)",
            (report_id, session_id, generated_by),
        )
    return report_id


def get_report(report_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return dict(row) if row else None


def get_reports_for_session(session_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM reports WHERE session_id = ? ORDER BY generated_at DESC", (session_id,)).fetchall()
        return [dict(r) for r in rows]


# --- Audit Log ---

def log_audit(action: str, user_id: Optional[int] = None, session_id: Optional[str] = None, detail: Optional[dict] = None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_log (user_id, session_id, action, detail) VALUES (?, ?, ?, ?)",
            (user_id, session_id, action, json.dumps(detail, ensure_ascii=False) if detail else None),
        )


def get_audit_logs(limit: int = 100, offset: int = 0) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT a.*, u.display_name as user_name
               FROM audit_log a LEFT JOIN users u ON a.user_id = u.id
               ORDER BY a.timestamp DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Init default admin user ---

def ensure_default_users():
    """Create default users if none exist."""
    import hashlib
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            # Default admin (password: admin123 — must be changed)
            pw_hash = hashlib.sha256("admin123".encode()).hexdigest()
            conn.execute(
                "INSERT INTO users (username, password_hash, role, display_name) VALUES (?, ?, ?, ?)",
                ("admin", pw_hash, "admin", "관리자"),
            )
            conn.execute(
                "INSERT INTO users (username, password_hash, role, display_name) VALUES (?, ?, ?, ?)",
                ("tech1", pw_hash, "technician", "검사실 기사"),
            )
            conn.execute(
                "INSERT INTO users (username, password_hash, role, display_name) VALUES (?, ?, ?, ?)",
                ("doctor1", pw_hash, "clinician", "담당 의사"),
            )
