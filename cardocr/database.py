"""SQLite persistence for recognized business cards."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


CONTACT_FIELDS = (
    "name",
    "company",
    "job_title",
    "phone",
    "phone2",
    "email",
    "website",
    "address",
    "memo",
    "raw_text",
    "image_token",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    company TEXT NOT NULL DEFAULT '',
    job_title TEXT NOT NULL DEFAULT '',
    phone TEXT NOT NULL DEFAULT '',
    phone2 TEXT NOT NULL DEFAULT '',
    email TEXT NOT NULL DEFAULT '',
    website TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT '',
    memo TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL DEFAULT '',
    image_token TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_contacts_name ON contacts(name);
CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company);
CREATE INDEX IF NOT EXISTS idx_contacts_phone ON contacts(phone);
CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(contacts)")
            }
            if "phone2" not in columns:
                connection.execute(
                    "ALTER TABLE contacts ADD COLUMN phone2 TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_contacts_phone2 ON contacts(phone2)"
            )

    @staticmethod
    def _clean(data: dict[str, Any]) -> dict[str, str]:
        return {
            field: str(data.get(field, "") or "").strip()
            for field in CONTACT_FIELDS
        }

    @staticmethod
    def _to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def create_contact(self, data: dict[str, Any]) -> dict[str, Any]:
        values = self._clean(data)
        columns = ", ".join(CONTACT_FIELDS)
        placeholders = ", ".join("?" for _ in CONTACT_FIELDS)
        with self.connect() as connection:
            cursor = connection.execute(
                f"INSERT INTO contacts ({columns}) VALUES ({placeholders})",
                tuple(values[field] for field in CONTACT_FIELDS),
            )
            contact_id = int(cursor.lastrowid)
        contact = self.get_contact(contact_id)
        assert contact is not None
        return contact

    def get_contact(self, contact_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM contacts WHERE id = ?", (contact_id,)
            ).fetchone()
        return self._to_dict(row)

    def list_contacts(self, query: str = "") -> list[dict[str, Any]]:
        query = query.strip()
        sql = "SELECT * FROM contacts"
        params: Iterable[str] = ()
        if query:
            needle = f"%{query}%"
            sql += (
                " WHERE name LIKE ? OR company LIKE ? OR phone LIKE ? "
                "OR phone2 LIKE ? OR email LIKE ?"
            )
            params = (needle, needle, needle, needle, needle)
        sql += " ORDER BY updated_at DESC, id DESC"
        with self.connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def update_contact(self, contact_id: int, data: dict[str, Any]) -> dict[str, Any] | None:
        if self.get_contact(contact_id) is None:
            return None
        values = self._clean(data)
        assignments = ", ".join(f"{field} = ?" for field in CONTACT_FIELDS)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE contacts SET {assignments}, "
                "updated_at = datetime('now', 'localtime') WHERE id = ?",
                (*[values[field] for field in CONTACT_FIELDS], contact_id),
            )
        return self.get_contact(contact_id)

    def delete_contact(self, contact_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        return cursor.rowcount > 0

    def find_duplicates(
        self,
        phone: str = "",
        email: str = "",
        exclude_id: int | None = None,
        phone2: str = "",
    ) -> list[dict[str, Any]]:
        phone = "".join(character for character in phone if character.isdigit())
        phone2 = "".join(character for character in phone2 if character.isdigit())
        email = email.strip().lower()
        clauses: list[str] = []
        params: list[Any] = []
        if phone:
            clauses.append(
                "(replace(replace(replace(replace(replace(phone, '-', ''), ' ', ''), '.', ''), '(', ''), ')', '') = ? "
                "OR replace(replace(replace(replace(replace(phone2, '-', ''), ' ', ''), '.', ''), '(', ''), ')', '') = ?)"
            )
            params.extend((phone, phone))
        if phone2:
            clauses.append(
                "(replace(replace(replace(replace(replace(phone, '-', ''), ' ', ''), '.', ''), '(', ''), ')', '') = ? "
                "OR replace(replace(replace(replace(replace(phone2, '-', ''), ' ', ''), '.', ''), '(', ''), ')', '') = ?)"
            )
            params.extend((phone2, phone2))
        if email:
            clauses.append("lower(email) = ?")
            params.append(email)
        if not clauses:
            return []
        sql = "SELECT * FROM contacts WHERE (" + " OR ".join(clauses) + ")"
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count_contacts(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM contacts").fetchone()
        return int(row["count"])

    def image_reference_count(self, image_token: str) -> int:
        if not image_token:
            return 0
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM contacts WHERE image_token = ?",
                (image_token,),
            ).fetchone()
        return int(row["count"])
