from cardocr.database import Database


def test_contact_crud_and_search(tmp_path):
    database = Database(tmp_path / "contacts.db")
    database.initialize()
    created = database.create_contact(
        {
            "name": "홍길동",
            "company": "카드플로우",
            "phone": "010-1234-5678",
            "phone2": "053-123-4567",
            "email": "hong@example.com",
        }
    )

    assert created["id"] > 0
    assert database.count_contacts() == 1
    assert database.list_contacts("카드")[0]["name"] == "홍길동"
    assert database.list_contacts("1234")[0]["company"] == "카드플로우"
    assert database.list_contacts("4567")[0]["phone2"] == "053-123-4567"

    updated = database.update_contact(created["id"], {**created, "job_title": "팀장"})
    assert updated["job_title"] == "팀장"
    assert database.delete_contact(created["id"])
    assert database.get_contact(created["id"]) is None


def test_duplicate_detection_normalizes_phone_and_email(tmp_path):
    database = Database(tmp_path / "contacts.db")
    database.initialize()
    first = database.create_contact(
        {"name": "김민수", "phone": "010-2222-3333", "email": "MIN@EXAMPLE.COM"}
    )

    assert database.find_duplicates(phone="010 2222 3333")[0]["id"] == first["id"]
    assert database.find_duplicates(email="min@example.com")[0]["id"] == first["id"]
    assert database.find_duplicates(phone="01022223333", exclude_id=first["id"]) == []


def test_existing_database_is_migrated_with_second_phone(tmp_path):
    path = tmp_path / "legacy.db"
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE contacts (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
                company TEXT NOT NULL DEFAULT '', job_title TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '', email TEXT NOT NULL DEFAULT '',
                website TEXT NOT NULL DEFAULT '', address TEXT NOT NULL DEFAULT '',
                memo TEXT NOT NULL DEFAULT '', raw_text TEXT NOT NULL DEFAULT '',
                image_token TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )"""
        )

    database = Database(path)
    database.initialize()
    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(contacts)")}
    assert "phone2" in columns
