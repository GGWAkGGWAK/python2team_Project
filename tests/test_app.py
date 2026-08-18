import io

import cv2
import numpy as np

from cardocr import create_app
from cardocr.ocr_engine import OCRLine


def make_app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "api.db"),
            "SCAN_DIR": str(tmp_path / "scans"),
        }
    )


def test_home_health_and_contact_crud(tmp_path):
    client = make_app(tmp_path).test_client()

    home = client.get("/")
    assert home.status_code == 200
    assert b'id="image-upload"' in home.data
    assert b'id="upload-dropzone"' in home.data
    assert b"image/jpeg,image/png,image/webp,image/bmp" in home.data
    assert client.get("/api/health").get_json()["service"] == "cardocr"
    assert client.get("/assets/styles.css?v=frontend-v7").mimetype == "text/css"
    assert client.get("/assets/app.js?v=frontend-v7").mimetype == "text/javascript"

    response = client.post(
        "/api/contacts",
        json={
            "name": "이서준",
            "company": "오픈테크",
            "phone": "010-1111-2222",
            "email": "seo@example.com",
        },
    )
    assert response.status_code == 201
    contact = response.get_json()["data"]

    assert client.get("/api/contacts?q=오픈").get_json()["count"] == 1
    assert client.put(
        f"/api/contacts/{contact['id']}", json={**contact, "job_title": "연구원"}
    ).get_json()["data"]["job_title"] == "연구원"
    assert client.delete(f"/api/contacts/{contact['id']}").status_code == 200


def test_duplicate_response_and_exports(tmp_path):
    client = make_app(tmp_path).test_client()
    payload = {"name": "박지우", "phone": "010-9999-8888", "email": "jiwoo@example.com"}
    assert client.post("/api/contacts", json=payload).status_code == 201

    duplicate = client.post("/api/contacts", json={**payload, "name": "다른 이름"})
    assert duplicate.status_code == 409
    assert duplicate.get_json()["error"]["code"] == "DUPLICATE_CONTACT"

    csv_response = client.get("/api/export?format=csv")
    assert csv_response.status_code == 200
    assert "박지우" in csv_response.data.decode("utf-8-sig")

    xlsx_response = client.get("/api/export?format=xlsx")
    assert xlsx_response.status_code == 200
    assert xlsx_response.data.startswith(b"PK")


def test_delete_contact_removes_unreferenced_scan(tmp_path):
    app = make_app(tmp_path)
    token = "a" * 32 + ".jpg"
    scan = tmp_path / "scans" / token
    scan.write_bytes(b"test-image")
    client = app.test_client()
    response = client.post(
        "/api/contacts",
        json={"name": "이미지 고객", "image_token": token},
    )
    contact_id = response.get_json()["data"]["id"]

    assert scan.is_file()
    assert client.delete(f"/api/contacts/{contact_id}").status_code == 200
    assert not scan.exists()


def test_parse_endpoint(tmp_path):
    client = make_app(tmp_path).test_client()
    response = client.post(
        "/api/parse",
        json={"raw_text": "주식회사 테스트\n김하늘 대표\n010-3333-4444\nsky@example.com"},
    )
    fields = response.get_json()["data"]["fields"]
    assert fields["name"] == "김하늘"
    assert fields["phone"] == "010-3333-4444"


def test_ocr_pipeline_with_stubbed_engine(tmp_path, monkeypatch):
    from cardocr import routes

    monkeypatch.setattr(
        routes.engine,
        "recognize",
        lambda _image: [
            OCRLine("주식회사 테스트", 0.98, [[0, 0], [1, 0], [1, 1], [0, 1]]),
            OCRLine("김하늘 대표", 0.96, [[0, 2], [1, 2], [1, 3], [0, 3]]),
            OCRLine("010-3333-4444", 0.99, [[0, 4], [1, 4], [1, 5], [0, 5]]),
            OCRLine("sky@example.com", 0.97, [[0, 6], [1, 6], [1, 7], [0, 7]]),
        ],
    )
    image = np.full((420, 720, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (707, 407), (30, 30, 30), 5)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    app = make_app(tmp_path)
    response = app.test_client().post(
        "/api/ocr",
        data={"image": (io.BytesIO(encoded.tobytes()), "card.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["fields"]["name"] == "김하늘"
    assert data["fields"]["email"] == "sky@example.com"
    assert data["preview"].startswith("data:image/jpeg;base64,")
    assert data["processing_ms"]["total"] >= 0
    assert (tmp_path / "scans" / data["image_token"]).is_file()


def test_ocr_upload_rejects_unsupported_extension(tmp_path):
    client = make_app(tmp_path).test_client()
    response = client.post(
        "/api/ocr",
        data={"image": (io.BytesIO(b"not-an-image"), "card.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "JPG, PNG, WEBP, BMP" in response.get_json()["error"]["message"]
