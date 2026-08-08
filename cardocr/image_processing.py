"""OpenCV-based card detection, perspective correction and OCR preprocessing."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


@dataclass(slots=True)
class CardImage:
    image: np.ndarray
    detected: bool
    corners: list[list[int]]
    quality_warnings: list[str]


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise InvalidImageError("이미지 데이터가 비어 있습니다.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("지원하지 않거나 손상된 이미지입니다.")
    if image.shape[0] < 120 or image.shape[1] < 180:
        raise InvalidImageError("이미지가 너무 작습니다. 최소 180×120 픽셀이 필요합니다.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise InvalidImageError("이미지를 JPEG로 변환하지 못했습니다.")
    return encoded.tobytes()


def as_data_url(image: np.ndarray) -> str:
    payload = base64.b64encode(encode_jpeg(image)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _warp(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = _order_points(points)
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    width = max(width, 640)
    height = max(height, 360)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left]), destination
    )
    warped = cv2.warpPerspective(image, matrix, (width, height))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _quality_warnings(image: np.ndarray) -> list[str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    warnings: list[str] = []
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < 65:
        warnings.append("이미지가 어둡습니다. 조명을 밝게 해 주세요.")
    elif brightness > 225:
        warnings.append("빛 반사가 강합니다. 명함의 각도를 조정해 주세요.")
    if sharpness < 45:
        warnings.append("초점이 흐립니다. 카메라를 고정한 뒤 다시 촬영해 주세요.")
    return warnings


def detect_and_rectify(image: np.ndarray) -> CardImage:
    original = image.copy()
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = resized.shape[0] * resized.shape[1]
    candidate: np.ndarray | None = None
    best_score = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        rect = cv2.minAreaRect(polygon)
        rect_width, rect_height = rect[1]
        if min(rect_width, rect_height) <= 0:
            continue
        aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.7) / 1.7)
        score = area / image_area + aspect_score * 0.25
        if score > best_score:
            best_score = score
            candidate = polygon.reshape(4, 2).astype("float32") / scale

    if candidate is None:
        return CardImage(
            image=original,
            detected=False,
            corners=[],
            quality_warnings=[
                "명함 외곽선을 찾지 못해 전체 이미지로 OCR을 진행했습니다."
            ]
            + _quality_warnings(original),
        )

    corrected = _warp(original, candidate)
    return CardImage(
        image=corrected,
        detected=True,
        corners=np.rint(_order_points(candidate)).astype(int).tolist(),
        quality_warnings=_quality_warnings(corrected),
    )


def prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    if image.shape[1] < 1200:
        ratio = 1200 / image.shape[1]
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def draw_camera_guide(frame: np.ndarray) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]
    guide_width = int(width * 0.72)
    guide_height = int(guide_width / 1.72)
    guide_height = min(guide_height, int(height * 0.65))
    guide_width = int(guide_height * 1.72)
    x1 = (width - guide_width) // 2
    y1 = (height - guide_height) // 2
    x2, y2 = x1 + guide_width, y1 + guide_height
    color = (64, 220, 160)
    length = max(24, int(min(width, height) * 0.06))
    for x, y, sx, sy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x2, y2, -1, -1),
        (x1, y2, 1, -1),
    ):
        cv2.line(display, (x, y), (x + sx * length, y), color, 3)
        cv2.line(display, (x, y), (x, y + sy * length), color, 3)
    cv2.putText(
        display,
        "Place the business card inside the guide",
        (max(16, x1), max(32, y1 - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def image_metadata(image: np.ndarray) -> dict[str, Any]:
    return {"width": int(image.shape[1]), "height": int(image.shape[0])}
