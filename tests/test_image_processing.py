import cv2
import numpy as np

from cardocr.image_processing import (
    decode_image,
    detect_and_rectify,
    encode_jpeg,
    resize_for_ocr,
)


def test_detect_and_rectify_synthetic_card():
    image = np.zeros((700, 1000, 3), dtype=np.uint8)
    polygon = np.array([[180, 170], [850, 115], [890, 520], [145, 555]], dtype=np.int32)
    cv2.fillConvexPoly(image, polygon, (248, 248, 248))
    cv2.polylines(image, [polygon], True, (40, 40, 40), 8)
    cv2.putText(image, "CARD FLOW", (280, 330), cv2.FONT_HERSHEY_SIMPLEX, 2, (15, 15, 15), 4)

    result = detect_and_rectify(image)

    assert result.detected is True
    assert len(result.corners) == 4
    assert result.image.shape[1] > result.image.shape[0]


def test_jpeg_round_trip():
    source = np.full((240, 400, 3), 180, dtype=np.uint8)
    decoded = decode_image(encode_jpeg(source))
    assert decoded.shape == source.shape


def test_resize_for_ocr_caps_large_image_and_preserves_aspect_ratio():
    source = np.zeros((2400, 4000, 3), dtype=np.uint8)

    resized = resize_for_ocr(source, max_long_edge=2000)

    assert resized.shape == (1200, 2000, 3)
    assert resize_for_ocr(resized, max_long_edge=2000) is resized
