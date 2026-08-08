"""Optional OpenCV VideoCapture workflow for a directly attached webcam."""

from __future__ import annotations

from pathlib import Path

import cv2

from .image_processing import detect_and_rectify, draw_camera_guide, encode_jpeg


def capture_card(camera_index: int = 0, output: str | Path = "captured_card.jpg") -> Path | None:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"카메라 {camera_index}번을 열 수 없습니다.")

    destination = Path(output)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다.")
            cv2.imshow("Business Card Capture - SPACE: capture / Q: quit", draw_camera_guide(frame))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key in (ord(" "), 13):
                card = detect_and_rectify(frame)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encode_jpeg(card.image, quality=94))
                return destination
    finally:
        camera.release()
        cv2.destroyAllWindows()
