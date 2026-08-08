"""Capture and rectify one business card with an OpenCV camera window."""

from __future__ import annotations

import argparse

from cardocr.camera import capture_card


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV 명함 촬영 도구")
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 번호")
    parser.add_argument("--output", default="captured_card.jpg", help="저장할 이미지 경로")
    args = parser.parse_args()
    result = capture_card(args.camera, args.output)
    print(f"저장 완료: {result}" if result else "촬영을 취소했습니다.")


if __name__ == "__main__":
    main()
