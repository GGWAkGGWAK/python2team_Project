# CardFlow OCR - 카메라 기반 명함 인식 및 고객정보 관리

카메라 또는 이미지에서 명함을 읽고, 인식 결과를 확인·수정한 뒤 SQLite에 저장하는 Python/Flask 프로젝트입니다. 기획서의 흐름인 **카메라 입력 → 원근 보정 → OCR → 사용자 확인 → DB 저장·검색**을 한 화면에 구현했습니다.

처음 설치하거나 다른 PC에 배포할 때는 `설치_및_실행_가이드.txt`를 먼저 확인하세요.

## 주요 기능

- 브라우저 카메라 촬영 및 JPG/PNG/WEBP/BMP 업로드
- OpenCV 외곽선 검출, 기울기·원근 보정, 밝기·초점 안내
- PP-OCRv5 한국어·영어 딥러닝 인식과 EasyOCR 자동 보완
- 이름, 회사, 직책, 전화번호 1·2, 이메일, 웹사이트, 주소 자동 분류
- 인식 원문 확인 및 수정 후 재분류
- SQLite 고객 등록·검색·수정·삭제, 전화번호/이메일 중복 경고
- UTF-8 CSV 및 서식이 적용된 Excel 내보내기
- OpenCV `VideoCapture` 기반 별도 카메라 촬영 도구

## 권장 환경

- Windows 10/11
- Python 3.11 권장 (PaddlePaddle와 EasyOCR/PyTorch 공통 호환 환경)
- 노트북 내장 카메라 또는 USB 웹캠

## 설치와 실행

PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

브라우저에서 <http://127.0.0.1:5000>을 엽니다. 브라우저가 카메라 권한을 물으면 허용하세요. 카메라 권한 없이도 이미지 업로드로 사용할 수 있습니다.

> 최초 인식 시 PP-OCRv5 검출 모델과 한국어·영어 인식 모델을 사용자 폴더에 내려받기 때문에 인터넷 연결이 필요하고 첫 실행에 시간이 걸릴 수 있습니다. 이후에는 내려받은 모델을 재사용합니다. PP-OCRv5 결과가 부족하거나 실행되지 않으면 EasyOCR가 자동으로 보완합니다.

## OpenCV 카메라 창으로 촬영

브라우저 대신 Python의 `VideoCapture`를 직접 확인하려면 다음을 실행합니다.

```powershell
python capture_card.py --camera 0 --output captured_card.jpg
```

명함을 가이드 안에 맞추고 `Space` 또는 `Enter`를 누르면 원근 보정한 이미지가 저장됩니다. `Q` 또는 `Esc`는 취소입니다. 카메라가 여러 대면 `--camera 1`처럼 번호를 바꿉니다.

## 데이터 저장 위치

- 고객 DB: `instance/business_cards.db`
- 보정된 명함 이미지: `instance/scans/`

`instance/`는 Git에서 제외됩니다. 실제 고객 개인정보가 들어갈 수 있으므로 이 폴더를 공개 저장소에 올리지 마세요.

## 테스트

```powershell
pip install -r requirements-dev.txt
pytest -q
```

테스트는 명함 필드 분류, 전화번호/이메일 중복 검사, OpenCV 원근 보정, 고객 CRUD, CSV/Excel 출력을 확인합니다.

## 운영 실행

개발 서버 대신 Windows용 WSGI 서버인 Waitress를 사용할 수 있습니다.

```powershell
waitress-serve --listen=127.0.0.1:5000 app:app
```

외부 PC에 공개할 때는 기본 개발용 `SECRET_KEY`를 환경변수로 교체하고 HTTPS·접근 제어·DB 백업 정책을 추가하세요. 이 프로젝트는 기본적으로 한 대의 로컬 PC에서 사용하는 교육용 MVP입니다.

## 문제 해결

- **OCR 설치 필요**: 가상환경이 활성화된 상태에서 `python -m pip install -r requirements.txt`를 다시 실행합니다.
- **카메라가 열리지 않음**: Windows 설정의 카메라 개인정보 보호에서 브라우저 접근을 허용하고, Zoom 등 카메라를 점유한 프로그램을 종료합니다.
- **인식률이 낮음**: 명함을 평평하게 놓고 반사를 피하며, 글자가 선명하도록 화면을 가득 채워 촬영합니다. 결과는 저장 전에 직접 수정할 수 있습니다.
- **첫 OCR이 느림**: PP-OCRv5 모델 다운로드와 초기 메모리 로딩 때문입니다. 두 번째 요청부터 빨라집니다.
