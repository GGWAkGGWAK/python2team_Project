# CardFlow OCR - 카메라 기반 명함 인식 및 고객정보 관리

카메라 또는 이미지에서 명함을 읽고, 인식 결과를 확인·수정한 뒤 SQLite에 저장하는 Python/Flask 프로젝트입니다. 기획서의 흐름인 **카메라 입력 → 원근 보정 → OCR → 사용자 확인 → DB 저장·검색**을 한 화면에 구현했습니다.

처음 설치하거나 다른 PC에 배포할 때는 `설치_및_실행_가이드.txt`를 먼저 확인하세요.

## 추가 매뉴얼

- [WSL2 기반 LayoutXLM 학습 매뉴얼](docs/LAYOUTXLM_WSL_TRAINING_MANUAL.md)
- [Git LFS 모델 업로드·다운로드 매뉴얼](docs/GIT_LFS_MODEL_MANUAL.md)

LayoutXLM 학습은 Detectron2 호환성을 위해 WSL2 Ubuntu 환경을 권장합니다. 학습된 `model.safetensors`는 약 1.4GB이므로 GitHub에 올릴 때 일반 Git이 아니라 Git LFS를 사용해야 합니다.

## 주요 기능

- 브라우저 카메라 촬영 및 JPG/PNG/WEBP/BMP 업로드(파일 선택·드래그앤드롭·즉시 미리보기)
- OpenCV 외곽선 검출, 기울기·원근 보정, 밝기·초점 안내
- 경량 PP-OCRv5 mobile 한국어·영어 딥러닝 인식과 EasyOCR 자동 보완
- 선택형 LayoutXLM 문서 AI로 이름·회사·직책·주소 필드 보완(학습 모델 필요)
- 앱 시작 후 OCR 모델 백그라운드 준비 및 고해상도 입력 자동 최적화
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

> 앱 실행 직후 경량 PP-OCRv5 mobile 모델을 백그라운드에서 준비합니다. 처음 한 번은 모델을 사용자 폴더에 내려받기 때문에 인터넷 연결이 필요하며 상단 상태가 `OCR 준비됨`으로 바뀐 뒤 인식하면 대기시간이 줄어듭니다. 이후에는 내려받은 모델을 재사용합니다. PP-OCRv5 결과가 부족하거나 실행되지 않으면 EasyOCR가 자동으로 보완합니다.

## 선택형 LayoutXLM 필드 분류

기본 설치에서는 기존 규칙 기반 분류가 사용됩니다. 명함 데이터로 미세조정한 LayoutXLM 모델이 있으면 OCR 텍스트와 위치를 함께 분석해 이름·회사·직책·주소를 보완하고, 전화·팩스·이메일은 기존 정규식으로 검증합니다.

```powershell
pip install -r requirements-ai.txt
$env:CARDOCR_LAYOUT_MODEL_DIR = "$PWD\models\business-card-layoutxlm"
python app.py
```

학습 데이터 형식과 학습 명령은 `training/README.md`를 확인하세요. 학습 모델이 없거나 실행에 실패해도 애플리케이션은 규칙 기반 방식으로 계속 동작합니다.

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
- **첫 OCR이 느림**: 상단의 `OCR 모델 준비 중`이 `OCR 준비됨`으로 바뀔 때까지 기다린 뒤 업로드합니다. 터미널의 `OCR 완료` 로그에서 단계별 처리시간을 확인할 수 있습니다.
