# LayoutXLM 명함 필드 분류 학습

현재 애플리케이션은 학습 모델이 없어도 기존 규칙 기반 분류로 정상 동작합니다. LayoutXLM을 사용하려면 명함 이미지와 OCR 텍스트·좌표·정답 라벨을 준비해 별도로 미세조정해야 합니다.

## 데이터 형식

JSONL 한 줄이 명함 한 장입니다.

- `image`: JSONL 파일 기준 이미지 상대 경로
- `words`: OCR 텍스트 목록
- `boxes`: 각 텍스트의 `[x0, y0, x1, y1]` 좌표(0~1000 정규화)
- `labels`: 각 텍스트의 BIO 라벨

지원 라벨은 `NAME`, `COMPANY`, `POSITION`, `ADDRESS`, `TELEPHONE`, `MOBILE`, `FAX`, `EMAIL`, `WEBSITE`와 `O`입니다. 실제 개인정보가 포함된 학습 이미지와 JSONL 파일은 공개 저장소에 커밋하지 마세요.

## 학습

LayoutXLM은 Windows CPU에서 학습하기 무겁기 때문에 CUDA GPU가 있는 Linux 또는 Google Colab 환경을 권장합니다.

```powershell
pip install -r requirements-ai.txt
python training/train_layoutxlm.py --train training/train.jsonl --validation training/validation.jsonl --output models/business-card-layoutxlm
```

## 앱 연결

학습 결과 폴더를 프로젝트에 복사한 뒤 환경변수를 지정합니다.

```powershell
$env:CARDOCR_LAYOUT_MODEL_DIR = "$PWD\models\business-card-layoutxlm"
python app.py
```

`GET /api/health`의 `field_classifier.mode`가 `layoutxlm-hybrid`이고 `configured`가 `true`이면 모델 경로를 찾은 상태입니다. 첫 명함 인식 때 모델이 메모리에 로딩됩니다. AI 실행에 실패하면 오류를 기록하고 기존 규칙 기반 결과를 자동으로 사용합니다.

연락처 형식은 정규식이 안정적이므로 AI는 이름·회사·직책·주소만 보완하거나 높은 신뢰도에서 교체합니다. 전화·휴대전화·팩스·이메일·웹사이트는 기존 규칙이 최종 검증합니다.
