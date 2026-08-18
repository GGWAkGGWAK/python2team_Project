# LayoutXLM 명함 필드 분류 모델 학습 매뉴얼

이 문서는 Windows 10/11에서 WSL2 Ubuntu를 이용해 CardFlow OCR의 LayoutXLM 필드 분류 모델을 학습하고 실행하는 방법을 설명합니다.

LayoutXLM은 OCR 엔진이 아닙니다. PP-OCRv5가 검출한 텍스트와 위치를 입력받아 이름, 회사, 직책, 주소 등의 필드로 분류합니다. 전화번호, 휴대전화, 팩스, 이메일, 웹사이트는 기존 규칙 기반 검증과 함께 사용됩니다.

## 1. 학습 데이터

학습 파일은 `training/` 폴더에 있습니다.

- `train.jsonl`: 학습 데이터
- `validation.jsonl`: 검증 데이터
- `images/`: 명함 이미지
- `train_layoutxlm.py`: 학습 스크립트
- `extract_synthetic_ocr.py`: OCR 텍스트·좌표 추출 스크립트
- `build_synthetic_dataset.py`: 검수 라벨을 JSONL로 생성하는 스크립트

JSONL 한 줄은 명함 한 장이며 다음 값을 포함합니다.

```json
{
  "image": "images/example.png",
  "words": ["회사명", "홍길동", "010-1234-5678"],
  "boxes": [[50, 60, 300, 120], [400, 250, 550, 320], [400, 700, 650, 750]],
  "labels": ["B-COMPANY", "B-NAME", "B-MOBILE"]
}
```

`boxes`는 원본 픽셀이 아니라 `0~1000` 범위로 정규화된 `[x0, y0, x1, y1]` 좌표입니다. 지원 라벨은 `NAME`, `COMPANY`, `POSITION`, `ADDRESS`, `TELEPHONE`, `MOBILE`, `FAX`, `EMAIL`, `WEBSITE`, `O`의 BIO 형식입니다.

현재 제공된 합성 데이터 6장은 학습 절차 확인용입니다. 실제 일반화 성능을 높이려면 서로 다른 디자인, 글꼴, 배치, 촬영 환경을 가진 명함을 수십~수백 장 추가해야 합니다.

## 2. WSL2와 Ubuntu 설치

관리자 권한 Windows PowerShell에서 실행합니다.

```powershell
wsl --install
wsl --update
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

설치 후 Windows를 재부팅하고 Ubuntu를 실행해 Linux 사용자 이름과 비밀번호를 만듭니다. 설치 확인 명령은 Windows PowerShell에서 실행합니다.

```powershell
wsl --list --verbose
```

정상 예시:

```text
NAME            STATE      VERSION
Ubuntu-22.04    Running    2
```

### 오류 0x80370114

관리자 PowerShell에서 필요한 Windows 기능을 활성화한 뒤 재부팅합니다.

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
bcdedit.exe /set hypervisorlaunchtype auto
wsl.exe --update
```

같은 오류가 반복되면 작업 관리자의 `성능 → CPU → 가상화`가 `사용`인지 확인합니다. 비활성화 상태라면 BIOS/UEFI에서 Intel VT-x 또는 AMD SVM을 활성화해야 합니다.

## 3. Windows 프로젝트를 Ubuntu로 복사

이 절부터는 Ubuntu 터미널에서 실행합니다. 예시 Windows 프로젝트 경로는 다음과 같습니다.

```text
C:\Users\dltlg\Desktop\SW파일럿\명함 OCR 프로그램\python2team_Project
```

WSL에서는 다음 경로로 접근합니다.

```text
/mnt/c/Users/dltlg/Desktop/SW파일럿/명함 OCR 프로그램/python2team_Project
```

필수 도구를 설치합니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git git-lfs build-essential ninja-build rsync
```

Windows 가상환경은 Linux에서 사용할 수 없으므로 제외하고 복사합니다.

```bash
mkdir -p ~/projects/python2team_Project
rsync -a \
  --exclude=".venv" \
  --exclude=".venv-wsl" \
  --exclude="__pycache__" \
  "/mnt/c/Users/dltlg/Desktop/SW파일럿/명함 OCR 프로그램/python2team_Project/" \
  ~/projects/python2team_Project/
cd ~/projects/python2team_Project
```

## 4. Ubuntu 학습 환경 구성

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

NVIDIA GPU가 있다면 먼저 확인합니다.

```bash
nvidia-smi
```

이 프로젝트에서 시험한 RTX 4060 및 CUDA 12.x 환경은 다음 PyTorch 패키지를 사용했습니다.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

GPU가 없다면 CPU 패키지를 설치합니다.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

설치를 확인합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

AI 의존성과 Detectron2를 설치합니다.

```bash
python -m pip install -r requirements-ai.txt
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
```

Detectron2 설치 중 `No module named 'torch'`가 나타나면 일반 설치 명령을 반복하지 말고 반드시 `--no-build-isolation`을 사용합니다.

```bash
python -c "import torch, detectron2; print('Detectron2 정상'); print('GPU:', torch.cuda.is_available())"
```

## 5. LayoutXLM 학습

8GB GPU에서는 배치 크기 1을 사용합니다.

```bash
python training/train_layoutxlm.py \
  --train training/train.jsonl \
  --validation training/validation.jsonl \
  --output models/business-card-layoutxlm \
  --epochs 8 \
  --batch-size 1
```

학습이 시작되면 `loss`, `eval_loss`, `epoch`과 진행률이 출력됩니다. 다음 문구가 나와야 완료된 것입니다.

```text
학습 모델 저장 완료: .../models/business-card-layoutxlm
```

기본 모델의 `classifier.weight`와 `classifier.bias`가 새로 초기화됐다는 안내는 정상입니다. 명함 필드 분류용 출력 계층을 이번 데이터로 새로 학습한다는 의미입니다.

최종 모델을 확인합니다.

```bash
ls -lh models/business-card-layoutxlm
```

실행에 필요한 주요 파일:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `sentencepiece.bpe.model`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`

`checkpoint-*` 폴더는 중간 학습 상태이며 앱 실행에는 필요하지 않습니다.

## 6. 최종 모델을 Windows 프로젝트로 복사

Ubuntu에서 실행합니다. 중간 체크포인트를 제외하지 않으면 수십 GB를 복사할 수 있으므로 `--exclude`를 유지합니다.

```bash
mkdir -p "/mnt/c/Users/dltlg/Desktop/SW파일럿/명함 OCR 프로그램/python2team_Project/models/business-card-layoutxlm"
rsync -avh --progress \
  --exclude="checkpoint-*" \
  models/business-card-layoutxlm/ \
  "/mnt/c/Users/dltlg/Desktop/SW파일럿/명함 OCR 프로그램/python2team_Project/models/business-card-layoutxlm/"
```

`rsync -a`는 성공해도 아무 메시지를 출력하지 않을 수 있습니다. 진행 상황이 필요하면 위 예시처럼 `-avh --progress`를 사용합니다. 동일한 복사를 여러 번 실행했다면 `pgrep -a rsync`로 확인하고 불필요한 프로세스를 종료한 뒤 한 번만 다시 실행합니다.

## 7. WSL에서 애플리케이션 실행

LayoutXLM은 추론 시에도 Detectron2가 필요하므로 네이티브 Windows보다 WSL에서 앱을 실행하는 것이 안전합니다.

```bash
cd ~/projects/python2team_Project
source .venv/bin/activate
python -m pip install -r requirements.txt
export CARDOCR_LAYOUT_MODEL_DIR="$PWD/models/business-card-layoutxlm"
python app.py
```

Ubuntu 터미널을 닫지 않고 Windows 브라우저에서 접속합니다.

- 앱: <http://127.0.0.1:5000>
- 상태 확인: <http://127.0.0.1:5000/api/health>

모델 경로를 찾은 상태:

```json
{
  "configured": true,
  "mode": "layoutxlm-hybrid",
  "ready": false,
  "error": ""
}
```

명함 OCR을 한 번 실행해 모델까지 로딩한 상태:

```json
{
  "configured": true,
  "mode": "layoutxlm-hybrid",
  "ready": true,
  "error": ""
}
```

`ready: false`는 첫 OCR 전에는 정상입니다. OCR 후에도 `false`이면 Ubuntu 터미널과 `field_classifier.error`를 확인합니다.

## 8. 성능 해석

현재 앱은 혼합 분류 방식입니다.

- 이름·회사·직책·주소: LayoutXLM이 규칙 기반 결과를 보완
- 전화·휴대전화·팩스·이메일·웹사이트: 정규식과 표기 규칙으로 최종 검증
- 기존 값이 있으면 AI 신뢰도가 충분히 높을 때만 교체

모델이 실행됐는지는 `/api/ocr` 응답의 `field_classifier.used`와 `predictions`로 확인합니다. 학습 데이터에 포함된 명함이 아니라 새로운 명함으로 검증해야 실제 일반화 성능을 평가할 수 있습니다.

