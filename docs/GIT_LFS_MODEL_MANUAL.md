# Git LFS 모델 업로드·다운로드 매뉴얼

이 문서는 약 1.4GB인 LayoutXLM 학습 모델을 GitHub에 올리고 다른 PC에서 내려받는 방법을 설명합니다.

GitHub 일반 Git 저장소는 100MB를 초과하는 파일을 차단합니다. `model.safetensors`는 Git LFS로 추적해야 합니다. GitHub Free/Pro의 LFS 단일 파일 제한은 현재 2GB이므로 이 프로젝트의 최종 모델 한 개는 업로드할 수 있습니다. 요금제와 저장공간·대역폭 한도는 변경될 수 있으므로 GitHub 공식 문서를 함께 확인하세요.

- Git LFS 안내: <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage>
- Git LFS 사용량: <https://docs.github.com/en/billing/concepts/product-billing/git-lfs>

## 1. 업로드 대상

업로드할 최종 모델 폴더:

```text
models/business-card-layoutxlm/
```

필수 파일:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `sentencepiece.bpe.model`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`

업로드하지 않을 파일:

- `checkpoint-*` 중간 학습 폴더
- `.venv/`, `.venv-wsl/`
- `instance/` 고객 DB와 실제 명함 스캔 이미지
- Hugging Face 사용자 캐시

`.gitignore`에 다음 항목이 있는지 확인합니다.

```gitignore
models/business-card-layoutxlm/checkpoint-*/
.venv/
.venv-wsl/
instance/
```

## 2. Git LFS 설정과 모델 업로드

Windows PowerShell에서 프로젝트로 이동합니다.

```powershell
cd "C:\Users\dltlg\Desktop\SW파일럿\명함 OCR 프로그램\python2team_Project"
```

GitHub Desktop을 설치했다면 Git LFS 실행 파일도 일반적으로 함께 설치됩니다. 초기화하고 최종 가중치 파일만 LFS로 추적합니다.

```powershell
git lfs install
git lfs track "models/business-card-layoutxlm/model.safetensors"
```

위 명령은 `.gitattributes`를 생성하거나 수정합니다. 다음 내용이 있어야 합니다.

```text
models/business-card-layoutxlm/model.safetensors filter=lfs diff=lfs merge=lfs -text
```

중간 체크포인트가 제외되는지 먼저 확인합니다.

```powershell
git status --short
```

그다음 모델과 LFS 설정을 추가합니다.

```powershell
git add .gitattributes
git add .gitignore
git add models/business-card-layoutxlm
git status
```

`checkpoint-*`가 Staged files에 나타나면 커밋하지 말고 `.gitignore`를 다시 확인합니다. 정상이라면 커밋하고 푸시합니다.

```powershell
git commit -m "Add trained LayoutXLM model with Git LFS"
git push
```

GitHub Desktop을 사용할 경우에도 먼저 `git lfs track`을 실행하고 `.gitattributes`를 포함한 다음 평소처럼 Commit과 Push를 진행합니다.

LFS 추적 여부 확인:

```powershell
git lfs ls-files
```

다음 경로가 표시되어야 합니다.

```text
models/business-card-layoutxlm/model.safetensors
```

## 3. Windows에서 저장소와 모델 다운로드

Git을 사용한 복제를 권장합니다. GitHub의 `Download ZIP`은 저장소 설정에 따라 실제 LFS 파일 대신 작은 포인터 파일만 포함할 수 있습니다.

```powershell
git lfs install
git clone <GitHub 저장소 URL>
cd python2team_Project
git lfs pull
```

이미 저장소를 받은 상태라면 다음 명령만 실행합니다.

```powershell
git pull
git lfs pull
git lfs checkout
```

모델 크기를 확인합니다.

```powershell
Get-Item models\business-card-layoutxlm\model.safetensors | Select-Object FullName,Length
```

약 1.4GB이면 정상입니다. 몇 바이트 또는 몇 KB이면 LFS 포인터만 내려받은 상태이므로 `git lfs pull`을 다시 실행합니다.

## 4. Ubuntu/WSL에서 저장소와 모델 다운로드

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
git clone <GitHub 저장소 URL>
cd python2team_Project
git lfs pull
```

모델 확인:

```bash
ls -lh models/business-card-layoutxlm/model.safetensors
git lfs ls-files
```

기존 저장소를 갱신할 때:

```bash
git pull
git lfs pull
git lfs checkout
```

## 5. 다운로드한 모델 실행

WSL/Linux 환경에서 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-ai.txt
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
python -m pip install -r requirements.txt
```

CPU만 사용하는 환경은 PyTorch 설치 주소를 다음과 같이 변경합니다.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

모델 경로를 설정하고 실행합니다.

```bash
export CARDOCR_LAYOUT_MODEL_DIR="$PWD/models/business-card-layoutxlm"
python app.py
```

Windows 브라우저에서 상태를 확인합니다.

```text
http://127.0.0.1:5000/api/health
```

정상 연결 값:

```json
{
  "configured": true,
  "dependencies_installed": true,
  "mode": "layoutxlm-hybrid",
  "error": ""
}
```

첫 명함 OCR 후 `ready: true`가 되면 모델 로딩까지 완료된 것입니다.

## 6. 자주 발생하는 문제

### GitHub에서 100MB 제한 오류

`model.safetensors`가 일반 Git 객체로 추가된 상태입니다. `.gitattributes`와 `git lfs ls-files`를 확인하고 커밋 전에 LFS 추적을 설정합니다. 이미 일반 Git 커밋에 포함했다면 해당 커밋의 대용량 파일을 LFS로 이전해야 하므로 새 커밋만 반복해서 만들지 마세요.

### 모델 크기가 몇 KB밖에 되지 않음

LFS 포인터만 받은 상태입니다.

```bash
git lfs pull
git lfs checkout
```

### LFS 다운로드가 중단됨

GitHub LFS 저장공간 또는 월간 대역폭 한도를 확인합니다. 저장소 소유자의 LFS 사용량이 한도를 초과하면 실제 모델 대신 포인터만 내려받을 수 있습니다.

### ZIP 다운로드에서 모델이 없음

GitHub 저장소의 `Settings → Archives → Include Git LFS objects in archives` 설정을 확인하거나 Git clone과 `git lfs pull`을 사용합니다.

### 모델은 있지만 `rules-only`로 표시됨

다음 파일과 환경변수를 확인합니다.

```bash
ls -lh models/business-card-layoutxlm/config.json
ls -lh models/business-card-layoutxlm/model.safetensors
echo "$CARDOCR_LAYOUT_MODEL_DIR"
```

환경변수를 다시 지정한 뒤 서버를 재시작합니다.

```bash
export CARDOCR_LAYOUT_MODEL_DIR="$PWD/models/business-card-layoutxlm"
python app.py
```

