# 🎙️ AI Real-Time Translation & Meeting Assistant

> **실시간 다국어 회의 통역 및 AI 기반 화자 분리/요약 솔루션** > OpenAI의 **Whisper**로 듣고, **Google Translate**로 번역하며, **Pyannote**로 화자를 구별하고, **KoBART**로 핵심을 요약합니다.

---

## 📌 프로젝트 소개 (Overview)

이 프로젝트는 글로벌 미팅이나 다국어 대화 환경에서 실시간 소통을 돕고, 회의가 끝난 후에는 **누가 무슨 말을 했는지(Speaker Diarization)** 분석하여 **요약본(Summary)**을 제공하는 올인원 웹 애플리케이션입니다.

* **저사양 최적화:** VRAM 2GB~4GB GPU 및 CPU 환경에서도 구동 가능하도록 설계되었습니다.

---

## ✨ 핵심 기능 (Key Features)

### 1. 🎙️ 실시간 음성 번역 (Real-Time Translation)
* **고성능 STT:** `Faster-Whisper` 모델을 사용하여 빠르고 정확하게 음성을 텍스트로 변환합니다.
* **VAD 탑재:** `Silero VAD`를 적용하여 무음 구간을 필터링하고 목소리만 정확히 감지합니다.
* **즉시 번역:** 감지된 문장을 실시간으로 한국어로 번역하여 화면에 출력합니다.

### 2. 🗣️ 정밀 화자 분석 (Speaker Diarization)
* **화자 식별:** 회의 종료 후, 녹음된 오디오를 분석하여 **"누가(Speaker A, B...)"** 말했는지 타임스탬프와 함께 구분합니다.
* **문맥 보정:** `WhisperX`와 `Pyannote.audio` 파이프라인을 통해 STT 결과와 화자 정보를 정밀하게 매칭합니다.

### 3. 📑 긴 문맥 AI 요약 (Meeting Summarization)
* **Map-Reduce 요약:** `KoBART` 모델을 활용하여 10분 이상의 긴 회의 내용도 문맥을 놓치지 않고 요약합니다.
* **자동 분할 처리:** 텍스트 길이가 모델 한계를 초과할 경우, 자동으로 분할 요약 후 통합합니다.

### 4. 💾 세션 관리 및 데이터 처리
* **자동 저장:** 모든 회의 기록(텍스트)과 오디오(`.wav`)는 데이터베이스와 로컬 스토리지에 자동 저장됩니다.
* **관리 기능:** 지난 세션을 다시 불러오거나, 불필요한 데이터를 영구 삭제할 수 있습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

* **Frontend:** HTML5, CSS3, JavaScript (Socket.IO Client)
* **Backend:** Python Flask, Flask-Socket.IO, SQLite
* **AI Models:**
    * **STT:** Faster-Whisper (OpenAI Whisper implementation)
    * **VAD:** Silero VAD
    * **Diarization:** Pyannote.audio, WhisperX
    * **Summarization:** KoBART (SKT-AI)
* **Audio Processing:** FFmpeg, SoundDevice, NumPy

---

## ⚙️ 사전 준비 (Prerequisites)

설치 전 다음 항목들이 준비되어 있어야 합니다.

1.  **Python 3.10 이상:** 설치 시 `Add Python to PATH` 옵션 체크 필수.
2.  **FFmpeg:** 오디오 처리를 위한 필수 라이브러리.
    * *(팁: `ffmpeg.exe`를 프로젝트 폴더에 넣으면 별도 설치 없이 작동합니다.)*
3.  **Hugging Face 토큰:** 화자 분리 모델 사용을 위해 필요합니다.
    * [Hugging Face 설정 페이지](https://huggingface.co/settings/tokens)에서 `Read` 권한 토큰 발급.
    * **필수:** 다음 모델들의 사용자 약관 동의(Access Request)를 해야 합니다.
        * `pyannote/speaker-diarization-3.1`
        * `pyannote/segmentation-3.0`

---

## 🚀 설치 및 실행 (Quick Start)

### 1. 라이브러리 설치
동봉된 설치 스크립트를 실행하여 가상 환경을 만들고 의존성을 설치합니다.
* **Windows:** `install_libs.bat` 더블 클릭

### 2. 설정 파일 수정 (`config.py`)
`config.py` 파일을 열어 발급받은 Hugging Face 토큰을 입력합니다.

```python
# config.py
HF_TOKEN = "hf_your_token_here"  # 본인의 토큰으로 변경
```
### 3. 실행
동봉된 실행 스크립트를 실행합니다.
* **Windows:** `run_project.bat` 더블 클릭