# 🎙️ AI Real-Time Translation \& Meeting Assistant

# 실시간 다국어 회의 통역 및 AI 기반 화자 분리/요약 솔루션

# 

# OpenAI의 Whisper로 듣고, Google Translate로 번역하며, Pyannote로 화자를 구별하고, KoBART로 핵심을 요약합니다.

# 

# 📌 프로젝트 소개 (Overview)

# 이 프로젝트는 글로벌 미팅이나 다국어 대화 환경에서 실시간 소통을 돕고, 회의가 끝난 후에는 누가 무슨 말을 했는지(Speaker Diarization) 분석하여 \*\*요약본(Summary)\*\*을 제공하는 올인원 웹 애플리케이션입니다.

# 

# 저사양 GPU(VRAM 2GB~4GB) 및 CPU 환경에서도 구동 가능하도록 최적화되어 있습니다.

# 

# ✨ 핵심 기능 (Key Features)

# 1\. 🎙️ 실시간 음성 번역 (Real-Time Translation)

# 고성능 STT: Faster-Whisper 모델을 사용하여 빠르고 정확하게 음성을 텍스트로 변환합니다.

# 

# VAD 탑재: Silero VAD를 적용하여 무음 구간을 필터링하고 목소리만 정확히 감지합니다.

# 

# 즉시 번역: 감지된 문장을 실시간으로 한국어로 번역하여 화면에 출력합니다.

# 

# 2\. 🗣️ 정밀 화자 분석 (Speaker Diarization)

# 화자 식별: 회의 종료 후, 녹음된 오디오를 분석하여 "누가(Speaker A, B...)" 말했는지 타임스탬프와 함께 구분합니다.

# 

# 문맥 보정: WhisperX와 Pyannote.audio 파이프라인을 통해 STT 결과와 화자 정보를 정밀하게 매칭합니다.

# 

# 3\. 📑 긴 문맥 AI 요약 (Meeting Summarization)

# Map-Reduce 요약: KoBART 모델을 활용하여 10분 이상의 긴 회의 내용도 문맥을 놓치지 않고 요약합니다.

# 

# 자동 분할 처리: 텍스트 길이가 모델 한계를 초과할 경우, 자동으로 분할 요약 후 통합합니다.

# 

# 4\. 💾 세션 관리 및 데이터 처리

# 자동 저장: 모든 회의 기록(텍스트)과 오디오(.wav)는 데이터베이스와 로컬 스토리지에 자동 저장됩니다.

# 

# 내보내기: 대화 로그와 분석 결과를 .txt 파일로 다운로드할 수 있습니다.

# 

# 관리 기능: 지난 세션을 다시 불러오거나, 불필요한 데이터를 영구 삭제할 수 있습니다.

# 

# 🛠️ 기술 스택 (Tech Stack)

# Frontend: HTML5, CSS3, JavaScript (Socket.IO Client)

# 

# Backend: Python Flask, Flask-Socket.IO, SQLite

# 

# AI Models:

# 

# STT: Faster-Whisper (OpenAI Whisper implementation)

# 

# VAD: Silero VAD

# 

# Diarization: Pyannote.audio, WhisperX

# 

# Summarization: KoBART (SKT-AI)

# 

# Audio Processing: FFmpeg, SoundDevice, NumPy

# 

# ⚙️ 사전 준비 (Prerequisites)

# 설치 전 다음 항목들이 준비되어 있어야 합니다.

# 

# Python 3.10 이상: 설치 시 Add Python to PATH 옵션 체크 필수.

# 

# FFmpeg: 오디오 처리를 위한 필수 라이브러리. (다운로드 가이드)

# 

# 팁: ffmpeg.exe를 프로젝트 폴더에 넣으면 별도 설치 없이 작동합니다.

# 

# Hugging Face 토큰: 화자 분리 모델 사용을 위해 필요합니다.

# 

# Hugging Face 설정 페이지에서 Read 권한 토큰 발급.

# 

# 다음 모델들의 사용자 약관 동의(Access Request) 필수:

# 

# pyannote/speaker-diarization-3.1

# 

# pyannote/segmentation-3.0

# 

# 🚀 설치 및 실행 (Quick Start)

# 1\. 라이브러리 설치

# 동봉된 설치 스크립트를 실행하여 가상 환경을 만들고 의존성을 설치합니다.

# 

# Windows: install\_libs.bat 더블 클릭

# 

# 2\. 설정 파일 수정 (config.py)

# config.py 파일을 열어 발급받은 Hugging Face 토큰을 입력합니다.

# 

# Python

# 

# \# config.py

# HF\_TOKEN = "hf\_your\_token\_here"  # 본인의 토큰으로 변경

# 3\. 서버 실행

# 실행 스크립트를 더블 클릭하여 서버를 시작합니다.

# 

# Windows: run\_project.bat 더블 클릭

# 

# 서버가 정상적으로 실행되면 브라우저에서 \*\*http://127.0.0.1:5000\*\*으로 접속하세요.

# 

# 📖 사용 가이드 (User Guide)

# 실시간 번역 시작

# 

# \[Start Translation] 버튼 클릭 후 세션 이름(예: Meeting\_01)을 입력합니다.

# 

# 마이크에 대고 말하면 실시간으로 인식 및 번역되어 로그에 표시됩니다.

# 

# 세션 종료 및 저장

# 

# 회의가 끝나면 반드시 \[End Session]을 눌러 녹음을 종료하고 데이터를 저장합니다.

# 

# 화자 분석 (Diarization)

# 

# 메인 화면의 드롭다운에서 분석할 세션을 선택합니다.

# 

# \[화자 분석] 버튼을 클릭합니다. (CPU 모드에서는 오디오 길이의 1~2배 시간이 소요될 수 있습니다.)

# 

# 분석이 완료되면 로그 창에 화자별(Speaker A, B...)로 정리된 대본이 표시됩니다.

# 

# 회의 요약 (Summary)

# 

# \[요약] 버튼을 누르면 KoBART 모델이 전체 대화 내용을 요약하여 팝업창에 보여줍니다.

# 

# ⚠️ 트러블슈팅 (Troubleshooting)

# CUDA out of memory 오류:

# 

# GPU VRAM이 부족할 때 발생합니다. config.py에서 MODEL\_TYPE을 tiny로 변경하거나, DEVICE를 cpu로 설정하여 CPU 모드를 사용하세요.

# 

# FFmpeg not found 오류:

# 

# FFmpeg가 설치되지 않았거나 시스템 경로(PATH)에 등록되지 않았습니다.

# 

# 화자 분석이 실행되지 않음:

# 

# Hugging Face 토큰이 올바른지, 해당 모델 페이지에서 약관 동의를 했는지 확인하세요.

# 

# 📂 디렉토리 구조

# Bash

# 

# If\_Project/

# ├── app.py               # 메인 Flask 서버 (Socket.IO 핸들러)

# ├── audio\_processor.py   # 오디오 스트리밍, 녹음, VAD, STT 처리

# ├── diarize\_handler.py   # 화자 분리 및 후처리 로직

# ├── summary\_handler.py   # KoBART 기반 텍스트 요약 로직

# ├── db\_handler.py        # SQLite DB 데이터 입출력 관리

# ├── config.py            # 프로젝트 설정 및 토큰 관리

# ├── templates/

# │   └── translation.html # 웹 프론트엔드 UI

# ├── wav/                 # 녹음된 세션별 오디오 파일 저장소

# ├── translations.db      # 대화 로그 저장 데이터베이스

# ├── requirements.txt     # 파이썬 라이브러리 목록

# └── run\_project.bat      # 간편 실행 스크립트

