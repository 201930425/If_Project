# --- 설정 (여기서 성능 조절) ---

# 모델 타입: "tiny", "base", "small", "medium", "large-v3"
MODEL_TYPE = "small"

# 오디오 처리 주기(초).
# BLOCK_DURATION = 8
# VAD 기반 코드는 음성의 길이를 자동 감지하므로 BLOCK_DURATION 불필요

# Whisper beam_size (기본값 5)
BEAM_SIZE = 8

# --- 기본 설정 ---
LANGUAGE = "en" # 일본어: ja, 영어: en, 한국어: ko 등 , None: 언어 자동인식
TARGET_LANG = "ko"

# --- 데이터베이스 설정 ---
DB_NAME = "translations.db"

# ⬇️ --- (수정) --- ⬇️
# '대화 요약' 모델 로드에 거듭 실패하여,
# 로드가 확인된 'gogamza' 모델로 되돌립니다.
KOBART_MODEL_NAME = "gogamza/kobart-summarization"
# KOBART_MODEL_NAME = "j-min/kobart-talk-summary"
# ⬆️ --- (수정 완료) --- ⬆️

# --- 서버 설정 ---
HOST = "0.0.0.0"
PORT = 5000

# --- (스테레오 믹스 설정) ---
# 사용자의 오디오 입력 장치 인덱스 (None = PC 기본 사운드)
# '스테레오 믹스' 등을 사용하려면 'check_mic.py'로 인덱스 확인

# INPUT_DEVICE_INDEX = 12  # 예시: 특정 오디오 장치 인덱스 ( 기본값: None )
INPUT_DEVICE_INDEX = None
#'스테레오 믹스' 사용시 재생 -> "스피커", 녹음 -> "스테레오믹스" 잡아줘야함
#'VB-CABLE' 사용시 VB input and VB output 둘다 잡아줘야함


# ============================================
# 🎙️ 음성 감지 (VAD, Voice Activity Detection) 설정
# ============================================

# VAD 민감도 (0~3)
# 값이 높을수록 소리 감지에 민감 (하지만 잡음도 감지할 수 있음)
VAD_MODE = 2
# VAD_MODE = 2

# webrtcvad는 10, 20, 30ms 프레임만 지원
FRAME_DURATION_MS = 30  # 감지 주기(ms)

# 침묵 지속 시간(밀리초)
# 말 멈춘 후 이 시간 동안 침묵이면 인식 시작
SILENCE_TIMEOUT_MS = 600
# SILENCE_TIMEOUT_MS = 800 (기본값) # 숫자 커지면 문장길이가 길어지는 문제 발생
