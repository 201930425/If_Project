# --- 설정 (여기서 성능 조절) ---

# 모델 타입: "tiny", "base", "small", "medium", "large-v3"
MODEL_TYPE = "small"
# MODEL_TYPE = "medium"

# 오디오 처리 주기(초).
BLOCK_DURATION = 8
# BLOCK_DURATION = 8

# Whisper beam_size (기본값 5)
BEAM_SIZE = 5
# BEAM_SIZE = 10

# --- 기본 설정 ---
LANGUAGE = "en"
TARGET_LANG = "ko"
VOLUME_THRESHOLD = 0.001
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
# 사용자의 오디오 입력 장치 인덱스 (None = 기본 마이크)
# '스테레오 믹스' 등을 사용하려면 'check_mic.py'로 인덱스 확인
INPUT_DEVICE_INDEX = None
# INPUT_DEVICE_INDEX = 12

