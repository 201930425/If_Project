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
KOBART_MODEL_NAME = "gogamza/kobart-summarization"
INPUT_DEVICE_INDEX = None

# --- 서버 설정 ---
HOST = "0.0.0.0"
PORT = 5000

