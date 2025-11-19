# ============================================
# ⚙️ Whisper + Streaming + Flask 실시간 번역 설정파일
# ============================================

# --- Whisper 모델 설정 ---
# 모델 타입: "tiny", "base", "small", "medium", "large-v3"
MODEL_TYPE = "tiny"

# Whisper beam search 크기 (높을수록 정확하지만 느림)
BEAM_SIZE = 8

# 언어 설정
# 예시: 영어(en), 일본어(ja), 한국어(ko), None(자동 인식)
LANGUAGE = "en"
TARGET_LANG = "ko"

# --- 데이터베이스 설정 ---
DB_NAME = "translations.db"

# --- KoBART 요약 모델 ---
KOBART_MODEL_NAME = "gogamza/kobart-summarization"

# --- 서버 설정 ---
HOST = "0.0.0.0"
PORT = 5000

# --- 오디오 입력 장치 ---
# 특정 장치를 지정하지 않으면 자동 기본 입력 사용
INPUT_DEVICE_INDEX = None
# 예: INPUT_DEVICE_INDEX = 12

# ============================================
# 🎙️ 음성 감지 (VAD, Voice Activity Detection)
# ============================================

# ⭐️ [수정] Silero-VAD 설정으로 변경
# (Silero-VAD) VAD 민감도 임계값 (0~1 사이, 낮을수록 민감)
VAD_THRESHOLD = 0.45

# (Silero VAD는 16kHz에서 512, 1024, 1536 샘플 권장)
# 512 샘플 = 32ms
VAD_FRAME_SIZE = 512 # 512 샘플 (32ms)

# ❌ (제거) VAD_MODE = 1
# ❌ (제거) FRAME_DURATION_MS = 30

# 말이 끝났다고 판단할 무음 지속 시간(ms)
SILENCE_TIMEOUT_MS = 700

# --- 오디오 스트리밍 설정 ---
RATE = 16000
# ⭐️ [수정] FRAME_SIZE를 VAD_FRAME_SIZE로 교체
FRAME_SIZE = VAD_FRAME_SIZE
# ❌ (제거) FRAME_DURATION = FRAME_DURATION_MS
CHUNK_DURATION_SEC = 3.0  # ✅ 청크 3초 단위로 변경
CHUNK_SIZE = int(RATE * CHUNK_DURATION_SEC)

# ⚠️ [필수] Hugging Face 토큰 (https://huggingface.co/settings/tokens)
# (보안을 위해 '' 안에 넣는 것보다 환경 변수 사용을 권장합니다. 아래 팁 참조)
HF_TOKEN = "hf_your_token_here"

# VRAM 2GB 한계로 CPU 사용
DIARIZE_DEVICE = "cpu"
DIARIZE_COMPUTE_TYPE = "int8" # CPU에서 실행하기 위한 타입

# 후처리에 사용할 Whisper 모델 (실시간 모델과 다르게 설정 가능)
# (예: 실시간은 "tiny", 후처리는 "base" 또는 "small" 사용)
DIARIZE_MODEL_TYPE = "tiny"