# ============================================
# ⚙️ Whisper + Streaming + Flask 실시간 번역 설정파일
# ============================================

# --- Whisper 모델 설정 ---
# 모델 타입: "tiny", "base", "small", "medium", "large-v3"
MODEL_TYPE = "small"

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
INPUT_DEVICE_INDEX = "CABLE Output(VB-Audio Virtual C"
# 예: INPUT_DEVICE_INDEX = 12

# ============================================
# 🎙️ 음성 감지 (VAD, Voice Activity Detection)
# ============================================

# 0~3 (민감도 높을수록 잘 감지하지만 잡음 포함 가능)
VAD_MODE = 1

# VAD 분석 단위 (ms)
# webrtcvad는 10, 20, 30ms만 지원
FRAME_DURATION_MS = 30

# 말이 끝났다고 판단할 무음 지속 시간(ms)
# 너무 길면 문장 완성이 늦어지고, 너무 짧으면 끊김 발생
SILENCE_TIMEOUT_MS = 700

# --- 오디오 스트리밍 설정 ---
RATE = 16000
FRAME_DURATION = FRAME_DURATION_MS
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
CHUNK_DURATION_SEC = 3.0  # ✅ 청크 3초 단위로 변경
CHUNK_SIZE = int(RATE * CHUNK_DURATION_SEC)
