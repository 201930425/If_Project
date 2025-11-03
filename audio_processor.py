# audio_processor.py
import sounddevice as sd
import numpy as np
import queue
import time
import webrtcvad
import collections
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from datetime import datetime, timezone, timedelta
import threading
import traceback

# 로컬 모듈
from config import (
    MODEL_TYPE, LANGUAGE, TARGET_LANG,
    BEAM_SIZE, INPUT_DEVICE_INDEX,
    VAD_MODE, FRAME_DURATION_MS, SILENCE_TIMEOUT_MS
)
from db_handler import insert_transcript

# --- 설정 & 초기화 ---
RATE = 16000
FRAME_DURATION = FRAME_DURATION_MS  # 예: 30 (ms)
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 샘플 수(프레임당)
assert FRAME_DURATION in (10, 20, 30), "webrtcvad는 10/20/30 ms 프레임만 지원합니다."

vad = webrtcvad.Vad(VAD_MODE)  # 0~3: 민감도
audio_q = queue.Queue()

print(f"🎧 Whisper 모델 ({MODEL_TYPE}) 로드 중...")
model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")

# ---------------- 오류 수정위한 코드 확인 공간 ------------- #
# 전체 장치 리스트 출력
for i, dev in enumerate(sd.query_devices()):
    print(i, dev['name'], "max_input_channels=", dev['max_input_channels'])

# 현재 기본 장치 정보(튜플: (input_index, output_index))
print("default device:", sd.default.device)


# ------------------------------
def audio_callback(indata, frames, time_, status):
    """sounddevice.InputStream의 콜백 (indata는 numpy.ndarray)"""
    if status:
        # 입력 버퍼 오버런 등 상태 로그
        print(f"[Audio status] {status}")
    # indata는 numpy array (frames, channels)
    # copy해서 큐에 넣어 안전하게 사용
    try:
        audio_q.put(indata.copy())
    except Exception:
        # 매우 드물게 callback 내 에러가 발생하면 워닝만 남기고 계속
        print("⚠️ audio_callback 큐에 넣기 실패:")
        traceback.print_exc()


def translate_text_local(text, target_lang=TARGET_LANG):
    if not text or not text.strip():
        return "[빈 문자열]"
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"⚠️ 번역 실패: {e}")
        return "[번역 실패]"


def process_audio_segment(raw_blocks):
    """
    raw_blocks: list of ndarray (int16) 블록들
    반환: 인식된 텍스트 (문자열)
    """
    if not raw_blocks:
        return ""

    # 블록( ndarray shape=(frame_size, 1) )들을 연결
    data = np.concatenate(raw_blocks, axis=0)  # shape (N, 1)
    # mono shape -> flatten
    if data.ndim > 1:
        data = data.flatten()

    # faster_whisper에 맞게 float32 정규화 (필요 시)
    audio_float32 = data.astype(np.float32) / 32768.0

    try:
        segments, _ = model.transcribe(audio_float32, language=LANGUAGE, beam_size=BEAM_SIZE)
        combined_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return combined_text
    except Exception as e:
        print(f"⚠️ Whisper 처리 오류: {e}")
        traceback.print_exc()
        return ""


def main_audio_loop(session_id, socketio, stop_event=None):
    """
    session_id: 세션 식별자
    socketio: flask_socketio 또는 python-socketio 서버 인스턴스
    stop_event: threading.Event()로 외부에서 종료 신호 가능
    """
    print(f"🗂️ 세션 시작: {session_id}")

    # silence timeout 프레임 수 계산
    silence_timeout_frames = int(SILENCE_TIMEOUT_MS / FRAME_DURATION_MS)

    try:
        with sd.InputStream(
            device=INPUT_DEVICE_INDEX,
            samplerate=RATE,
            blocksize=FRAME_SIZE,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            if INPUT_DEVICE_INDEX is not None:
                try:
                    info = sd.query_devices(INPUT_DEVICE_INDEX)
                    print(f"🎧 장치: {info['name']} (인덱스 {INPUT_DEVICE_INDEX})")
                except Exception:
                    print(f"🎧 지정 장치 인덱스 {INPUT_DEVICE_INDEX}에서 녹음 시작.")
            else:
                print("🎤 PC 기본 사운드에서 녹음 시작.")

            buffer_blocks = []           # 현재 발화 블록 저장
            speaking = False
            silence_counter = 0

            while True:
                if stop_event is not None and stop_event.is_set():
                    print("🛑 stop_event 수신: 종료합니다.")
                    break

                try:
                    # 큐에서 가능한 모든 블록 수집
                    if audio_q.empty():
                        time.sleep(0.005)
                        continue

                    block = audio_q.get()  # numpy.ndarray (FRAME_SIZE, 1)
                    # webrtcvad는 raw bytes(16-bit PCM little-endian) 형태 입력을 받음
                    # block이 int16 ndarray라면 .tobytes()로 전달
                    is_speech = False
                    try:
                        is_speech = vad.is_speech(block.tobytes(), RATE)
                    except Exception as e:
                        # 안전장치: vad 호출 실패 시 RMS fallback (희박한 경우)
                        rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
                        is_speech = rms > 500  # 임시 임계값
                        print(f"⚠️ vad 실패 → RMS fallback 사용 (rms={rms})")

                    if is_speech:
                        buffer_blocks.append(block)
                        speaking = True
                        silence_counter = 0
                    elif speaking:
                        # 말하고 있다가 침묵으로 바뀐 경우 블록을 계속 모으고 침묵 카운트 증가
                        buffer_blocks.append(block)
                        silence_counter += 1

                    # 발화가 끝났다고 판단 시(침묵 지속)
                    if speaking and silence_counter >= silence_timeout_frames:
                        print("🛑 말 멈춤 감지 → 인식 처리")
                        try:
                            text = process_audio_segment(buffer_blocks)
                            if text:
                                print(f"🎤 인식: {text}")
                                translated = translate_text_local(text)
                                print(f"🌐 번역: {translated}")

                                # DB 저장 (예외 내부 처리)
                                try:
                                    insert_transcript(session_id, text, translated)
                                except Exception as e:
                                    print(f"⚠️ DB 저장 오류: {e}")

                                # socketio 이벤트 전송
                                try:
                                    kst = timezone(timedelta(hours=9))
                                    now_time = datetime.now(kst).strftime("%H:%M:%S")
                                    socketio.emit('new_translation', {
                                        'original': text,
                                        'translated': translated,
                                        'time': now_time,
                                        'session_id': session_id
                                    })
                                except Exception as e:
                                    print(f"⚠️ socketio 전송 오류: {e}")

                        finally:
                            # 버퍼 초기화
                            buffer_blocks = []
                            speaking = False
                            silence_counter = 0

                except KeyboardInterrupt:
                    print("🛑 키보드 인터럽트: 종료합니다.")
                    break
                except Exception as e:
                    # 루프 내 에러시 로그 찍고 잠시 쉬었다가 계속
                    print(f"오디오 루프 내부 오류: {e}")
                    traceback.print_exc()
                    time.sleep(0.5)

    except sd.PortAudioError as e:
        print("❌ 오디오 장치 오류:", e)
    except Exception as e:
        print("❌ 알 수 없는 오류:", e)
        traceback.print_exc()
