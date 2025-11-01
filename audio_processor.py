import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import time
from deep_translator import GoogleTranslator
from datetime import datetime, timezone, timedelta  # 1. 시간 임포트

# 로컬 모듈 임포트
from config import (
    MODEL_TYPE, LANGUAGE, BLOCK_DURATION, TARGET_LANG,
    VOLUME_THRESHOLD, BEAM_SIZE, INPUT_DEVICE_INDEX  # 2. INPUT_DEVICE_INDEX 추가
)
from db_handler import insert_transcript

# --- Whisper 모델 및 오디오 큐 ---
print(f"🎧 Whisper 모델 ({MODEL_TYPE}) 로드 중...")
model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
audio_q = queue.Queue()


# --------------------------------

def audio_callback(indata, frames, time_, status):
    """오디오 데이터를 큐에 넣습니다."""
    if status:
        print(status)
    audio_q.put(indata.copy())


def translate_text_local(text, target_lang=TARGET_LANG):
    """텍스트를 번역합니다."""
    if not text.strip():
        return "[빈 문자열]"
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        print(f"⚠️ 번역 실패: {e}")
        return "[번역 실패]"


def is_speech(buffer, threshold=VOLUME_THRESHOLD):
    """최소 볼륨을 체크합니다."""
    rms = np.sqrt(np.mean(buffer ** 2))
    return rms > threshold


# 3. main_audio_loop 인수가 latest_data에서 socketio로 변경됨
def main_audio_loop(session_id, socketio):
    """
    메인 오디오 처리 스레드 함수.
    socketio 객체를 통해 클라이언트로 직접 데이터를 전송합니다.
    """
    print(f"🗂️ 세션 시작: {session_id}")

    try:
        # 4. sd.InputStream에 device= 설정 추가 (이전 수정 사항 반영)
        with sd.InputStream(
                device=INPUT_DEVICE_INDEX,
                samplerate=16000,
                channels=1,
                callback=audio_callback
        ):
            if INPUT_DEVICE_INDEX is not None:
                try:
                    device_info = sd.query_devices(INPUT_DEVICE_INDEX)
                    print(f"🎧 [스레드] 지정된 장치 '{device_info['name']}' (인덱스: {INPUT_DEVICE_INDEX})에서 녹음 시작.")
                except Exception:
                    print(f"🎧 [스레드] 지정된 장치 (인덱스: {INPUT_DEVICE_INDEX})에서 녹음 시작.")
            else:
                print("🎤 [스레드] '기본 마이크'에서 음성 인식 + 번역 + DB 저장 시작 (Ctrl+C로 종료)")

            buffer = np.zeros((0,), dtype=np.float32)
            last_text = ""

            while True:
                try:
                    while not audio_q.empty():
                        block = audio_q.get()
                        buffer = np.concatenate((buffer, block.flatten()))

                    if len(buffer) < 16000 * BLOCK_DURATION:
                        time.sleep(0.1)
                        continue

                    segment_to_process = buffer
                    buffer = np.zeros((0,), dtype=np.float32)

                    if is_speech(segment_to_process):
                        segments, _ = model.transcribe(
                            segment_to_process.flatten(),
                            language=LANGUAGE,
                            beam_size=BEAM_SIZE
                        )
                        combined_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

                        if combined_text and combined_text != last_text:
                            last_text = combined_text
                            print(f"🎤 인식: {combined_text}")
                            translated = translate_text_local(combined_text)
                            print(f"🌐 번역: {translated}")

                            insert_transcript(session_id, combined_text, translated)

                            # 5. --- (핵심 변경) ---
                            # latest_data 딕셔너리 대신 socketio.emit()으로 데이터 전송
                            kst = timezone(timedelta(hours=9))
                            now_time = datetime.now(kst).strftime("%H:%M:%S")

                            socketio.emit('new_translation', {
                                'original': combined_text,
                                'translated': translated,
                                'time': now_time
                            })
                            # --- (변경 완료) ---

                except KeyboardInterrupt:
                    print("🛑 [스레드] 음성 인식 종료.")
                    break
                except Exception as e:
                    print(f"오디오 루프 오류: {e}")
                    time.sleep(1)

    except sd.PortAudioError as e:
        print("\n" + "=" * 50)
        print(f"❌ 오디오 장치 오류: {e}")
        if INPUT_DEVICE_INDEX is not None:
            print(f"지정한 입력 장치 인덱스 '{INPUT_DEVICE_INDEX}'를 열 수 없습니다.")
        else:
            print("기본 입력 장치(마이크)를 열 수 없습니다.")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"❌ 알 수 없는 오디오 스레드 시작 오류: {e}")

