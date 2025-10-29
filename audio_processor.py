import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import time
from deep_translator import GoogleTranslator

# 로컬 모듈 임포트
from config import (
    MODEL_TYPE, LANGUAGE, BLOCK_DURATION, TARGET_LANG,
    VOLUME_THRESHOLD, BEAM_SIZE
)
from db_handler import insert_transcript

# (OBS 관련 'utils' 임포트 제거)

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


def main_audio_loop(session_id, latest_data):
    """
    메인 오디오 처리 스레드 함수.
    latest_data 딕셔너리를 직접 수정하여 app.py와 통신합니다.
    """
    print(f"🗂️ 세션 시작: {session_id}")
    latest_data["session_id"] = session_id

    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
        print("🎤 [스레드] 음성 인식 + 번역 + DB 저장 시작 (Ctrl+C로 종료)")
        buffer = np.zeros((0,), dtype=np.float32)
        last_text = ""

        while True:
            try:
                # 큐에서 데이터 가져와 버퍼에 누적
                while not audio_q.empty():
                    block = audio_q.get()
                    buffer = np.concatenate((buffer, block.flatten()))

                # 버퍼가 최소 처리 단위(BLOCK_DURATION)보다 짧으면 대기
                if len(buffer) < 16000 * BLOCK_DURATION:
                    time.sleep(0.1)
                    continue

                # 처리할 세그먼트 준비 및 버퍼 비우기 (딜레이 방지)
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

                        # --- OBS 파일 업데이트 코드 제거됨 ---

                        # DB 업데이트
                        insert_transcript(session_id, combined_text, translated)

                        # app.py와 통신 (메인 스레드용)
                        latest_data["original"] = combined_text
                        latest_data["translated"] = translated

            except KeyboardInterrupt:
                print("🛑 [스레드] 음성 인식 종료.")
                break
            except Exception as e:
                print(f"오디오 루프 오류: {e}")
                time.sleep(1)

