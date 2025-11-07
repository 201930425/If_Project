import sounddevice as sd
import numpy as np
import queue
import time
import webrtcvad
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from datetime import datetime, timezone, timedelta
import traceback
import noisereduce as nr
import asyncio
import edge_tts  # 🔊 추가됨 — Edge TTS
import tempfile
import os
import soundfile as sf


import config
from db_handler import insert_transcript
from config import (
    MODEL_TYPE, LANGUAGE, TARGET_LANG,
    BEAM_SIZE, INPUT_DEVICE_INDEX, FRAME_SIZE,
    VAD_MODE, FRAME_DURATION_MS, RATE, CHUNK_DURATION_SEC, CHUNK_SIZE
)

print(f"🎧 Whisper 모델({MODEL_TYPE}) 로드 중...")
model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
vad = webrtcvad.Vad(VAD_MODE)
audio_q = queue.Queue()

# --- 번역 ---
def translate_text_local(text, target_lang=TARGET_LANG):
    if not text or not text.strip():
        return ""
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"⚠️ 번역 실패: {e}")
        return ""

# --- 🔊 Edge-TTS 비동기 음성 출력 (병렬 실행 가능) ---
async def speak_text_async(text, target_lang=TARGET_LANG):
    """비동기 음성 출력"""
    if not text.strip():
        return

    try:
        # 언어별 음성 선택
        voice_map = {
            "ko": "ko-KR-SunHiNeural",
            "en": "en-US-JennyNeural",
            "ja": "ja-JP-NanamiNeural",
            "zh-CN": "zh-CN-XiaoxiaoNeural",
            "de": "de-DE-KatjaNeural",
            "fr": "fr-FR-DeniseNeural"
        }
        voice = voice_map.get(target_lang, "en-US-JennyNeural")

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            output_path = tmp.name

        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(output_path)

        # 재생을 별도 스레드로 수행 → Whisper 인식 중에도 재생됨
        def play_audio_file():
            try:
                data, samplerate = sf.read(output_path)
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                print(f"⚠️ 음성 재생 오류: {e}")
            finally:
                try:
                    os.remove(output_path)
                except:
                    pass

        # 비동기로 실행
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, play_audio_file)

    except Exception as e:
        print(f"⚠️ 음성 출력 실패: {e}")


def speak_text(text, target_lang=TARGET_LANG):
    """스레드 환경에서도 안전하게 Edge-TTS 실행"""
    if not text.strip():
        return

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 이미 asyncio 루프가 실행 중이라면 비동기로 태스크 생성
            asyncio.create_task(speak_text_async(text, target_lang))
        else:
            # 현재 스레드에 이벤트 루프가 없으면 새로 생성
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(speak_text_async(text, target_lang))
            new_loop.close()

    except Exception as e:
        print(f"⚠️ speak_text 실행 오류: {e}")



# --- 문장 완성 감지 (구두점 + 무음 기반) ---
def is_sentence_complete(text):
    """문장이 끝났는지 판별"""
    if not text.strip():
        return False
    text = text.strip()
    return text.endswith((".", "!", "?", "요", "다", "죠", "네", "습니다"))

# --- 오디오 콜백 ---
def audio_callback(indata, frames, time_, status):
    if status:
        print(f"[Audio status] {status}")
    try:
        audio_q.put(indata.copy())
    except Exception:
        traceback.print_exc()

# --- 음성 감지 ---
def is_speech_chunk(data_chunk, rate=RATE, frame_ms=30):
    """RMS + VAD 기반 음성 감지"""
    frame_length = int(rate * frame_ms / 1000)
    bytes_data = data_chunk.tobytes()
    speech_frames = 0
    frame_count = 0

    for i in range(0, len(bytes_data), frame_length * 2):
        frame = bytes_data[i:i + frame_length * 2]
        if len(frame) < frame_length * 2:
            break
        frame_count += 1
        try:
            if vad.is_speech(frame, rate):
                speech_frames += 1
        except webrtcvad.Error:
            continue

    return speech_frames > 0

# --- 메인 루프 ---
def main_audio_streaming(session_id, socketio, stop_event=None):
    print(f"🗂️ 세션 시작 (스트리밍 모드): {session_id}")

    buffer = np.zeros((0, 1), dtype=np.int16)
    sentence_buffer = ""
    previous_text = ""
    last_emit_time = time.time()
    silence_counter = 0

    try:
        with sd.InputStream(
            device=INPUT_DEVICE_INDEX,
            samplerate=RATE,
            blocksize=FRAME_SIZE,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            print("🎤 실시간 음성 인식 시작...")

            while True:
                if stop_event is not None and stop_event.is_set():
                    print("🛑 stop_event 수신: 종료합니다.")
                    break

                if not audio_q.empty():
                    block = audio_q.get()
                    buffer = np.concatenate((buffer, block), axis=0)

                    if len(buffer) >= CHUNK_SIZE:
                        data_chunk = buffer[:CHUNK_SIZE]
                        buffer = buffer[CHUNK_SIZE:]

                        try:
                            # 🔉 음성 감지
                            if not is_speech_chunk(data_chunk, RATE):
                                silence_counter += 1
                                if silence_counter >= 2 and sentence_buffer.strip():
                                    # ✅ 1.5초 이상 무음 → 문장 완료로 간주
                                    kst = timezone(timedelta(hours=9))
                                    now_time = datetime.now(kst).strftime("%H:%M:%S")

                                    translated = translate_text_local(sentence_buffer)
                                    print(f"✅ 완성 문장: {sentence_buffer}")
                                    print(f"🌐 번역 결과: {translated}\n")

                                    # 🔊 번역된 문장 음성 출력
                                    speak_text(translated, TARGET_LANG)

                                    socketio.emit('partial_translation', {
                                        'original': sentence_buffer.strip(),
                                        'translated': translated,
                                        'time': now_time,
                                        'session_id': session_id
                                    })

                                    insert_transcript(session_id, sentence_buffer.strip(), translated)
                                    sentence_buffer = ""
                                    previous_text = ""
                                    silence_counter = 0
                                continue
                            else:
                                silence_counter = 0

                                # 🔉 노이즈 제거
                                reduced = nr.reduce_noise(y=data_chunk.flatten(), sr=RATE)

                                # ⭐️ [수정] 0으로 나누기 오류(RuntimeWarning) 방지
                                max_val = np.max(np.abs(reduced))

                                if max_val > 0:
                                    normalized_audio = reduced / max_val
                                else:
                                    normalized_audio = reduced

                                reduced_int16 = np.int16(normalized_audio * 32767)
                                audio_float32 = reduced_int16.astype(np.float32) / 32768.0

                            # 🧠 Whisper 인식
                            segments, _ = model.transcribe(
                                audio_float32,
                                language=config.LANGUAGE,
                                beam_size=BEAM_SIZE,
                                vad_filter=True,
                                no_speech_threshold=0.6,
                                log_prob_threshold=-1.0,
                                condition_on_previous_text=False
                            )
                            partial_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

                            if partial_text and partial_text != previous_text:
                                new_part = partial_text.replace(previous_text, "").strip()
                                if new_part:
                                    sentence_buffer += " " + new_part
                                    previous_text = partial_text
                                    print(f"🧩 부분 인식 누적: {new_part}")

                                # 종결어미 기반 문장 완성 감지
                                if is_sentence_complete(sentence_buffer):
                                    kst = timezone(timedelta(hours=9))
                                    now_time = datetime.now(kst).strftime("%H:%M:%S")

                                    translated = translate_text_local(sentence_buffer)
                                    print(f"✅ 완성 문장: {sentence_buffer}")
                                    print(f"🌐 번역 결과: {translated}\n")

                                    # 🔊 번역된 문장 음성 출력
                                    speak_text(translated, TARGET_LANG)

                                    socketio.emit('partial_translation', {
                                        'original': sentence_buffer.strip(),
                                        'translated': translated,
                                        'time': now_time,
                                        'session_id': session_id
                                    })

                                    insert_transcript(session_id, sentence_buffer.strip(), translated)
                                    sentence_buffer = ""
                                    previous_text = ""

                        except Exception as e:
                            print(f"⚠️ 스트리밍 처리 오류: {e}")
                            traceback.print_exc()
                else:
                    time.sleep(0.01)

    except sd.PortAudioError as e:
        print("❌ 오디오 장치 오류:", e)
    except Exception as e:
        print("❌ 알 수 없는 오류:", e)
        traceback.print_exc()
