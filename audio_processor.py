# # ⭐️ [신규] CUDA DLL 경로를 스크립트 최상단에 직접 추가 #gpu사용시
# import os
#
# # 1. CUDA Toolkit 경로 (기존)
# cuda_toolkit_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
# # ⭐️ 2. cuDNN 경로 (새로 찾은 정확한 경로)
# cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.15\bin\12.9"
#
# # ⭐️ [수정] 2개의 경로를 모두 리스트로 관리
# paths_to_add = [cuda_toolkit_path, cudnn_path]
#
# for path in paths_to_add:
#     # ⭐️ 3. os.environ["PATH"]에 수동 추가 (MINGW64 호환성)
#     try:
#         if path and os.path.exists(path) and path not in os.environ.get("PATH", ""):
#             print(f"✅ (최상단) os.environ['PATH']에 경로 추가: {path}")
#             os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
#         elif not os.path.exists(path):
#              print(f"⚠️ (최상단) 경고: 경로를 찾을 수 없습니다: {path}")
#     except Exception as e:
#         print(f"⚠️ (최상단) PATH 환경 변수 설정 실패 (무시): {e}")
#
#     # ⭐️ 4. (기존) os.add_dll_directory 사용 (Python 3.8+ 권장 방식)
#     try:
#         if path and os.path.exists(path):
#             print(f"✅ (최상단) os.add_dll_directory로 경로 추가: {path}")
#             os.add_dll_directory(path)
#     except Exception as e:
#         print(f"⚠️ (최상단) DLL 경로 추가 실패 (무시): {e}")
# # ⭐️ [신규] 여기까지 수정 ---
import os
import sounddevice as sd
import numpy as np
import queue
import time
# import webrtcvad # ❌ (제거)
import torch  # ⭐️ [추가] Silero VAD에 필요
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from datetime import datetime, timezone, timedelta
import traceback
import noisereduce as nr
import wave

import config
from db_handler import insert_transcript
from config import (
    MODEL_TYPE, LANGUAGE, TARGET_LANG,
    BEAM_SIZE, INPUT_DEVICE_INDEX, FRAME_SIZE,
    VAD_THRESHOLD, RATE, CHUNK_DURATION_SEC, CHUNK_SIZE
)

print(f"🎧 Whisper 모델({MODEL_TYPE}) 로드 중...")
# ⭐️ [수정] float16 -> int8_float16 (GTX 1050 호환 모드)
# model = WhisperModel(MODEL_TYPE, device="cuda", compute_type="int8") #gpu사용시
# print("✅ Whisper 모델 로드 완료 (Device: CUDA/GPU)") #gpu사용시
model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
print("✅ Whisper 모델 로드 완료 (Device: CPU)")


# ⭐️ Silero VAD 모델 로드
print("🎧 Silero VAD 모델 로드 중... (torch 필요)")
try:
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=True)
    # ⭐️ [제거] 이 라인을 삭제하거나 주석 처리하세요.
    # vad_model.to("cuda")
    print("✅ Silero VAD (ONNX) 모델 로드 완료 (Device: CPU).") # ⭐️ 로그 수정
except Exception as e:
    print(f"⚠️ Silero VAD 모델 로드 실패: {e}")
    print("torch, torchaudio가 설치되었는지, 인터넷 연결이 되어있는지 확인하세요.")
    vad_model = None

# ❌ (제거) vad = webrtcvad.Vad(VAD_MODE)
audio_q = queue.Queue()


# --- 번역 ---
def translate_text_local(text, target_lang=TARGET_LANG):
    if not text or not text.strip():
        return ""

    # ⭐ 한국어 탭 → 번역 필요 없음
    if target_lang == config.LANGUAGE:
        return ""

    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"⚠️ 번역 실패: {e}")
        return ""


# --- 문장 완성 감지 (구두점 + 무음 기반) ---
def is_sentence_complete(text):
    """문장이 끝났는지 판별"""
    if not text.strip():
        return False
    text = text.strip()
    endings = (
        ".", "!", "?",  # 영어
        "요", "다", "죠", "네", "습니다",  # 한국어
        "。", "です", "ます", "ね", "か"  # ⭐️ 일본어 추가
    )
    return text.endswith(endings)


# ⭐️ [신규] Silero VAD용 헬퍼 함수 (버그 수정)
def is_chunk_speech(data_chunk, vad_model, rate=RATE, frame_size=FRAME_SIZE, threshold=VAD_THRESHOLD):
    """
    긴 오디오 청크(data_chunk)를 VAD_FRAME_SIZE(512) 샘플 크기로 나누어
    하나라도 음성으로 감지되면 True를 반환합니다.
    """
    if not vad_model:
        print("⚠️ VAD 모델이 없어 음성으로 간주합니다.")
        return True  # VAD 모델 로드 실패 시 무조건 음성으로 처리

    # data_chunk는 (N, 1) 형태일 수 있으므로 flatten()
    data_flat = data_chunk.flatten()

    # 512 샘플(VAD_FRAME_SIZE) 단위로 반복
    for i in range(0, len(data_flat), frame_size):
        frame = data_flat[i: i + frame_size]

        # ⭐️ (중요) 마지막 프레임이 512보다 작으면 VAD가 오류를 일으킴
        if len(frame) < frame_size:
            continue

        # 1. int16 numpy -> float32 tensor
        audio_float32_tensor = torch.from_numpy(frame.astype(np.float32) / 32768.0)

        # 2. VAD 모델 실행 (음성 확률 반환)
        try:
            # ⭐️ .item()은 tensor(0.xx) -> 0.xx (float)로 변환
            speech_prob = vad_model(audio_float32_tensor, rate).item()

            # 3. 임계값과 비교
            if speech_prob >= threshold:
                return True  # 음성 감지됨
        except Exception as e:
            # CHUNK_SIZE가 512의 배수가 아닌 경우 등 예외 처리
            # print(f"VAD 프레임 처리 오류 (무시): {e}")
            pass

            # 루프가 끝날 때까지 음성이 감지되지 않음
    return False


# --- 오디오 콜백 ---
def audio_callback(indata, frames, time_, status):
    if status:
        print(f"[Audio status] {status}")
    try:
        audio_q.put(indata.copy())
    except Exception:
        traceback.print_exc()


# --- ❌ (제거) 'is_speech_chunk' (webrtcvad) ---


# --- 메인 루프 ---
def main_audio_streaming(session_id, socketio, stop_event=None):
    print(f"🗂️ 세션 시작 (스트리밍 모드): {session_id}")

    # ⭐️ [신규] .wav 파일 쓰기 준비
    output_dir = "wav"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 출력 디렉토리 생성: {output_dir}")

    wave_file = None
    wave_file_name = os.path.join(output_dir, f"{session_id}.wav") # wav/session_id.wav
    try:
        wave_file = wave.open(wave_file_name, 'wb')
        wave_file.setnchannels(1)  # 모노 (1 채널)
        wave_file.setsampwidth(2)  # 2바이트 (int16)
        wave_file.setframerate(RATE)  # 16000
        print(f"🌊 오디오 파일 녹음 시작: {wave_file_name}")
    except Exception as e:
        print(f"⚠️ [오류] {wave_file_name} 파일 생성 실패: {e}")
        wave_file = None  # 파일 쓰기 비활성화

    buffer = np.zeros((0, 1), dtype=np.int16)
    sentence_buffer = ""
    previous_text = ""
    last_emit_time = time.time()
    silence_counter = 0

    try:
        with sd.InputStream(
                device=INPUT_DEVICE_INDEX,
                samplerate=RATE,
                blocksize=FRAME_SIZE,  # ⭐️ config.py에서 512로 변경됨
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
                    # ⭐️ [신규] 1. 오디오 조각을 .wav 파일에 저장
                    if wave_file:
                        try:
                            wave_file.writeframes(block.tobytes())
                        except Exception as e:
                            print(f"⚠️ [오류] {wave_file_name} 파일 쓰기 중단: {e}")
                            wave_file.close()  # 오류 발생 시 파일 닫기
                            wave_file = None  # 더 이상 쓰지 않음

                    buffer = np.concatenate((buffer, block), axis=0)

                    if len(buffer) >= CHUNK_SIZE:
                        data_chunk = buffer[:CHUNK_SIZE]
                        buffer = buffer[CHUNK_SIZE:]

                        try:
                            # 🔉 [수정] 음성 감지 (Silero VAD 헬퍼 함수 사용)
                            # ⭐️ data_chunk(48000)를 헬퍼 함수로 전달
                            if not is_chunk_speech(data_chunk, vad_model, RATE, FRAME_SIZE, VAD_THRESHOLD):
                                # (무음으로 간주 - 기존 'if not is_speech_chunk' 로직)
                                silence_counter += 1
                                if silence_counter >= 2 and sentence_buffer.strip():
                                    # ✅ 1.5초 이상 무음 → 문장 완료로 간주
                                    kst = timezone(timedelta(hours=9))
                                    now_time = datetime.now(kst).strftime("%H:%M:%S")

                                    translated = translate_text_local(sentence_buffer, target_lang=config.TARGET_LANG)
                                    print(f"✅ 완성 문장: {sentence_buffer}")
                                    print(f"🌐 번역 결과: {translated}\n")

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

                                # ⭐️ (중요) 원본 로직과 동일하게, 음성이 아니면 번역/인식 스킵
                                continue

                            else:
                                # (음성으로 간주 - 기존 'else' 블록)
                                silence_counter = 0

                                # 🔉 노이즈 제거
                                reduced = nr.reduce_noise(y=data_chunk.flatten(), sr=RATE)

                                # ⭐️ [수정] 0으로 나누기 오류(RuntimeWarning) 방지
                                max_val = np.max(np.abs(reduced))

                                if max_val > 0:
                                    # 신호가 있을 때만 정규화
                                    normalized_audio = reduced / max_val
                                else:
                                    # 완전한 무음인 경우 (max_val == 0)
                                    normalized_audio = reduced  # (이미 0으로 채워진 배열)

                                reduced_int16 = np.int16(normalized_audio * 32767)
                                audio_float32 = reduced_int16.astype(np.float32) / 32768.0

                            # 🧠 Whisper 인식 (무음이 아닐 때만 이쪽으로 넘어옴)
                            segments, _ = model.transcribe(
                                audio_float32,
                                language=config.LANGUAGE,
                                beam_size=BEAM_SIZE,
                                # --- ⭐️ 환각(쓰레기값) 억제 옵션 추가 ---
                                vad_filter=True,  # VAD 필터를 사용해 음성이 없는 세그먼트를 제거
                                no_speech_threshold=0.4,  # 이 값 이하의 '음성 확률'은 무시
                                log_prob_threshold=-1.0,  # 신뢰도가 너무 낮은 토큰(단어)을 억제
                                condition_on_previous_text=False  # 이전 텍스트에 덜 의존하여 반복 환각을 줄임
                            )
                            partial_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

                            if partial_text and partial_text != previous_text:
                                # ✅ 새로 추가된 부분만 추출
                                new_part = partial_text.replace(previous_text, "").strip()
                                if new_part:
                                    sentence_buffer += " " + new_part
                                    previous_text = partial_text
                                    print(f"🧩 부분 인식 누적: {new_part}")

                                # 종결어미 기반 문장 완성 감지
                                if is_sentence_complete(sentence_buffer):
                                    kst = timezone(timedelta(hours=9))
                                    now_time = datetime.now(kst).strftime("%H:%M:%S")

                                    translated = translate_text_local(sentence_buffer, target_lang=config.TARGET_LANG)
                                    print(f"✅ 완성 문장: {sentence_buffer}")
                                    print(f"🌐 번역 결과: {translated}\n")

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
    finally:
        # ⭐️ [신규] 세션이 끝나면 .wav 파일 닫기
        if wave_file:
            wave_file.close()
            print(f"🌊 오디오 파일 저장 완료: {wave_file_name}")