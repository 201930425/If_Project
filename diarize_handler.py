import os
import torch
import whisperx
from deep_translator import GoogleTranslator
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np
import pandas as pd

from config import (
    HF_TOKEN,
    DIARIZE_DEVICE,
    DIARIZE_COMPUTE_TYPE,
    DIARIZE_MODEL_TYPE,
    LANGUAGE, # 실시간 모드와 동일한 언어 사용
    TARGET_LANG
)

# ============================================
# ⚙️ 설정
# ============================================
if HF_TOKEN == "여기에_HUGGINGFACE_TOKEN_붙여넣기":
    print("="*50)
    print("⚠️ [설정 오류] config.py 파일에 HF_TOKEN을 입력해야 합니다.")
    print("="*50)
# ⚠️ [필수] Hugging Face 토큰을 여기에 입력하세요 (https://huggingface.co/settings/tokens)
# pyannote/speaker-diarization-3.1 모델 접근에 필요합니다.

DEVICE = DIARIZE_DEVICE
COMPUTE_TYPE = DIARIZE_COMPUTE_TYPE
MODEL_TYPE = DIARIZE_MODEL_TYPE

# --- 모델 캐시 (전역 변수) ---
# (app.py에서 여러 번 호출할 경우를 대비해 모델을 메모리에 유지)
model_cache = {
    "whisper": None,
    "align": None,
    "diarize": None
}


# ============================================
# 🔄 번역 헬퍼 함수
# ============================================
def translate_text(text, target=TARGET_LANG):
    """
    입력된 텍스트를 목표 언어로 번역합니다.
    (audio_processor.py의 함수와 유사)
    """
    if not text or not text.strip():
        return ""
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        print(f"⚠️ (후처리) 번역 실패: {e}")
        return "[번역 실패]"


# ============================================
# 🚀 모델 로드 함수 (필요시 호출)
# ============================================

def load_whisper_model():
    """WhisperX STT 모델을 로드합니다."""
    if model_cache["whisper"] is None:
        print("🔄 (후처리) WhisperX 모델 로드 중... (CPU, 최초 1회 시간 소요)")
        model_cache["whisper"] = whisperx.load_model(
            MODEL_TYPE,
            DEVICE,
            compute_type=COMPUTE_TYPE,
            language=LANGUAGE
        )
    return model_cache["whisper"]


def load_align_model():
    """WhisperX 정렬 모델을 로드합니다."""
    if model_cache["align"] is None:
        print("🔄 (후처리) 정렬 모델 로드 중... (CPU, 최초 1회 시간 소요)")
        model_a, metadata = whisperx.load_align_model(
            language_code=LANGUAGE,
            device=DEVICE
        )
        model_cache["align"] = (model_a, metadata)
    return model_cache["align"]


def load_diarize_model():
    """Pyannote 화자 분리 모델을 로드합니다."""
    if HF_TOKEN == "여기에_HUGGINGFACE_TOKEN_붙여넣기":
        print("⚠️ [오류] HF_TOKEN이 설정되지 않았습니다. 화자 분리를 스킵합니다.")
        return None

    if model_cache["diarize"] is None:
        print("🔄 (후처리) 화자 분리 모델 로드 중... (CPU, 최초 1회 시간 소요)")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
            pipeline.to(torch.device(DEVICE))
            model_cache["diarize"] = pipeline
        except Exception as e:
            print(f"⚠️ 화자 분리 모델 로드 실패: {e}")
            print("Hugging Face 토큰이 유효한지 확인하세요.")
            return None
    return model_cache["diarize"]


# ============================================
# 🎙️ 메인 분석 함수
# ============================================

def run_diarization(session_id):
    """
    저장된 .wav 파일을 기반으로 화자 분리 및 번역을 수행합니다.
    (CPU에서 실행되므로 매우 느립니다)
    """

    # --- 1. 오디오 파일 확인 ---
    # (audio_processor.py가 이 이름으로 저장했다고 가정)
    audio_file = f"{session_id}.wav"

    if not os.path.exists(audio_file):
        print(f"❌ (후처리) 오디오 파일 없음: {audio_file}")
        return f"[오류] 세션 오디오 파일({audio_file})을 찾을 수 없습니다."

    print(f"✅ (후처리) 세션 '{session_id}' 분석 시작... (CPU 사용, 매우 느릴 수 있음)")

    try:
        # --- 2. 오디오 로드 ---
        # (WhisperX는 16kHz 모노를 기대합니다)
        audio_data, sr = sf.read(audio_file, dtype='float32')
        if audio_data.ndim > 1:  # 스테레오 -> 모노
            audio_data = np.mean(audio_data, axis=1)
        if sr != 16000:
            # (resampling 로직이 필요하지만, audio_processor.py가 16kHz로 저장했다면 생략 가능)
            print(f"⚠️ 경고: 오디오 샘플레이트가 16kHz가 아닙니다. ({sr}Hz)")
            # 간단한 다운샘플링 (권장: librosa 사용)
            if sr > 16000:
                step = sr // 16000
                audio_data = audio_data[::step]

    except Exception as e:
        print(f"❌ (후처리) 오디오 파일 로드 실패: {e}")
        return "[오류] 오디오 파일 로드에 실패했습니다."

    final_transcript = []

    try:
        # --- 3. Whisper STT 실행 ---
        print("🔄 (1/4) 음성 인식(STT) 실행 중...")
        model = load_whisper_model()
        result = model.transcribe(audio_data, batch_size=4)  # CPU 배치 크기

        # --- 4. 정렬 모델 실행 (단어 타임스탬프) ---
        print("🔄 (2/4) 타임스탬프 정렬 중...")
        align_model, metadata = load_align_model()
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio_data,
            DEVICE,
            return_char_alignments=False
        )

        # --- 5. 화자 분리 모델 실행 ---
        print("🔄 (3/4) 화자 분리 실행 중...")
        diarize_model = load_diarize_model()

        if diarize_model is None:
            # 화자 분리 실패 시, 화자 없이 텍스트만 번역
            print("⚠️ (3/4) 화자 분리 모델 로드 실패. 일반 번역으로 대체합니다.")
            for segment in result["segments"]:
                text = segment.get("text", "").strip()
                if text:
                    translated = translate_text(text)
                    final_transcript.append(f"**[내용]**: {text}\n*({translated})*\n")
            return "\n".join(final_transcript)

        # Pyannote가 파일 경로를 필요로 할 수 있으므로 audio_data 대신 audio_file 사용
        diarize_result = diarize_model(audio_file)

        # ⭐️ [신규] pyannote 3.x 호환성 해결 (KeyError: 'e' 수정)
        # whisperx가 Annotation 객체를 잘못 파싱하므로, 수동으로 DataFrame 변환
        print("🔄 (3.5/4) 화자 분리 결과 포맷 변환 중...")
        diarize_segments = []
        for segment, track, speaker in diarize_result.itertracks(yield_label=True):
            diarize_segments.append({
                'start': segment.start,
                'end': segment.end,
                'speaker': speaker
            })

        if not diarize_segments:
            print("⚠️ (후처리) 화자 분리 모델이 아무도 감지하지 못했습니다. 일반 번역으로 대체합니다.")
            # (fallback to no-speaker logic)
            for segment in result["segments"]:
                text = segment.get("text", "").strip()
                if text:
                    translated = translate_text(text)
                    final_transcript.append(f"**[내용]**: {text}\n*({translated})*\n")
            return "\n".join(final_transcript)

        # ⭐️ 수동으로 변환한 DataFrame 생성
        diarize_df = pd.DataFrame(diarize_segments)
        # ⭐️ [신규] 여기까지 ---

        # --- 6. STT 결과와 화자 분리 결과 병합 ---
        print("🔄 (4/4) 화자와 텍스트 병합 중...")
        final_result = whisperx.assign_word_speakers(diarize_df, result)

        # --- 7. 결과 포맷팅 및 번역 ---
        print("✅ 분석 완료. 최종 텍스트 포맷팅 및 번역 중...")

        current_speaker = None
        current_transcript = ""

        for segment in final_result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()

            if not text:
                continue

            if speaker == current_speaker:
                # 같은 화자가 말을 이어가는 경우
                current_transcript += " " + text
            else:
                # 화자가 바뀐 경우 (이전 대화 저장)
                if current_speaker is not None and current_transcript:
                    translated = translate_text(current_transcript)
                    final_transcript.append(f"**{current_speaker}**: {current_transcript}\n*({translated})*\n")

                # 새 화자로 초기화
                current_speaker = speaker
                current_transcript = text

        # 마지막 대화 저장
        if current_speaker is not None and current_transcript:
            translated = translate_text(current_transcript)
            final_transcript.append(f"**{current_speaker}**: {current_transcript}\n*({translated})*\n")

        if not final_transcript:
            return "[분석 결과] 인식된 텍스트가 없습니다."

        return "\n".join(final_transcript)

    except Exception as e:
        print(f"❌ (후처리) 분석 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return f"[오류] 분석 파이프라인 중단: {e}"


# ============================================
# 🧪 테스트용 (직접 실행 시)
# ============================================
if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 테스트할 세션 ID (예: "2025-11-11_11:38.wav" 파일이 있어야 함)
    TEST_SESSION_ID = "diarizeTest"

    if HF_TOKEN == "여기에_HUGGINGFACE_TOKEN_붙여넣기":
        print("=" * 50)
        print("⚠️ 테스트 실패: HF_TOKEN을 `config.py` 파일 상단에 입력하세요.")
        print("=" * 50)
    elif not os.path.exists(f"{TEST_SESSION_ID}.wav"):
        print("=" * 50)
        print(f"⚠️ 테스트 실패: '{TEST_SESSION_ID}.wav' 파일을 찾을 수 없습니다.")
        print("테스트를 위해 오디오 파일을 프로젝트 폴더에 준비해주세요.")
        print("=" * 50)
    else:
        print(f"=== '{TEST_SESSION_ID}' 화자 분리 테스트 시작 ===")
        result = run_diarization(TEST_SESSION_ID)
        print("\n=== ✨ 최종 결과 ===\n")
        print(result)
        print("\n=== 테스트 종료 ===")