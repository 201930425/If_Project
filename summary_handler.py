import torch
import traceback
import threading
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from db_handler import fetch_data_from_db
from config import KOBART_MODEL_NAME
import math

# --- KoBART 모델 상태 변수 ---
kobart_model = None
kobart_tokenizer = None
kobart_loading = False
latest_summary = "[요약은 '요약 보기'를 누르세요]"
DEVICE = "cpu"  # ⭐️ KoBART는 CPU로 실행 (VRAM 부족)


# -----------------------------

def load_kobart_model():
    """KoBART 모델을 메모리로 로드합니다."""
    global kobart_model, kobart_tokenizer, kobart_loading, latest_summary, DEVICE
    if kobart_model is None and not kobart_loading:
        kobart_loading = True
        latest_summary = "[KoBART 모델 로드 중...]"  # 상태 업데이트

        # ⭐️ (VRAM 2GB로는 GPU 가속 실패)
        DEVICE = "cpu"
        print(f"🔄 KoBART 모델 로드 중... (Device: {DEVICE})")

        try:
            kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                KOBART_MODEL_NAME,
                ignore_mismatched_sizes=True
            )
            kobart_model = BartForConditionalGeneration.from_pretrained(
                KOBART_MODEL_NAME,
                ignore_mismatched_sizes=True
            )
            kobart_model.to(DEVICE)  # ⭐️ CPU로 설정

            print(f"✅ KoBART 모델 로드 완료. (Device: {DEVICE})")
            latest_summary = "[모델 로드 완료. 요약 버튼을 다시 눌러주세요]"
            return True
        except Exception as e:
            print(f"⚠️ KoBART 모델 로드 실패: {e}")
            kobart_model = None
            kobart_tokenizer = None
            latest_summary = "[모델 로드 실패. 관리자에게 문의하세요]"
            return False
        finally:
            kobart_loading = False


# ⭐️ 헬퍼 함수: 실제 요약 실행기 (파라미터화)
def _summarize_internal(text_chunk, max_gen_len=150, min_gen_len=30):
    """주어진 텍스트 조각(chunk)을 지정된 길이로 요약합니다."""
    global kobart_tokenizer, kobart_model, DEVICE

    try:
        # <s>, </s> 태그 추가
        text_with_tags = '<s>' + text_chunk + '</s>'

        inputs = kobart_tokenizer(
            text_with_tags,
            return_tensors="pt",
            max_length=1024,  # ⭐️ 모델의 최대 입력 길이
            truncation=True,
            padding="max_length"
        )

        # ⭐️ 입력 텐서를 모델과 동일한 장치로 이동
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        summary_ids = kobart_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=4,
            max_length=max_gen_len,  # ⭐️ 가변 길이 적용
            min_length=min_gen_len,  # ⭐️ 가변 길이 적용
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        summary_raw = kobart_tokenizer.decode(summary_ids[0])
        summary_cleaned = summary_raw.replace('<s>', '').replace('</s>', '').replace('<usr>', '').strip()

        return summary_cleaned

    except Exception as e:
        print(f"⚠️ 요약(내부) 중 오류 발생: {e}")
        traceback.print_exc()
        return "[요약 조각 생성 실패]"


# ⭐️ [수정] Map-Reduce 로직 + 단일 청크 최적화 + 길이 옵션
def summarize_text(text, length_mode="medium"):
    """
    KoBART 모델을 사용하여 텍스트를 요약합니다.
    length_mode: 'short', 'medium', 'long'
    """
    global kobart_tokenizer, kobart_model
    if not text.strip():
        return "[요약할 텍스트가 없습니다]"
    if kobart_model is None or kobart_tokenizer is None:
        load_kobart_model()
        if kobart_model is None:
            return "[KoBART 모델이 로드되지 않았습니다]"

    print(f"🔄 요약 작업 시작... (모드: {length_mode})")

    # ⭐️ 1. 목표 요약 길이 설정
    if length_mode == "short":
        final_max = 100
        final_min = 20
    elif length_mode == "long":
        # A4 용지 1장 목표 (약 1000토큰)
        final_max = 1000
        final_min = 600
    else:  # medium
        final_max = 250
        final_min = 50

    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    if not sentences:
        return "[요약할 텍스트가 없습니다]"

    # ⭐️ 2. Map 단계: 청크화
    max_chunk_tokens = 1000
    current_chunk_sentences = []
    current_chunk_tokens = 0
    intermediate_summaries = []

    print(f" (1/3) 총 {len(sentences)}개 문장 청크화 시작...")

    for sentence in sentences:
        sentence_tokens = len(kobart_tokenizer.tokenize(sentence))

        if current_chunk_tokens + sentence_tokens > max_chunk_tokens:
            # 청크가 꽉 찼으면 '중간 요약' 실행 (Map)
            # 중간 요약은 정보 손실을 막기 위해 적당한 길이(150) 유지
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_summary = _summarize_internal(chunk_text, max_gen_len=150, min_gen_len=30)
                intermediate_summaries.append(chunk_summary)

            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens

    # ⭐️ 3. 마지막 청크 처리 (중요 수정)
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)

        # ⭐️ [핵심 수정] 만약 이것이 '첫 번째이자 마지막' 청크라면 (즉, 전체 텍스트가 한 번에 들어간다면)
        # 중간 요약(150토큰)을 거치지 않고 바로 '최종 목표 길이(final_max)'로 요약합니다.
        if not intermediate_summaries:
            print(" (2/3) 단일 청크 요약 실행 (Reduce 생략)...")

            # ⭐️ 안전 장치: 원문이 너무 짧은데 min_length가 크면 환각(반복) 발생하므로 조절
            input_len = len(kobart_tokenizer.tokenize(chunk_text))
            safe_min = min(final_min, input_len)  # 원문보다 길게 요약하라고 강제하지 않음

            # 여기서 바로 최종 결과 생성
            final_summary_text = _summarize_internal(chunk_text, max_gen_len=final_max, min_gen_len=safe_min)

            # 포맷팅 후 바로 리턴
            final_summary_formatted = final_summary_text.replace(". ", ".\n")
            print("✅ 요약 작업 완료.")
            return final_summary_formatted

        else:
            # 이전 청크들이 있다면 이것도 그냥 중간 요약의 하나일 뿐임
            chunk_summary = _summarize_internal(chunk_text, max_gen_len=150, min_gen_len=30)
            intermediate_summaries.append(chunk_summary)

    if not intermediate_summaries:
        return "[요약 생성 실패]"

    print(f" (2/3) {len(intermediate_summaries)}개 중간 요약 생성 완료.")

    # ⭐️ 4. Reduce 단계: 중간 요약본들을 합쳐서 최종 요약
    combined_summary_text = "\n".join(intermediate_summaries)

    print(" (3/3) 중간 요약본들을 합쳐 최종 요약 중...")
    # Reduce 단계에서도 안전 장치 적용
    input_len = len(kobart_tokenizer.tokenize(combined_summary_text))
    safe_min = min(final_min, input_len)

    final_summary_text = _summarize_internal(combined_summary_text, max_gen_len=final_max, min_gen_len=safe_min)

    # ⭐️ 5. 최종 포맷팅
    final_summary_formatted = final_summary_text.replace(". ", ".\n")
    print("✅ 요약 작업 완료.")

    return final_summary_formatted


def generate_summary_thread(latest_data):
    """(구버전 호환용)"""
    pass