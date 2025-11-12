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


# ⭐️ [신규] 헬퍼 함수: 실제 요약 실행기
def _summarize_internal(text_chunk):
    """주어진 텍스트 조각(chunk)을 요약합니다."""
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
            max_length=150,  # 중간 요약 최대 길이
            min_length=30,  # 중간 요약 최소 길이
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


# ⭐️ [수정] Map-Reduce 로직이 적용된 메인 요약 함수
def summarize_text(text, max_len=256):  # max_len은 최종 요약본 기준
    """
    KoBART 모델을 사용하여 텍스트를 요약합니다.
    1024 토큰이 넘는 긴 텍스트는 Map-Reduce 방식으로 자동 처리합니다.
    """
    global kobart_tokenizer, kobart_model
    if not text.strip():
        return "[요약할 텍스트가 없습니다]"
    if kobart_model is None or kobart_tokenizer is None:
        load_kobart_model()  # ⭐️ 모델이 없으면 로드 시도
        if kobart_model is None:
            return "[KoBART 모델이 로드되지 않았습니다]"

    print("🔄 요약 작업 시작...")

    # ⭐️ 1. 전체 텍스트를 문장(줄바꿈) 기준으로 분리
    # (db_handler.py가 \n으로 합쳐주기로 함)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    if not sentences:
        return "[요약할 텍스트가 없습니다]"

    # ⭐️ 2. Map 단계: 문장들을 1024 토큰 청크로 묶기
    max_chunk_tokens = 1000  # 1024의 안전 마진
    current_chunk_sentences = []
    current_chunk_tokens = 0
    intermediate_summaries = []

    print(f" (1/3) 총 {len(sentences)}개 문장 청크화 시작...")

    for sentence in sentences:
        # 현재 문장의 토큰 수 계산
        sentence_tokens = len(kobart_tokenizer.tokenize(sentence))

        if current_chunk_tokens + sentence_tokens > max_chunk_tokens:
            # ⭐️ 토큰 한도 초과: 현재까지의 청크를 요약
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                print(f"  ... 청크 요약 중 (토큰 약 {current_chunk_tokens}개)")
                chunk_summary = _summarize_internal(chunk_text)
                intermediate_summaries.append(chunk_summary)

            # 새 청크 시작
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            # ⭐️ 토큰 한도 미만: 현재 청크에 문장 추가
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens

    # ⭐️ 마지막 남은 청크 요약
    if current_chunk_sentences:
        print(f"  ... 마지막 청크 요약 중 (토큰 약 {current_chunk_tokens}개)")
        chunk_text = " ".join(current_chunk_sentences)
        chunk_summary = _summarize_internal(chunk_text)
        intermediate_summaries.append(chunk_summary)

    if not intermediate_summaries:
        return "[요약 생성 실패]"

    print(f" (2/3) {len(intermediate_summaries)}개 중간 요약 생성 완료.")

    # ⭐️ 3. Reduce 단계: 중간 요약본들을 합쳐서 최종 요약
    combined_summary_text = "\n".join(intermediate_summaries)

    # ⭐️ 만약 중간 요약이 1개 뿐이면 (텍스트가 1024 토큰 미만이었으면)
    if len(intermediate_summaries) == 1:
        final_summary_text = intermediate_summaries[0]
    else:
        # ⭐️ 중간 요약본들의 합이 1024 토큰을 넘으면, 최종 요약도 잘릴 수 있지만
        # (이 경우 재귀적으로 처리해야 하나, CPU 부담으로 1회로 제한)
        print(" (3/3) 중간 요약본들을 합쳐 최종 요약 중...")
        final_summary_text = _summarize_internal(combined_summary_text)

    # ⭐️ 4. 최종 포맷팅 (줄바꿈 추가)
    final_summary_formatted = final_summary_text.replace(". ", ".\n")
    print("✅ 요약 작업 완료.")

    return final_summary_formatted


def generate_summary_thread(latest_data):
    """
    (깃 오리지널 버전)
    *현재* 세션의 데이터를 가져와 요약하고 전역 변수를 업데이트합니다.
    """
    global latest_summary
    print("🔄 요약 생성 시작...")
    latest_summary = "[요약 생성 중...]"  # 상태 업데이트

    session_id = latest_data.get("session_id")  # .get()으로 안전하게 접근
    if not session_id:
        latest_summary = "[요약할 세션 ID가 없습니다]"
        return

    full_text = fetch_data_from_db(session_id)  # *현재* 세션 ID로 조회
    if full_text:
        latest_summary = summarize_text(full_text)
        print(f"✅ 요약 생성 완료 (세션: {session_id})")
    elif not full_text:
        latest_summary = "[DB에 요약할 데이터가 없습니다]"