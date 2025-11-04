import torch
import traceback
import threading
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from db_handler import fetch_data_from_db
from config import KOBART_MODEL_NAME

# --- KoBART 모델 상태 변수 ---
kobart_model = None
kobart_tokenizer = None
kobart_loading = False
latest_summary = "[요약은 '요약 보기'를 누르세요]"


# -----------------------------

def load_kobart_model():
    """KoBART 모델을 메모리로 로드합니다."""
    global kobart_model, kobart_tokenizer, kobart_loading, latest_summary
    if kobart_model is None and not kobart_loading:
        kobart_loading = True
        latest_summary = "[KoBART 모델 로드 중...]"  # 상태 업데이트
        print("🔄 KoBART 모델 로드 중... (최초 1회 시간이 걸릴 수 있습니다)")
        try:
            # (test_summary.py에서 경고가 발생했던) 깃 오리지널 버전
            kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                KOBART_MODEL_NAME,
                ignore_mismatched_sizes=True
            )
            kobart_model = BartForConditionalGeneration.from_pretrained(
                KOBART_MODEL_NAME,
                ignore_mismatched_sizes=True
            )
            print("✅ KoBART 모델 로드 완료.")
            latest_summary = "[모델 로드 완료. 요약 버튼을 다시 눌러주세요]"  # 상태 업데이트
            return True  # (test_summary.py 호환을 위해 True 반환)
        except Exception as e:
            print(f"⚠️ KoBART 모델 로드 실패: {e}")
            kobart_model = None
            kobart_tokenizer = None
            latest_summary = "[모델 로드 실패. 관리자에게 문의하세요]"  # 상태 업데이트
            return False  # (test_summary.py 호환을 위해 False 반환)
        finally:
            kobart_loading = False


def summarize_text(text, max_len=256):
    """
    (모든 수정 사항이 적용된 버전)
    KoBART 모델을 사용하여 텍스트를 요약합니다.
    """
    global kobart_tokenizer, kobart_model
    if not text.strip():
        return "[요약할 텍스트가 없습니다]"
    if kobart_model is None or kobart_tokenizer is None:
        return "[KoBART 모델이 로드되지 않았습니다]"

    try:
        # (Fix 1: <s>, </s> 태그 추가 - 품질 향상)
        text_with_tags = '<s>' + text + '</s>'

        inputs = kobart_tokenizer(
            text_with_tags,  # 수정
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        summary_ids = kobart_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=4,
            max_length=max_len,
            min_length=60,  # (Fix 2: 최소 길이 강제 - 품질 향상)
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        # (Fix 3: KeyError 수정 및 <usr> 태그 제거)
        summary_raw = kobart_tokenizer.decode(summary_ids[0])
        summary_cleaned = summary_raw.replace('<s>', '').replace('</s>', '').replace('<usr>', '').strip()

        # (New Request: 줄바꿈 추가)
        # 마침표 뒤에 공백이 오는 경우, 마침표 + 줄바꿈으로 변경
        summary_final = summary_cleaned.replace(". ", ".\n")

        return summary_final

    except Exception as e:
        print(f"⚠️ 요약 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return "[요약 생성 실패]"


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

