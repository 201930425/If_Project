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
    global kobart_model, kobart_tokenizer, kobart_loading
    if kobart_model is None and not kobart_loading:
        kobart_loading = True
        print("🔄 KoBART 모델 로드 중... (최초 1회 시간이 걸릴 수 있습니다)")
        try:
            kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(KOBART_MODEL_NAME)
            kobart_model = BartForConditionalGeneration.from_pretrained(
                KOBART_MODEL_NAME,
                ignore_mismatched_sizes=True
            )
            print("✅ KoBART 모델 로드 완료.")
        except Exception as e:
            print(f"⚠️ KoBART 모델 로드 실패: {e}")
            kobart_model = None
            kobart_tokenizer = None
        finally:
            kobart_loading = False


def summarize_text(text, max_len=256):
    """KoBART 모델을 사용하여 텍스트를 요약합니다."""
    global kobart_tokenizer, kobart_model
    if not text.strip():
        return "[요약할 텍스트가 없습니다]"
    if kobart_model is None or kobart_tokenizer is None:
        return "[KoBART 모델이 로드되지 않았습니다]"

    try:
        inputs = kobart_tokenizer(
            text,
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
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        summary = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"⚠️ 요약 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return "[요약 생성 실패]"


def generate_summary_thread(latest_data):
    """현재 세션의 데이터를 가져와 요약하고 전역 변수를 업데이트합니다."""
    global latest_summary
    print("🔄 요약 생성 시작...")
    session_id = latest_data.get("session_id")  # .get()으로 안전하게 접근
    if not session_id:
        latest_summary = "[세션이 시작되지 않아 요약할 수 없습니다]"
        return

    full_text = fetch_data_from_db(session_id)
    if full_text:
        latest_summary = summarize_text(full_text)
        print(f"✅ 요약 생성 완료 (세션: {session_id})")
    elif not full_text:
        latest_summary = "[DB에 요약할 데이터가 없습니다]"

