from db_handler import fetch_data_from_db, get_latest_session_id
from summary_handler import load_kobart_model, summarize_text
from config import KOBART_MODEL_NAME

def run_test():
    """
    DB에서 '가장 최근' 세션 데이터를 가져와 KOBART 요약을 테스트합니다.
    """
    print(f"=== KOBART 요약 테스트 시작 ===")
    print(f"사용 중인 모델: {KOBART_MODEL_NAME}\n")

    # 1. 모델 로드
    success = load_kobart_model()

    # 모델 로드 실패 시 스크립트 즉시 중지
    if not success:
        print("\n❌ 모델 로드에 실패하여 테스트를 중단합니다.")
        print("모델 이름을 확인하거나 인터넷 연결을 확인하세요.")
        print("==============================")
        return

    # 2. DB에서 *가장 최근* 세션 ID 가져오기
    print("\nDB에서 가장 최근 세션 ID를 찾는 중...")
    session_id = get_latest_session_id()

    if not session_id:
        print("\n⚠️ 오류: DB에 저장된 세션이 없습니다.")
        print("먼저 app.py를 실행하여 음성 인식을 진행해 주세요.")
        print("==============================")
        return

    print(f"가장 최근 세션 ID '{session_id}'의 데이터를 불러오는 중...")
    full_text = fetch_data_from_db(session_id)

    if not full_text:
        print("\n⚠️ 오류: 해당 세션에 요약할 텍스트가 없습니다.")
        print("==============================")
        return

    print(f"총 {len(full_text)}자 텍스트 로드 완료.")

    # 3. 요약 실행
    print("\n--- [ 원본 텍스트 (일부) ] ---")
    print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
    print("----------------------------")

    print(f"\n🔄 요약 생성 중... (모델: {KOBART_MODEL_NAME})")
    summary = summarize_text(full_text)

    # 4. 결과 출력
    print("\n--- [ ✨ 최종 요약 결과 ] ---")
    print(summary)
    print("----------------------------")
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    run_test()

