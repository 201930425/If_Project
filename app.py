from flask import Flask, render_template, jsonify
import threading
from datetime import datetime

# --- 로컬 모듈 임포트 ---
from config import HOST, PORT
from db_handler import init_db
# (OBS 관련 'utils' 임포트 제거)
from audio_processor import main_audio_loop
from summary_handler import (
    latest_summary as global_latest_summary,
    kobart_model,
    kobart_loading,
    load_kobart_model,
    generate_summary_thread
)

# -------------------------

# --- Flask 앱 및 전역 변수 초기화 ---
app = Flask(__name__)

# 실시간 번역 데이터 (오디오 스레드가 여기 씀)
latest_data = {
    "session_id": "",
    "original": "[대기 중...]",
    "translated": "[Waiting...]"
}
# 요약 모드 (메인 스레드가 관리)
summary_mode = False


# ---------------------------------


# --- Flask 라우트 (웹페이지 및 API) ---
@app.route("/")
def index():
    """메인 HTML 페이지를 렌더링합니다."""
    return render_template("index.html")


@app.route("/subtitle")
def get_subtitle():
    """최신 번역 또는 요약 데이터를 JSON으로 반환합니다."""
    global latest_data, summary_mode

    if summary_mode:
        # 요약 모드일 때
        return jsonify({
            "original": f"--- 요약 모드 (세션: {latest_data.get('session_id', 'N/A')}) ---",
            "translated": global_latest_summary,  # summary_handler의 변수 사용
            "mode": "summary"
        })
    else:
        # 일반 번역 모드일 때
        return jsonify({
            "original": latest_data["original"],
            "translated": latest_data["translated"],
            "mode": "full"
        })


@app.route("/toggle_summary")
def toggle_summary():
    """요약 모드를 토글하고, 필요시 모델 로드/요약 생성을 스레드로 시작합니다."""
    global summary_mode, latest_data

    summary_mode = not summary_mode

    if summary_mode:
        # summary_handler의 상태 변수 사용
        if kobart_model is None and not kobart_loading:
            # 모델 로드 스레드 시작
            threading.Thread(target=load_kobart_model, daemon=True).start()
            # summary_handler.latest_summary = "[KoBART 모델 로드 중... 잠시 후 다시 눌러주세요]"

        elif kobart_model is not None:
            # 요약 스레드 시작
            # summary_handler.latest_summary = "[요약 생성 중...]"
            threading.Thread(target=generate_summary_thread, args=(latest_data,), daemon=True).start()

        elif kobart_loading:
            # 모델이 로드 중일 때는 아무것도 하지 않음 (메시지는 이미 설정됨)
            pass

    return jsonify({"mode": "summary" if summary_mode else "full"})


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    init_db()  # DB 초기화

    # (OBS 파일 초기화 코드 제거됨)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 오디오 스레드 시작 (latest_data 딕셔너리를 넘겨줌)
    audio_thread = threading.Thread(
        target=main_audio_loop,
        args=(session_id, latest_data,),
        daemon=True
    )
    audio_thread.start()

    # Flask 웹 서버 시작 (메인 스레드)
    print(f"🌍 웹 서버 시작: http://{HOST}:{PORT} 에서 확인하세요")
    # debug=False로 설정해야 KoBART 모델 로딩이 두 번 실행되지 않습니다.
    app.run(host=HOST, port=PORT, debug=False)

