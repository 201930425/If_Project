from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
from datetime import datetime
from config import HOST, PORT
# ⬇️ get_all_session_ids 임포트 추가
from db_handler import init_db, get_latest_session_id, fetch_data_from_db, get_all_session_ids
from audio_processor import main_audio_streaming
from summary_handler import load_kobart_model, summarize_text

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# --- Flask 라우트 ---
@app.route("/")
def index():
    return render_template("translation.html")


# --- 클라이언트 연결/해제 로그 ---
@socketio.on("connect")
def handle_connect():
    print("✅ 클라이언트 연결됨 (웹 브라우저 접속 확인)")


@socketio.on("disconnect")
def handle_disconnect():
    print("❌ 클라이언트 연결 해제됨")


# --- ⭐️ [수정] "요약 창 열기" (최초) 요청 핸들러 ---
@socketio.on("request_summary")
def handle_summary_request(data):
    """
    (수정) 클라이언트가 요약 팝업을 *처음* 열 때 호출됩니다.
    1. 모든 세션 ID 목록
    2. 가장 최근 세션 ID
    3. 가장 최근 세션의 요약
    위 3가지를 모두 전송합니다.
    """
    print("🔄 (최초) 요약 요청 수신... 모든 세션 목록과 최신 요약을 반환합니다.")
    try:
        all_sessions = get_all_session_ids()
        latest_session_id = None
        summary = "[요약할 세션 데이터가 없습니다]"

        if all_sessions:
            latest_session_id = all_sessions[0]  # 목록의 첫 번째가 최신
            full_text = fetch_data_from_db(latest_session_id)
            if full_text:
                print(f"✅ 세션 '{latest_session_id}' 텍스트 요약 중...")
                summary = summarize_text(full_text)
            else:
                summary = "[DB에 요약할 텍스트가 없습니다]"

        # ⭐️ 클라이언트로 3가지 데이터를 모두 전송
        socketio.emit("summary_data_updated", {
            'all_sessions': all_sessions,
            'current_session_id': latest_session_id,
            'summary': summary
        })

    except Exception as e:
        print(f"⚠️ 최초 요약 처리 중 오류: {e}")
        socketio.emit("summary_data_updated", {
            'all_sessions': [],
            'current_session_id': None,
            'summary': f"[요약 생성 실패: {e}]"
        })


# --- ⭐️ [신규] "특정 세션" 요약 요청 핸들러 ---
@socketio.on("request_specific_summary")
def handle_specific_summary_request(data):
    """
    (신규) 클라이언트가 드롭다운에서 특정 세션을 선택했을 때 호출됩니다.
    """
    session_id = data.get("session_id")
    if not session_id:
        return  # 무시

    print(f"🔄 (특정) 요약 요청 수신... 세션: {session_id}")
    try:
        full_text = fetch_data_from_db(session_id)
        summary = ""

        if not full_text:
            summary = "[선택된 세션에 요약할 텍스트가 없습니다]"
        else:
            print(f"✅ 세션 '{session_id}' 텍스트 요약 중...")
            summary = summarize_text(full_text)

        # ⭐️ 클라이언트로 '현재 세션'과 '요약'만 업데이트
        # (all_sessions는 보낼 필요 없음. 클라이언트가 이미 갖고 있음)
        socketio.emit("summary_data_updated", {
            'current_session_id': session_id,
            'summary': summary
        })

    except Exception as e:
        print(f"⚠️ 특정 세션 요약 처리 중 오류: {e}")
        socketio.emit("summary_data_updated", {
            'current_session_id': session_id,
            'summary': f"[요약 생성 실패: {e}]"
        })


# --- Whisper 자동 세션 함수 ---
# ... (start_auto_session, init_summary_model, if __name__ == "__main__": 블록은 그대로 둠) ...
def start_auto_session():
    """서버 실행 시 자동으로 Whisper 스트리밍을 시작"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n🎬 [자동 세션 시작] 세션 ID: {session_id}\n")

    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=main_audio_streaming,
        args=(session_id, socketio, stop_event),
        daemon=True
    )
    audio_thread.start()
    print("🎤 Whisper 실시간 음성 인식 스레드 시작됨 ✅")


def init_summary_model():
    """서버 시작 시 KoBART 모델을 미리 로드합니다."""
    print("🧠 KoBART 모델 로드 시도...")
    load_kobart_model()


if __name__ == "__main__":
    init_db()
    print("✅ DB 초기화 완료")
    threading.Thread(target=init_summary_model, daemon=True).start()
    print(f"🌍 Socket.IO 서버 시작: http://{HOST}:{PORT} 에서 접속 가능")
    threading.Thread(target=start_auto_session, daemon=True).start()
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)