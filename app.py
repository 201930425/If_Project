from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
from datetime import datetime
from config import HOST, PORT
from db_handler import init_db, get_latest_session_id, fetch_data_from_db
from audio_processor import main_audio_streaming
# ⬇️ 요약 기능에 필요한 모듈 임포트
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


# --- ⭐️ [추가] 요약 요청 처리 핸들러 ---
@socketio.on("request_summary")
def handle_summary_request(data):
    """클라이언트의 요약 요청을 처리하고 결과를 반환합니다."""
    print("🔄 요약 요청 수신...")
    try:
        # 1. DB에서 가장 최근 세션 ID 가져오기
        session_id = get_latest_session_id()
        if not session_id:
            print("⚠️ 요약할 세션이 없습니다.")
            socketio.emit("summary_result", "[요약할 세션 데이터가 없습니다]")
            return

        # 2. 해당 세션의 전체 텍스트 가져오기
        full_text = fetch_data_from_db(session_id)
        if not full_text:
            print("⚠️ 이 세션에 텍스트가 없습니다.")
            socketio.emit("summary_result", "[DB에 요약할 텍스트가 없습니다]")
            return

        # 3. 텍스트 요약 실행
        print(f"✅ 세션 '{session_id}' 텍스트 요약 중...")
        summary = summarize_text(full_text)

        # 4. 클라이언트로 결과 전송
        print("✅ 요약 완료. 클라이언트로 전송.")
        socketio.emit("summary_result", summary)

    except Exception as e:
        print(f"⚠️ 요약 처리 중 오류: {e}")
        socketio.emit("summary_result", f"[요약 생성 실패: {e}]")


# --- Whisper 자동 세션 함수 ---
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


# --- ⭐️ [추가] KoBART 모델 로드 함수 ---
def init_summary_model():
    """서버 시작 시 KoBART 모델을 미리 로드합니다."""
    print("🧠 KoBART 모델 로드 시도...")
    load_kobart_model()


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # DB 초기화
    init_db()
    print("✅ DB 초기화 완료")

    # ⭐️ [추가] KoBART 모델 미리 로드 (별도 스레드)
    threading.Thread(target=init_summary_model, daemon=True).start()

    # Flask-SocketIO 서버 정보
    print(f"🌍 Socket.IO 서버 시작: http://{HOST}:{PORT} 에서 접속 가능")

    # 서버 실행 직전에 자동 세션 시작
    threading.Thread(target=start_auto_session, daemon=True).start()

    # SocketIO 서버 실행
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)