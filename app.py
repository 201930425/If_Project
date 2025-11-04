from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
from datetime import datetime
from config import HOST, PORT
from db_handler import init_db
from audio_processor import main_audio_streaming

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

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # DB 초기화
    init_db()
    print("✅ DB 초기화 완료")

    # Flask-SocketIO 서버 정보
    print(f"🌍 Socket.IO 서버 시작: http://{HOST}:{PORT} 에서 접속 가능")

    # 서버 실행 직전에 자동 세션 시작
    threading.Thread(target=start_auto_session, daemon=True).start()

    # SocketIO 서버 실행
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
