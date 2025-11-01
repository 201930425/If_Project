from flask import Flask, render_template
from flask_socketio import SocketIO  # 1. SocketIO 임포트
import threading
from datetime import datetime

# --- 로컬 모듈 임포트 ---
from config import HOST, PORT
from db_handler import init_db
from audio_processor import main_audio_loop
# 2. summary_handler 임포트 제거 (새 HTML이 사용 안 함)
# -------------------------

# --- Flask 앱 및 전역 변수 초기화 ---
app = Flask(__name__)
# 3. SocketIO로 앱 초기화
socketio = SocketIO(app, cors_allowed_origins="*")

# 4. latest_data 및 summary_mode 전역 변수 제거
# (SocketIO가 실시간으로 데이터를 밀어주므로 필요 없음)

# ---------------------------------

# --- Flask 라우트 ---
@app.route("/")
def index():
    """메인 HTML 페이지를 렌더링합니다."""
    # 5. 렌더링할 템플릿 이름 변경
    return render_template("translation.html")

# 6. /subtitle 라우트 제거
# 7. /toggle_summary 라우트 제거

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    init_db()  # DB 초기화
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8. 오디오 스레드에 'socketio' 객체를 넘겨줌
    audio_thread = threading.Thread(
        target=main_audio_loop,
        args=(session_id, socketio,),  # latest_data 대신 socketio 전달
        daemon=True
    )
    audio_thread.start()

    # 9. app.run() 대신 socketio.run()으로 서버 실행
    print(f"🌍 Socket.IO 서버 시작: http://{HOST}:{PORT} 에서 확인하세요")
    # allow_unsafe_werkzeug=True는 PyCharm 같은 환경에서 필요할 수 있습니다.
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)

