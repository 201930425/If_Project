from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
from datetime import datetime
import config  # ⭐️ config 모듈 임포트
from config import HOST, PORT, LANGUAGE, TARGET_LANG  # ⭐️ 언어 설정 임포트
from db_handler import init_db, get_latest_session_id, fetch_data_from_db, get_all_session_ids, rename_session
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


# --- ⭐️ "요약 창 열기" (최초) 요청 핸들러 ---
@socketio.on("request_summary")
def handle_summary_request(data):
    """
    클라이언트가 요약 팝업을 *처음* 열 때 호출됩니다.
    1. 모든 세션 ID 목록
    2. 가장 최근 세션 ID
    3. 가장 최근 세션의 요약
    """
    print("🔄 (최초) 요약 요청 수신... 모든 세션 목록과 최신 요약을 반환합니다.")
    try:
        all_sessions = get_all_session_ids()
        latest_session_id = None
        summary = "[요약할 세션 데이터가 없습니다]"

        if all_sessions:
            latest_session_id = all_sessions[0]  # 최신 세션이 첫 번째
            full_text = fetch_data_from_db(latest_session_id)
            if full_text:
                print(f"✅ 세션 '{latest_session_id}' 텍스트 요약 중...")
                summary = summarize_text(full_text)
            else:
                summary = "[DB에 요약할 텍스트가 없습니다]"

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


# --- ⭐️ [신규] 🌐 언어 변경 기능 ---
@socketio.on("change_language")
def handle_language_change(data):
    """클라이언트에서 언어 변경 요청을 받음"""
    try:
        # ⭐️ config.py의 전역 변수 값을 직접 수정
        lang = data.get("language", "en")
        target = data.get("target", "ko")  # 목표 언어는 'ko'로 고정

        config.LANGUAGE = lang
        config.TARGET_LANG = target

        print(f"🌐 언어 변경됨 → 입력: {config.LANGUAGE}, 출력: {config.TARGET_LANG}")

        # ⭐️ audio_processor가 config를 다시 참조하도록 알릴 필요는 없음
        # (Python이 모듈을 참조하므로)

        # 클라이언트에 변경 완료를 알림
        socketio.emit("language_changed", {
            "language": config.LANGUAGE,
            "target": config.TARGET_LANG
        })

    except Exception as e:
        print(f"⚠️ 언어 변경 중 오류: {e}")
        socketio.emit("language_changed", {
            "language": "error",
            "target": "error",
            "error": str(e)
        })


# --- ⭐️ "특정 세션" 요약 요청 핸들러 ---
@socketio.on("request_specific_summary")
def handle_specific_summary_request(data):
    """클라이언트가 드롭다운에서 특정 세션을 선택했을 때 호출"""
    session_id = data.get("session_id")
    if not session_id:
        return

    print(f"🔄 (특정) 요약 요청 수신... 세션: {session_id}")
    try:
        full_text = fetch_data_from_db(session_id)
        summary = ""

        if not full_text:
            summary = "[선택된 세션에 요약할 텍스트가 없습니다]"
        else:
            print(f"✅ 세션 '{session_id}' 텍스트 요약 중...")
            summary = summarize_text(full_text)

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


# --- ⭐️ 세션 이름 변경 핸들러 ---
@socketio.on("request_rename_session")
def handle_rename_session(data):
    """클라이언트의 세션 이름 변경 요청을 처리"""
    old_id = data.get('old_id')
    new_id = data.get('new_id')

    if not old_id or not new_id:
        print("⚠️ 이름 변경 요청 오류: old_id 또는 new_id가 없습니다.")
        return

    if old_id == new_id:
        print("⚠️ 이름 변경 무시: 이름이 동일합니다.")
        return

    print(f"🔄 (이름 변경) 요청 수신: '{old_id}' -> '{new_id}'")

    try:
        success = rename_session(old_id, new_id)

        if success:
            all_sessions = get_all_session_ids()
            full_text = fetch_data_from_db(new_id)
            if not full_text:
                summary = "[세션 텍스트를 찾을 수 없습니다]"
            else:
                summary = summarize_text(full_text)

            print("✅ 이름 변경 성공. 클라이언트에 갱신된 데이터 전송.")
            socketio.emit("summary_data_updated", {
                'all_sessions': all_sessions,
                'current_session_id': new_id,
                'summary': summary
            })
        else:
            print("❌ 이름 변경 실패. 기존 데이터로 클라이언트 동기화 시도.")
            all_sessions = get_all_session_ids()
            full_text = fetch_data_from_db(old_id)
            summary = summarize_text(full_text)
            socketio.emit("summary_data_updated", {
                'all_sessions': all_sessions,
                'current_session_id': old_id,
                'summary': summary
            })
    except Exception as e:
        print(f"⚠️ 이름 변경 처리 중 심각한 오류: {e}")


# --- Whisper 자동 세션 시작 ---
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


# --- KoBART 모델 초기화 ---
def init_summary_model():
    """서버 시작 시 KoBART 모델을 미리 로드"""
    print("🧠 KoBART 모델 로드 시도...")
    load_kobart_model()


# --- 메인 실행부 ---
if __name__ == "__main__":
    init_db()
    print("✅ DB 초기화 완료")
    threading.Thread(target=init_summary_model, daemon=True).start()
    print(f"🌍 Socket.IO 서버 시작: http://{HOST}:{PORT} 에서 접속 가능")
    threading.Thread(target=start_auto_session, daemon=True).start()
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)