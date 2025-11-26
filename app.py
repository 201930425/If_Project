from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
from datetime import datetime
import config  # ⭐️ config 모듈 임포트
from config import HOST, PORT, LANGUAGE, TARGET_LANG  # ⭐️ 언어 설정 임포트
from db_handler import init_db, get_latest_session_id, fetch_data_from_db, get_all_session_ids, rename_session, \
    delete_session  # ⭐️ delete_session 임포트
from audio_processor import main_audio_streaming, audio_q
import queue
from summary_handler import load_kobart_model, summarize_text
import os  # ⭐️ [신규] .wav 파일 삭제를 위해 임포트

# ⭐️ [신규] diarize_handler 임포트
import diarize_handler

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- ⭐️ [수정] 오디오 스레드 관리를 위한 전역 변수 ---
current_audio_thread = None
current_stop_event = None

# ⭐️ [신규] 화자 분리 스레드 관리를 위한 전역 변수
current_diarize_thread = None


# ----------------------------------------------------


# --- Flask 라우트 ---
@app.route("/")
def index():
    return render_template("translation.html")


# --- 클라이언트 연결/해제 로그 ---
@socketio.on("connect")
def handle_connect():
    print("✅ 클라이언트 연결됨 (웹 브저 접속 확인)")


@socketio.on("disconnect")
def handle_disconnect():
    print("❌ 클라이언트 연결 해제됨")


# --- ⭐️ [신규] "세션 목록" (최초) 요청 핸들러 ---
@socketio.on("request_session_list")
def handle_session_list_request(data):  # ⭐️ (data) 인자 유지
    """
    클라이언트가 메인 페이지를 로드할 때 호출됩니다.
    1. 모든 세션 ID 목록
    2. 가장 최근 세션 ID
    """
    print("🔄 (최초) 세션 목록 요청 수신...")
    try:
        all_sessions = get_all_session_ids()
        latest_session_id = None
        if all_sessions:
            latest_session_id = all_sessions[0]  # 최신 세션이 첫 번째

        socketio.emit("session_list_updated", {
            'all_sessions': all_sessions,
            'latest_session': latest_session_id
        })
    except Exception as e:
        print(f"⚠️ 최초 세션 목록 전송 중 오류: {e}")
        socketio.emit("session_list_updated", {
            'all_sessions': [],
            'latest_session': None
        })


# --- ⭐️ [신규] 🌐 언어 변경 기능 ---
@socketio.on("change_language")
def handle_language_change(data):
    """클라이언트에서 언어 변경 요청을 받음"""
    try:
        lang = data.get("language")
        target = data.get("target")
        config.LANGUAGE = lang
        config.TARGET_LANG = target
        print(f"🌐 언어 변경됨 → 입력: {config.LANGUAGE}, 출력: {config.TARGET_LANG}")
        socketio.emit("language_changed", {
            "language": lang,
            "target": target
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
    """(수정) 클라이언트가 '요약' 버튼을 눌렀을 때 호출"""
    session_id = data.get("session_id")
    if not session_id:
        return

    print(f"🔄 (특정) 요약 요청 수신... 세션: {session_id}")

    # ⭐️ [신규] 요약은 CPU 시간이 걸리므로 별도 스레드로 분리
    threading.Thread(
        target=run_summary_thread,
        args=(session_id,),
        daemon=True
    ).start()


# ⭐️ [신규] 요약을 위한 스레드 함수
def run_summary_thread(session_id):
    """
    (백그라운드 스레드)
    summary_handler.py를 실행하고, 완료되면 팝업창으로 결과를 전송합니다.
    """
    try:
        full_text = fetch_data_from_db(session_id)
        summary = ""

        if not full_text:
            summary = "[선택된 세션에 요약할 텍스트가 없습니다]"
        else:
            print(f"✅ (스레드) 세션 '{session_id}' 텍스트 요약 중...")
            # ⭐️ [수정] Map-Reduce 요약 함수 호출 (오래 걸릴 수 있음)
            summary = summarize_text(full_text)

        # ⭐️ 팝업창 전용 이벤트로 전송
        socketio.emit("summary_data_updated", {
            'current_session_id': session_id,
            'summary': summary
        })

    except Exception as e:
        print(f"⚠️ (스레드) 특정 세션 요약 처리 중 오류: {e}")
        socketio.emit("summary_data_updated", {
            'current_session_id': session_id,
            'summary': f"[요약 생성 실패: {e}]"
        })


# --- ⭐️ 세션 이름 변경 핸들러 (사용자 HTML에서 제거됨) ---
# (참고: 이 핸들러는 translation.html에서 제거되었으므로 호출되지 않습니다)
@socketio.on("request_rename_session")
def handle_rename_session(data):
    """클라이언트의 세션 이름 변경 요청을 처리"""
    old_id = data.get('old_id')
    new_id = data.get('new_id')

    if not old_id or not new_id:
        print("⚠️ 이름 변경 요청 오류: old_id 또는 new_id가 없습니다.")
        return
    # ... (이하 로직은 생략, 필요시 복원) ...


# --- ⭐️ [신규] 세션 *삭제* 핸들러 ---
@socketio.on("request_delete_session")
def handle_delete_session(data):
    """클라이언트의 세션 삭제 요청을 처리"""
    global current_audio_thread
    session_id = data.get('session_id')

    if not session_id:
        print("⚠️ 세션 삭제 거부: 세션 ID가 없습니다.")
        return

    # ⭐️ [안전 장치] 실시간 번역 세션이 실행 중인지 확인
    if current_audio_thread is not None and current_audio_thread.is_alive():
        print("⚠️ 세션 삭제 거부: 실시간 번역 세션이 실행 중입니다.")
        # (클라이언트 측에서 이미 방지했지만, 서버에서도 한 번 더 확인)
        return

    print(f"🔄 (세션 삭제) 요청 수신: '{session_id}'")

    try:
        # 1. DB에서 삭제
        db_success = delete_session(session_id)

        # 2. wav/ 폴더에서 .wav 파일 삭제
        wav_file_path = os.path.join("wav", f"{session_id}.wav")
        file_success = False
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)
            print(f"✅ .wav 파일 삭제 완료: {wav_file_path}")
            file_success = True
        else:
            print(f"⚠️ .wav 파일 없음 (무시): {wav_file_path}")
            file_success = True  # 파일이 없어도 DB는 삭제되어야 하므로 성공으로 간주

        # 3. (중요) 모든 클라이언트의 세션 목록 갱신
        if db_success or file_success:
            all_sessions = get_all_session_ids()
            latest_session = all_sessions[0] if all_sessions else None

            socketio.emit("session_list_updated", {
                'all_sessions': all_sessions,
                'latest_session': latest_session  # 가장 최신 세션을 선택
            })
            print("✅ 세션 삭제 완료. 클라이언트 목록 갱신.")

    except Exception as e:
        print(f"⚠️ 세션 삭제 처리 중 심각한 오류: {e}")


# --- ⭐️ [신규] 번역 세션 시작 요청 핸들러 ---
@socketio.on("start_translation_session")
def handle_start_session(data):
    """
    클라이언트가 "시작" 버튼을 누르고 세션 ID를 입력했을 때 호출됩니다.
    """
    session_id = data.get("session_id")
    if not session_id or not session_id.strip():
        print("⚠️ [Session] 세션 ID가 없이 시작 요청을 받았습니다.")
        socketio.emit("session_start_failed", {"error": "세션 이름이 필요합니다."})
        return

    print(f"🔄 (세션 시작) 요청 수신... ID: {session_id}")
    start_new_audio_session(session_id)


# --- ⭐️ [신규] 번역 세션 중지 요청 핸들러 ---
@socketio.on("stop_translation_session")
def handle_stop_session(data):
    """클라이언트가 "중지" 버튼을 눌렀을 때 현재 세션을 중지시킵니다."""
    print("🔄 (세션 중지) 요청 수신...")
    stop_audio_session(notify_client=True)


# ⭐️ [수정] "화자 분리" 요청 핸들러 (메인 페이지 버튼용)
@socketio.on("request_diarization")
def handle_diarization_request(data):
    """
    클라이언트가 *메인 페이지*에서 "화자 분리" 버튼을 눌렀을 때 호출됩니다.
    """
    global current_audio_thread, current_diarize_thread

    session_id = data.get("session_id")
    if not session_id:
        print("⚠️ 화자 분리 거부: 세션 ID가 없습니다.")
        socketio.emit("diarization_result", {
            'session_id': None,
            'result_text': "[오류] 세션 ID가 전달되지 않았습니다."
        })
        return

    if current_audio_thread is not None and current_audio_thread.is_alive():
        print("⚠️ 화자 분리 거부: 실시간 번역 세션이 실행 중입니다.")
        socketio.emit("diarization_result", {
            'session_id': session_id,
            'result_text': "[오류] 실시간 번역을 먼저 중지해야 화자 분리를 실행할 수 있습니다."
        })
        return

    if current_diarize_thread is not None and current_diarize_thread.is_alive():
        print("⚠️ 화자 분리 거부: 이미 다른 세션의 화자 분리가 실행 중입니다.")
        socketio.emit("diarization_result", {
            'session_id': session_id,
            'result_text': "[오류] 이미 다른 화자 분리 작업이 실행 중입니다. 잠시 후 시도하세요."
        })
        return

    print(f"🔄 (화자 분리) 요청 수신... 대상 세션: {session_id}")

    current_diarize_thread = threading.Thread(
        target=run_diarization_thread,
        args=(session_id,),
        daemon=True
    )
    current_diarize_thread.start()


# ⭐️ [신규] 화자 분리를 위한 스레드 함수
def run_diarization_thread(session_id):
    """
    (백그라운드 스레드)
    diarize_handler.py를 실행하고, 완료되면 결과를 클라이언트에 전송합니다.
    """
    global current_diarize_thread

    try:
        result_text = diarize_handler.run_diarization(session_id)

        print(f"✅ (화자 분리) 완료. 세션: {session_id}")

        socketio.emit("diarization_result", {
            'session_id': session_id,
            'result_text': result_text
        })

    except Exception as e:
        print(f"❌ (화자 분리) 스레드 오류: {e}")
        socketio.emit("diarization_result", {
            'session_id': session_id,
            'result_text': f"[오류] 화자 분리 중 심각한 오류 발생: {e}"
        })
    finally:
        current_diarize_thread = None


# --- ⭐️ [신규] 오디오 세션 중지 함수 ---
def stop_audio_session(notify_client=True):
    global current_audio_thread, current_stop_event

    stopped_successfully = False
    if current_stop_event is not None and current_audio_thread is not None:
        print("🔄 [Session] 'stop_event' 전송. 스레드 중지 시도...")
        current_stop_event.set()

        print("🔄 [Session] 오디오 백로그 큐 비우는 중...")
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break
        print("✅ [Session] 큐 비우기 완료.")

        current_audio_thread.join(timeout=2.0)

        if not current_audio_thread.is_alive():
            print("✅ [Session] 스레드 중지 완료.")
            stopped_successfully = True
        else:
            print("⚠️ [Session] 스레드가 2초 내에 종료되지 않았습니다.")
    else:
        print("ℹ️ [Session] 중지할 활성 스레드가 없습니다.")
        stopped_successfully = True  # 중지할 것이 없어도 성공으로 간주

    current_audio_thread = None
    current_stop_event = None

    if notify_client:
        socketio.emit("session_stopped", {
            'message': '세션이 중지되었습니다. 새로 시작할 수 있습니다.'
        })
    return stopped_successfully


# --- ⭐️ [수정] Whisper 세션 시작/재시작 함수 ---
def start_new_audio_session(session_id):
    global current_audio_thread, current_stop_event

    stop_audio_session(notify_client=False)
    current_stop_event = threading.Event()

    print(f"\n🎬 [새 세션 시작] 세션 ID: {session_id}\n")

    current_audio_thread = threading.Thread(
        target=main_audio_streaming,
        args=(session_id, socketio, current_stop_event),
        daemon=True
    )
    current_audio_thread.start()
    print("🎤 Whisper 실시간 음성 인식 스레드 시작됨 ✅")

    socketio.emit("new_session_started", {
        'session_id': session_id,
        'message': '새로운 세션이 시작되었습니다.'
    })

    # ⭐️ [신규] 5. 모든 클라이언트의 세션 드롭다운 목록을 갱신
    try:
        all_sessions = get_all_session_ids()
        socketio.emit("session_list_updated", {
            'all_sessions': all_sessions,
            'latest_session': session_id  # 방금 시작한 세션을 선택
        })
        print(f"✅ 세션 목록 갱신 완료. (새 세션: {session_id})")
    except Exception as e:
        print(f"⚠️ 세션 목록 갱신 중 오류: {e}")


# --- KoBART 모델 초기화 ---
def init_summary_model():
    """서버 시작 시 KoBART 모델을 미리 로드"""
    print("🧠 KoBART 모델 로드 시도...")
    load_kobart_model()


# --- 메인 실행부 ---
if __name__ == "__main__":
    init_db()
    print("✅ DB 초기화 완료")

    # KoBART 모델 로드 스레드 시작
    threading.Thread(target=init_summary_model, daemon=True).start()

    print(f"🌍 Socket.IO 서버 시작: http://{HOST}:{PORT} 에서 접속 가능")
    print("✅ (준비 완료) 클라이언트의 '번역 시작' 요청을 대기합니다...")

    # Socket.IO 서버 실행 (메인 스레드)
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)