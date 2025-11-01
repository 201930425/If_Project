import sqlite3
from datetime import datetime, timezone, timedelta
from config import DB_NAME


def init_db():
    """데이터베이스 테이블을 초기화합니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_text TEXT,
            translated_text TEXT
        )
        ''')
        conn.commit()
        print(f"✅ DB '{DB_NAME}' 초기화 완료.")
    except Exception as e:
        print(f"⚠️ DB 초기화 실패: {e}")
    finally:
        if conn:
            conn.close()


def insert_transcript(session_id, original, translated):
    """번역 결과를 DB에 삽입합니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        kst = timezone(timedelta(hours=9))
        now = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO transcripts (session_id, timestamp, original_text, translated_text) VALUES (?, ?, ?, ?)",
            (session_id, now, original, translated)
        )
        conn.commit()
    except Exception as e:
        print(f"⚠️ DB 삽입 실패: {e}")
    finally:
        if conn:
            conn.close()


def fetch_data_from_db(session_id=None):
    """DB에서 텍스트를 가져옵니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        query = "SELECT translated_text FROM transcripts"
        params = []

        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        text_list = [row[0] for row in rows if row[0] and row[0].strip() not in ["[번역 실패]", "[빈 문자열]"]]

        if not text_list:
            return ""

        # "\n" (줄바꿈) 대신 " " (공백)으로 문장을 연결합니다.
        # 'gogamza' (뉴스) 모델은 하나의 긴 문단을 예상합니다.
        return " ".join(text_list)
    except Exception as e:
        print(f"⚠️ DB 읽기 실패: {e}")
        return ""
    finally:
        if conn:
            conn.close()


def get_latest_session_id():
    """DB에서 가장 최근의 session_id를 가져옵니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        query = "SELECT session_id FROM transcripts ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query)
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        print(f"⚠️ 최근 세션 ID 조회 실패: {e}")
        return None
    finally:
        if conn:
            conn.close()

