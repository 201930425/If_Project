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

def get_all_session_ids():
    """DB에서 모든 고유한 session_id 목록을 (최신순으로) 가져옵니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        # 1. 고유한 ID를 2. 타임스탬프 기준으로 3. 내림차순 정렬
        query = """
        SELECT DISTINCT session_id 
        FROM transcripts 
        ORDER BY timestamp DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        # [('20251105_115000',), ('20251105_100000',)] -> ['20251105_115000', '20251105_100000']
        return [row[0] for row in rows]
    except Exception as e:
        print(f"⚠️ 모든 세션 ID 조회 실패: {e}")
        return []
    finally:
        if conn:
            conn.close()


def rename_session(old_id, new_id):
    """DB에서 'old_id'를 'new_id'로 변경합니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()

        # 1. (선택적) 혹시 모를 중복 방지: new_id가 이미 있는지 확인
        cursor.execute("SELECT 1 FROM transcripts WHERE session_id = ? LIMIT 1", (new_id,))
        if cursor.fetchone():
            print(f"⚠️ 이름 변경 실패: '{new_id}'가 이미 존재합니다.")
            return False

        # 2. 이름 변경 실행
        query = "UPDATE transcripts SET session_id = ? WHERE session_id = ?"
        cursor.execute(query, (new_id, old_id))
        conn.commit()

        # 3. 변경 사항 확인
        if cursor.rowcount > 0:
            print(f"✅ DB 세션 이름 변경 완료: '{old_id}' -> '{new_id}' ({cursor.rowcount}개 레코드)")
            return True
        else:
            print(f"⚠️ 이름 변경 실패: '{old_id}'를 찾을 수 없습니다.")
            return False

    except Exception as e:
        print(f"⚠️ 세션 이름 변경 중 DB 오류: {e}")
        return False
    finally:
        if conn:
            conn.close()