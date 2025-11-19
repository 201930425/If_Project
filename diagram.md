flowchart TD

&nbsp;   %% --- 사용자 및 인터페이스 ---

&nbsp;   User(\[사용자])

&nbsp;   UI\["웹 인터페이스<br/>(translation.html)"]



&nbsp;   %% --- 메인 서버 ---

&nbsp;   Server\["Flask \& Socket.IO 서버<br/>(app.py)"]



&nbsp;   %% --- 저장소 ---

&nbsp;   DB\[("SQLite DB<br/>translations.db")]

&nbsp;   WavFiles\[("오디오 파일 저장소<br/>(wav/ 폴더)")]



&nbsp;   %% --- 연결 ---

&nbsp;   User -->|버튼 클릭| UI

&nbsp;   UI <-->|"Socket.IO 이벤트"| Server



&nbsp;   %% =================================================

&nbsp;   %% 1. 실시간 번역 프로세스 (Audio Processor)

&nbsp;   %% =================================================

&nbsp;   subgraph RealTime \["🎙️ 실시간 번역 (audio\_processor.py)"]

&nbsp;       Mic\[마이크 입력]

&nbsp;       VAD{"Silero VAD<br/>음성 감지?"}

&nbsp;       STT\["Whisper Base<br/>(STT)"]

&nbsp;       Trans\["Google Translate<br/>(번역)"]

&nbsp;       SaveWav\[WAV 파일 쓰기]



&nbsp;       Mic --> VAD

&nbsp;       VAD -- Yes --> SaveWav

&nbsp;       VAD -- Yes --> STT

&nbsp;       STT --> Trans

&nbsp;   end



&nbsp;   %% 실시간 흐름 연결

&nbsp;   Server -->|"Start Session"| RealTime

&nbsp;   SaveWav --> WavFiles

&nbsp;   Trans -->|"결과 전송"| Server

&nbsp;   Trans -->|"DB 저장"| DB



&nbsp;   %% =================================================

&nbsp;   %% 2. 후처리: 화자 분리 (Diarize Handler)

&nbsp;   %% =================================================

&nbsp;   subgraph Diarization \["🗣️ 화자 분리 (diarize\_handler.py)"]

&nbsp;       LoadWav\[WAV 파일 로드]

&nbsp;       WhisperX\["WhisperX<br/>(정밀 STT \& 정렬)"]

&nbsp;       Pyannote\["Pyannote.audio<br/>(화자 식별)"]

&nbsp;       Combine\[결과 병합 및 포맷팅]



&nbsp;       LoadWav --> WhisperX

&nbsp;       LoadWav --> Pyannote

&nbsp;       WhisperX --> Combine

&nbsp;       Pyannote --> Combine

&nbsp;   end



&nbsp;   %% 화자 분리 흐름 연결

&nbsp;   Server -->|"Request Diarization"| Diarization

&nbsp;   WavFiles --> LoadWav

&nbsp;   Combine -->|"분석 결과 전송"| Server



&nbsp;   %% =================================================

&nbsp;   %% 3. 후처리: 요약 (Summary Handler)

&nbsp;   %% =================================================

&nbsp;   subgraph Summarization \["📑 AI 요약 (summary\_handler.py)"]

&nbsp;       FetchText\[DB 텍스트 조회]

&nbsp;       MapReduce{"텍스트 길이 > 1024?"}

&nbsp;       ChunkSum\["청크별 요약 (Map)"]

&nbsp;       FinalSum\["최종 요약 (Reduce)"]

&nbsp;       SimpleSum\[단일 요약]



&nbsp;       FetchText --> MapReduce

&nbsp;       MapReduce -- Yes --> ChunkSum --> FinalSum

&nbsp;       MapReduce -- No --> SimpleSum

&nbsp;   end



&nbsp;   %% 요약 흐름 연결

&nbsp;   Server -->|"Request Summary"| Summarization

&nbsp;   DB --> FetchText

&nbsp;   FinalSum -->|"요약문 전송"| Server

&nbsp;   SimpleSum -->|"요약문 전송"| Server



&nbsp;   %% =================================================

&nbsp;   %% 4. 데이터 관리

&nbsp;   %% =================================================

&nbsp;   Server -->|"Request Delete"| DBDelete\[DB 데이터 삭제]

&nbsp;   Server -->|"Request Delete"| FileDelete\[WAV 파일 삭제]

&nbsp;   DBDelete --> DB

&nbsp;   FileDelete --> WavFiles

