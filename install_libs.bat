@echo off
:: ⭐️ [수정] 한글 깨짐 방지
chcp 65001 > nul

cd /d D:\If_Project

echo ==================================================
echo [ If_Project ] 라이브러리 자동 설치를 시작합니다...
echo ==================================================

:: 1. 가상환경이 없으면 생성
if not exist ".venv" (
    echo 가상환경(.venv)이 없어서 새로 생성합니다...
    python -m venv .venv
)

:: 2. 라이브러리 설치
echo 가상환경에 라이브러리를 설치합니다...
".venv\Scripts\pip.exe" install -r requirements.txt

echo.
echo ==================================================
echo 모든 라이브러리 설치가 완료되었습니다!
echo ==================================================
pause