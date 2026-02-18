@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo      Starting Simple Chatbot...
echo ==========================================

REM Activate venv
call venv\Scripts\activate

echo Launching Interface...
streamlit run app.py

pause
