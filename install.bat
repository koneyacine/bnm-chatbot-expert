@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo      Phi-3.5 Chatbot v4 INSTALLATION
echo ==========================================

REM Check if venv exists
if exist "venv" goto :venv_exists

echo Creating virtual environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error creating venv. Check Python installation.
    pause
    exit /b
)
echo Venv created.

:venv_exists
REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate

REM Installing dependencies
echo Installing dependencies (This may take a few minutes)...
python -m pip install --upgrade pip
REM Install PyTorch specifically for newer CUDA/Python
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
REM Install other requirements
python -m pip install -r requirements.txt

echo.
echo ==========================================
echo      INSTALLATION COMPLETE
echo ==========================================
echo.
echo Please put your DOCX files in the "data" folder.
echo Then run "run.bat" to start the chatbot.
echo.
pause
