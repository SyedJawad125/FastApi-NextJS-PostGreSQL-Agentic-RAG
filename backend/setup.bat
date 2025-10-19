# ============================================================================
# setup.bat - Windows Setup Script
# ============================================================================
@echo off
echo ======================================
echo Advanced Agentic RAG System - Setup
echo ======================================
echo.

REM Check Python
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt

REM Create directories
echo.
echo Creating directories...
if not exist data\vectors mkdir data\vectors
if not exist data\graphs mkdir data\graphs
if not exist data\documents mkdir data\documents
if not exist data\cache mkdir data\cache
if not exist logs mkdir logs
if not exist uploads mkdir uploads
if not exist uploaded_images mkdir uploaded_images

REM Copy .env.example
if not exist .env (
    echo.
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file and add your API keys
)

REM Initialize database
echo.
echo Initializing database...
python -c "from app.database import init_db; init_db()"

echo.
echo ======================================
echo Setup Complete!
echo ======================================
echo.
echo Next steps:
echo 1. Edit .env file and add your GROQ_API_KEY
echo 2. Run: venv\Scripts\activate.bat
echo 3. Run: python main.py
echo 4. Visit: http://localhost:8000/docs
echo.
pause
