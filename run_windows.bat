@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.11 -m venv .venv
) else (
    python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo Python environment setup failed. Install Python 3.11.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if not exist models\apple_orange_cnn.keras (
    .venv\Scripts\python.exe train_model.py
    if errorlevel 1 (
        echo Could not train CNN. Check internet and the error above.
        pause
        exit /b 1
    )
)
.venv\Scripts\python.exe -m streamlit run app.py
endlocal
