@echo off
chcp 65001 >nul
setlocal ENABLEDELAYEDEXPANSION

REM Переходим в директорию скрипта
cd /d %~dp0

echo [EYE_PARK] Подготовка окружения...

REM Создаем виртуальное окружение, если его нет
if not exist venv (
    echo Создаю виртуальное окружение venv...
    python -m venv venv
)

REM Активируем окружение
call venv\Scripts\activate.bat

echo Устанавливаю зависимости из requirements.txt...
pip install -r requirements.txt

echo.
echo Запускаю API (порт 8000)...
start "eye_park_api" cmd /c "cd /d %~dp0 && call venv\Scripts\activate.bat && python start_api.py && pause"

echo Запускаю Telegram-бота...
start "eye_park_bot" cmd /c "cd /d %~dp0 && call venv\Scripts\activate.bat && python start_bot.py && pause"

echo.
echo Всё запущено.
echo API:   http://localhost:8000

echo.
pause
