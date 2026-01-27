Param()

# Включаем UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "[EYE_PARK] Подготовка окружения..." -ForegroundColor Cyan

# Переходим в директорию скрипта
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Создаем виртуальное окружение, если его нет
if (-not (Test-Path -Path ".\venv")) {
    Write-Host "Создаю виртуальное окружение venv..." -ForegroundColor Yellow
    python -m venv venv
}

# Активируем окружение
$activatePath = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Host "Не найден файл активации виртуального окружения: $activatePath" -ForegroundColor Red
    Write-Host "Убедись, что Python установлен и команда 'python -m venv venv' отработала без ошибок." -ForegroundColor Red
    exit 1
}

Write-Host "Активирую виртуальное окружение..." -ForegroundColor Yellow
. $activatePath

Write-Host "Устанавливаю зависимости из requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Запускаю API (порт 8000)..." -ForegroundColor Green
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "Set-Location `"$ScriptDir`"; .\venv\Scripts\Activate.ps1; python start_api.py"

Start-Sleep -Seconds 2

Write-Host "Запускаю Telegram-бота..." -ForegroundColor Green
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "Set-Location `"$ScriptDir`"; .\venv\Scripts\Activate.ps1; python start_bot.py"

Write-Host ""
Write-Host "Всё запущено." -ForegroundColor Green
Write-Host "API:   http://localhost:8000"

