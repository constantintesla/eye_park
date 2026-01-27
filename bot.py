"""
Telegram бот для загрузки и анализа видео
Аналогично bot.py из audio_park
"""

import os
import asyncio
import requests
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
API_URL = os.getenv('API_URL', 'http://localhost:8000')

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен в переменных окружения")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


class AnalysisStates(StatesGroup):
    waiting_for_video = State()


# Инструкция для записи видео (единый протокол: калибровка + упражнение)
VIDEO_INSTRUCTIONS = """
📹 Пожалуйста, запишите ОДНО видео по следующему протоколу:

ЧАСТЬ 1 — КАЛИБРОВКА ВЗГЛЯДА (10–20 секунд)
1. Сядьте прямо перед камерой на расстоянии 50–70 см, не поворачивайте голову.
2. Смотрите прямо в камеру 2–3 секунды.
3. По очереди переведите взгляд:
   • максимально ВЛЕВО (2–3 секунды),
   • максимально ВПРАВО (2–3 секунды),
   • максимально ВНИЗ (2–3 секунды),
   • максимально ВВЕРХ (2–3 секунды).
4. Важно: двигайте только глазами, голову не поворачивайте.

ЧАСТЬ 2 — УПРАЖНЕНИЕ (40–70 секунд)
5. Снова смотрите прямо в камеру (фиксация) 3–5 секунд.
6. Быстро переводите взгляд между воображаемыми точками СЛЕВА и СПРАВА (горизонтальные «да–нет») 10–15 раз.
7. Быстро переводите взгляд между ВЕРХНЕЙ и НИЖНЕЙ точками (вертикальные «да–нет») 10–15 раз.
8. Плавно ведите взгляд по воображаемому кругу (по часовой стрелке и против) 10–15 секунд.
9. Далее смотрите естественно, как при разговоре, 10–15 секунд, не думая специально о морганиях.

РЕКОМЕНДАЦИИ
• Постарайтесь не закрывать лицо руками и не менять положение головы.
• Длительность видео: 50–90 секунд.
• Формат: MP4, AVI, MOV (стандартное видео с фронтальной камеры).
"""


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработка команды /start"""
    try:
        await message.answer(
            "👋 Добро пожаловать в систему анализа движения глаз!\n\n"
            "Эта система помогает выявить симптомы неврологических расстройств "
            "на основе анализа движения глаз, моргания и мимики.\n\n"
            "Используйте команду /analyze для начала анализа видео."
        )
    except Exception as e:
        logger.error(f"Ошибка в команде /start: {e}", exc_info=True)
        await message.answer("Произошла ошибка. Попробуйте позже.")


@dp.message(Command("analyze"))
async def cmd_analyze(message: Message, state: FSMContext):
    """Обработка команды /analyze"""
    try:
        await message.answer(VIDEO_INSTRUCTIONS)
        await state.set_state(AnalysisStates.waiting_for_video)
    except Exception as e:
        logger.error(f"Ошибка в команде /analyze: {e}", exc_info=True)
        await message.answer("Произошла ошибка. Попробуйте позже.")


@dp.message(Command("history"))
async def cmd_history(message: Message):
    """Обработка команды /history - история результатов"""
    try:
        response = requests.get(f"{API_URL}/api/results", timeout=10)
        if response.status_code == 200:
            results = response.json()
            if not results:
                await message.answer("История результатов пуста.")
                return
            
            # Показываем последние 5 результатов
            recent_results = results[-5:]
            history_text = "📊 Последние результаты анализа:\n\n"
            
            for result in reversed(recent_results):
                history_text += (
                    f"📅 {result.get('timestamp', 'N/A')}\n"
                    f"📁 {result.get('filename', 'N/A')}\n"
                    f"⚠️ Уровень риска: {result.get('risk_level', 'N/A')}\n"
                    f"📈 Вероятность: {result.get('risk_probability', 0.0)*100:.1f}%\n"
                    f"📊 EMSI: {result.get('emsi_score', 0.0):.2f} ({result.get('emsi_range', 'N/A')})\n"
                    f"─────────────────────\n"
                )
            
            await message.answer(history_text)
        else:
            await message.answer("Ошибка при получении истории результатов.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка подключения к API: {e}")
        await message.answer("Не удалось подключиться к API серверу. Убедитесь, что сервер запущен.")
    except Exception as e:
        logger.error(f"Ошибка в команде /history: {e}", exc_info=True)
        await message.answer(f"Ошибка: {str(e)}")


@dp.message(AnalysisStates.waiting_for_video)
async def process_video(message: Message, state: FSMContext):
    """Обработка загруженного видео"""
    # Поддерживаем обычное видео, документы с видео и видео-заметки (кружочки)
    if not message.video and not message.document and not message.video_note:
        await message.answer(
            "Пожалуйста, загрузите видео файл или кружочек.\n"
            "Используйте команду /analyze для повторной попытки."
        )
        await state.clear()
        return
    
    # Отправка сообщения о начале обработки
    processing_msg = await message.answer("⏳ Обработка видео... Это может занять несколько минут.")
    
    try:
        # Получение файла
        if message.video:
            file = await bot.get_file(message.video.file_id)
        elif message.video_note:
            # Видео-заметка (кружочек)
            file = await bot.get_file(message.video_note.file_id)
        else:  # документ с видео
            file = await bot.get_file(message.document.file_id)
        
        # Скачивание файла
        file_path = file.file_path
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        
        # Скачивание файла во временную директорию
        os.makedirs('temp', exist_ok=True)
        temp_file_path = os.path.join('temp', f"{message.from_user.id}_{file.file_id}.mp4")
        
        file_response = requests.get(file_url)
        with open(temp_file_path, 'wb') as f:
            f.write(file_response.content)
        
        # Подготовка информации о пользователе Telegram
        user_info = {
            'id': message.from_user.id,
            'username': message.from_user.username or '',
            'first_name': message.from_user.first_name or '',
            'last_name': message.from_user.last_name or '',
            'is_bot': message.from_user.is_bot
        }
        
        # Отправка файла на API для анализа с информацией о пользователе
        with open(temp_file_path, 'rb') as f:
            files = {'file': (os.path.basename(temp_file_path), f, 'video/mp4')}
            headers = {
                'X-Source': 'telegram',
                'X-User-Info': json.dumps(user_info)
            }
            api_response = requests.post(f"{API_URL}/api/analyze", files=files, headers=headers)
        
        # Удаление временного файла
        os.remove(temp_file_path)
        
        if api_response.status_code == 200:
            result = api_response.json()
            
            # Формирование отчета
            report_text = format_analysis_report(result)
            
            await processing_msg.edit_text(report_text)
        else:
            error_data = api_response.json()
            await processing_msg.edit_text(
                f"❌ Ошибка при анализе видео:\n{error_data.get('error', 'Неизвестная ошибка')}"
            )
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка подключения к API: {e}")
        await processing_msg.edit_text(
            "❌ Не удалось подключиться к API серверу.\n"
            "Убедитесь, что сервер запущен и доступен."
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}", exc_info=True)
        await processing_msg.edit_text(f"❌ Ошибка: {str(e)}")
    
    finally:
        try:
            await state.clear()
        except:
            pass


def format_analysis_report(result: dict) -> str:
    """Форматирование отчета анализа для отправки в Telegram"""
    report = "📊 РЕЗУЛЬТАТЫ АНАЛИЗА\n\n"
    
    # EMSI
    emsi = result.get('emsi', {})
    report += f"📈 EMSI Score: {emsi.get('emsi_score', 0.0):.2f}\n"
    report += f"📊 Диапазон: {emsi.get('emsi_range', 'N/A')}\n"
    report += f"💡 {emsi.get('interpretation', '')}\n\n"
    
    # Уровень риска
    risk_level = result.get('risk_level', 'Unknown')
    risk_prob = result.get('risk_probability', 0.0) * 100
    
    risk_emoji = {
        'Low': '✅',
        'Medium': '⚠️',
        'High': '🔴'
    }
    
    report += f"{risk_emoji.get(risk_level, '❓')} Уровень риска: {risk_level}\n"
    report += f"📊 Вероятность: {risk_prob:.1f}%\n\n"
    
    # Основные признаки
    features = result.get('features', {})
    report += "🔍 ОСНОВНЫЕ ПРИЗНАКИ:\n"
    report += f"  • Частота саккад: {features.get('saccade_frequency', 0.0):.2f} саккад/сек\n"
    report += f"  • Частота моргания: {features.get('blink_rate', 0.0):.2f} морганий/мин\n"
    report += f"  • Стабильность фиксаций: {features.get('fixation_stability', 0.0):.2f} градусов\n"
    report += f"  • Асимметрия: {features.get('asymmetry_left_right', 0.0)*100:.1f}%\n\n"
    
    # Оценки симптомов
    symptom_scores = result.get('symptom_scores', {})
    if symptom_scores:
        report += "📋 ОЦЕНКИ СИМПТОМОВ:\n"
        severity_names = {0: 'Нет', 1: 'Легкий', 2: 'Умеренный', 3: 'Тяжелый'}
        
        for symptom, score in list(symptom_scores.items())[:5]:  # Показываем первые 5
            severity = severity_names.get(score, 'N/A')
            report += f"  • {symptom}: {score} ({severity})\n"
        report += "\n"
    
    # Рекомендации
    recommendation = result.get('recommendation', '')
    if recommendation:
        report += f"💡 РЕКОМЕНДАЦИИ:\n{recommendation}\n"
    
    report += "\n⚠️ ВАЖНО: Данная система предназначена для исследовательских целей и не заменяет медицинскую диагностику."
    
    return report


@dp.message()
async def handle_other_messages(message: Message):
    """Обработка других сообщений"""
    await message.answer(
        "Используйте команды:\n"
        "/start - Начать работу\n"
        "/analyze - Начать анализ видео\n"
        "/history - Просмотр истории результатов"
    )


async def main():
    """Главная функция запуска бота"""
    logger.info("Запуск бота...")
    logger.info(f"API URL: {API_URL}")
    
    try:
        # Проверка подключения к API
        try:
            response = requests.get(f"{API_URL}/api/stats", timeout=5)
            logger.info(f"API доступен: {response.status_code}")
        except Exception as e:
            logger.warning(f"API недоступен: {e}. Бот будет работать, но анализ видео может не работать.")
        
        logger.info("Бот запущен и готов к работе")
        await dp.start_polling(bot, skip_updates=True)
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки (Ctrl+C)")
    except Exception as e:
        logger.error(f"Ошибка в боте: {e}", exc_info=True)
    finally:
        logger.info("Закрытие сессии бота...")
        try:
            await bot.session.close()
        except:
            pass
        logger.info("Бот остановлен")


if __name__ == '__main__':
    # Для Windows используем правильный event loop
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
