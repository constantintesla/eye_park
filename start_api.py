"""
Скрипт запуска API сервера
"""

import os
from api import app

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
