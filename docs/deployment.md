# Публикация Event Scenario Lab на Streamlit Community Cloud

Рабочее приложение — [Event Scenario Lab](https://event-scenario-lab.streamlit.app/).

1. Конфигурация развёртывания хранится в ветке `main`. В корне находятся `streamlit_app.py` и `requirements.txt`.
2. Войдите в [Streamlit Community Cloud](https://share.streamlit.io/) через аккаунт с доступом к `kziborova/Event-dashboard`. Выберите **Create app → Yup, I have an app**.
3. Укажите репозиторий `kziborova/Event-dashboard`, ветку `main`, файл `streamlit_app.py`. В поле **App URL** введите `event-scenario-lab`, если имя доступно.
4. В **Advanced settings** выберите **Python 3.12**, затем **Deploy**. Дашборд читает CSV и PKL из репозитория. Поле Secrets можно оставить пустым.
5. Откройте выданный сервисом URL в приватном окне. Проверьте «Обзор», «Аналитика», числовой результат в «Прогноз» и диаграмму в «Модель».
[Инструкция Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
