# Публикация на Streamlit Community Cloud

1. Убедитесь, что изменения из ветки портфолио вошли в `main`. В её корне должны быть `streamlit_app.py` и обновлённый `requirements.txt`.
2. Войдите в [Streamlit Community Cloud](https://share.streamlit.io/) через аккаунт с доступом к `kziborova/Event-dashboard`. Выберите **Create app → Yup, I have an app**.
3. Укажите репозиторий `kziborova/Event-dashboard`, ветку `main`, файл `streamlit_app.py`. В поле **App URL** введите `event-attendance-optimizer`, если имя доступно.
4. В **Advanced settings** выберите **Python 3.12**, затем **Deploy**. Дашборд читает CSV и PKL из репозитория. Поле Secrets можно оставить пустым.
5. Откройте выданный сервисом URL в приватном окне. Проверьте «Обзор», «Аналитика», числовой результат в «Прогноз» и диаграмму в «Модель».
6. Оберните изображение `assets/demo.svg` в обоих README в ссылку на проверенный URL и уберите подпись об ожидании публикации. На GitHub откройте **About → Edit repository metadata**, вставьте адрес в **Website** и сохраните.

[Инструкция Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
