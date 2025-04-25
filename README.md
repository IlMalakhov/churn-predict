# Streamlit-приложение: Анализ оттока клиентов

## Предварительные требования

Убедитесь, что установлено следующее:

- Python 3.8+
- pip

## Быстрый запуск

```bash
git clone https://github.com/IlMalakhov/churn-predict.git

cd churn-predict

python -m venv .venv

source .venv/bin/activate   # или .venv\Scripts\activate на Windows

pip install -r requirements.txt

streamlit run streamlit_app.py
```

## Структура репозитория
```plaintext
churn-predict/
├── .streamlit/                      → Конфигурация Streamlit
├── .gitignore                       → Файлы и папки, игнорируемые Git
├── README.md                        → Вы находитесь здесь!
├── requirements.txt                 → Список зависимостей проекта
├── Churn_final.ipynb                → Jupyter Notebook с анализом
├── streamlit_app.py                 → Веб‑приложение Streamlit
├── EDA_dashboard_le_finale.pbix     → Power BI дашборд
├── assets/                          → Ресурсы для приложения
│   ├── logo_white_font.png          → Логотип Ростелеком
│   ├── dash1.jpg                    → Скриншот Power BI дашборда (Клиенты)
│   └── dash2.jpg                    → Скриншот Power BI дашборда (Продавцы)
└── data/                            → Изначальные данные для анализа
    ├── customers.csv                → Данные о клиентах
    ├── geolocation.csv              → Геолокация клиентов
    ├── order_items.csv              → Данные о заказах
    ├── order_payments.csv           → Данные о платежах
    ├── order_reviews.csv            → Данные о отзывах
    ├── orders.csv                   → Данные о заказах
    ├── product_category_name_translation.csv → Перевод названий категорий товаров
    ├── products.csv                 → Данные о товарах
    └── sellers.csv                  → Данные о продавцах
```
