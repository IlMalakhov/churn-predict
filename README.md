# Streamlit-приложение: Анализ оттока клиентов
### MiBA для Rostelecom

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
---
## Команда и распределение ролей

<div>

| Участник | Основные задачи |
|:--------:|:---------------|
| **Бакaй Екатерина**<br>[GitHub](https://github.com/poopseecheque) | - Разработка ML-моделей<br>- Создание и внедрение веб-приложения |
| **Забродская Алена**<br>[GitHub](https://github.com/AlZa01) | - Предобработка данных<br>- Генерация признаков<br>- RFM-анализ и кластеризация |
| **Тетерина Екатерина**<br>[GitHub](https://github.com/KTeterina) | - Предобработка данных<br>- Генерация признаков<br>- RFM-анализ и кластеризация |
| **Пособилов Феодосий**<br>[GitHub](https://github.com/Posobiloff) | - Исследовательский анализ (EDA)<br>- Разработка дашбордов |
| **Ковалевская Екатерина**<br>[GitHub](https://github.com/KathrineKov) | - Визуализация данных<br>- Координация процесса |
| **Малахов Илья**<br>[GitHub](https://github.com/IlMalakhov) | - Разработка ML-моделей<br>- Создание и внедрение веб-приложения |

</div>

---
```diff
  __  __   _   ____             
 |  \/  | (_) |  _ \      /\    
 | \  / |  _  | |_) |    /  \   
 | |\/| | | | |  _ <    / /\ \  
 | |  | | | | | |_) |  / ____ \ 
 |_|  |_| |_| |____/  /_/    \_\

 __  __
 \ \/ /
  >  < 
 /_/\_\

@@  _____                  _            _                                    @@
@@ |  __ \                | |          | |                                   @@
@@ | |__) |   ___    ___  | |_    ___  | |   ___    ___    ___    _ __ ___   @@
@@ |  _  /   / _ \  / __| | __|  / _ \ | |  / _ \  / __|  / _ \  | '_ ` _ \  @@
@@ | | \ \  | (_) | \__ \ | |_  |  __/ | | |  __/ | (__  | (_) | | | | | | | @@
@@ |_|  \_\  \___/  |___/  \__|  \___| |_|  \___|  \___|  \___/  |_| |_| |_| @@

```
                                                                          

       

                                

                         

                                                                                                
                                                                                                

