# ==============================================
# ИМПОРТ БИБЛИОТЕК
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import holoviews as hv
import geoviews as gv
import datashader as ds
from datashader.utils import lnglat_to_meters as webm
from holoviews.operation.datashader import datashade, dynspread, rasterize
from holoviews.streams import RangeXY
from colorcet import rainbow, fire, bgy, kr
from bokeh.models import WMTSTileSource
from streamlit_bokeh import streamlit_bokeh

import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    recall_score, precision_score, f1_score, accuracy_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
from scipy import stats
from scipy.special import boxcox1p

# ==============================================
# НАСТРОЙКА СТРАНИЦЫ STREAMLIT
# ==============================================
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# ФУНКЦИИ ДЛЯ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ ДАННЫХ
# ==============================================

@st.cache_data
def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает все необходимые CSV-файлы, выполняет базовую предобработку,
    объединяет данные и вычисляет признаки, включая RFM и статус оттока.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Кортеж из двух DataFrame:
            - Основной DataFrame со всеми объединенными и обработанными данными.
            - DataFrame с RFM-метриками для каждого клиента.
    """
    # Загрузка данных из CSV файлов
    customers = pd.read_csv('data/customers.csv')
    orders = pd.read_csv('data/orders.csv')
    order_payments = pd.read_csv('data/order_payments.csv')
    order_reviews = pd.read_csv('data/order_reviews.csv')
    products = pd.read_csv('data/products.csv')
    product_category = pd.read_csv('data/product_category_name_translation.csv')
    sellers = pd.read_csv('data/sellers.csv')
    order_items = pd.read_csv('data/orders_items.csv')

    for df in [order_payments, order_reviews, product_category, sellers, order_items]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    # Преобразование столбцов с датами в формат datetime
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        invalid_count = orders[col].isnull().sum()

    order_reviews = pd.read_csv('data/order_reviews.csv')
    order_reviews['review_creation_date'] = pd.to_datetime(
        order_reviews['review_creation_date'], 
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')

    order_reviews['review_answer_timestamp'] = pd.to_datetime(
        order_reviews['review_answer_timestamp'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')
    
    order_items['shipping_limit_date'] = pd.to_datetime(
        order_items['shipping_limit_date'],
        format='%Y-%m-%d %H:%M:%S', 
        errors='coerce')

    order_items['shipping_limit_date.1'] = pd.to_datetime(
        order_items['shipping_limit_date.1'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')

    # Объединение данных о заказах и клиентах
    orders = pd.merge(
        orders,
        customers[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']],
        on='customer_id',
        how='inner'
    )
    orders.drop('customer_id', axis=1, inplace=True)

    orders['quantity'] = orders.groupby('customer_unique_id')['order_id'].transform('nunique')

    # Объединение заказов с отзывами
    merge1 = pd.merge(
        orders,
        order_reviews[['order_id', 'review_creation_date', 'review_score']],
        on='order_id',
        how='left'
    )

    #merge1.dropna(subset=['review_creation_date', 'review_score'], inplace=True)
    merge1.dropna(inplace=True)
    merge1['review_creation_date'] = pd.to_datetime(merge1['review_creation_date'])

    # Определение даты для анализа (последняя дата заказа + 1 день)
    analysis_date = merge1['order_purchase_timestamp'].max() + timedelta(days=1)

    ####################################################################################################################################################
    
    df_sorted = merge1.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    # группируем по клиенту и вычисляем разницу между заказами
    def calculate_avg_time_between_orders(group):
        if len(group) > 1:
            time_diffs = group['order_purchase_timestamp'].diff().dt.days  # разница в днях
            return time_diffs.median() #медиана
        else:
            return pd.NA  # если заказ один, возвращаем NA
    
    avg_time_between_orders = (
        df_sorted
        .groupby('customer_unique_id')
        .apply(calculate_avg_time_between_orders)
        .reset_index(name='avg_days_between_orders')
    )
    
    threshold = avg_time_between_orders.avg_days_between_orders.mean() # среднее - 77, медиана - 25
    
    # Для каждого клиента вычисляем разницу между датами заказов
    df_sorted['time_diff_days'] = df_sorted.groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    
    # Считаем медиану интервала между заказами по каждому клиенту — она характеризует их частоту покупок
    customer_intervals = df_sorted.groupby('customer_unique_id')['time_diff_days'].median().reset_index()
    customer_intervals.rename(columns={'time_diff_days': 'median_interval_days'}, inplace=True)
    
    # Получаем последнюю дату заказа (last_order_date) и общее число заказов (order_count) (сводка по клиенту)
    customer_stats = merge1.groupby('customer_unique_id').agg(
        last_order_date=('order_purchase_timestamp', 'max'),
        order_count=('order_id', 'nunique')
    ).reset_index()
    
    # Количество дней с момента последнего заказа
    customer_stats['days_since_last_order'] = (analysis_date - customer_stats['last_order_date']).dt.days
    
    customer_features = pd.merge(customer_stats, customer_intervals, on='customer_unique_id', how='left')
    
    # Определяем логику определения оттока
    churn_threshold_multiplier = 2.0 # Сколько раз нужно выждать средний интервал, прежде чем определять возможность оттока
    default_churn_days = threshold # Порог для клиентов с разовым заказом
    
    def calculate_dynamic_churn(row):
        if row['order_count'] == 1:
            # Если у клиента только один заказ, используем глобальный threshold как порог
            return 1 if row['days_since_last_order'] > default_churn_days else 0
        elif pd.isna(row['median_interval_days']) or row['median_interval_days'] <= 0:
             # Если у клиента несколько заказов, но неполные/аномальные интервалы — также fallback на глобальный порог.
             return 1 if row['days_since_last_order'] > default_churn_days else 0
        else:
            # Иначе применяем персонализированный порог: медианный интервал * коэффициент, где коэффициент (churn_threshold_multiplier = 2.0) можно настраивать.
            personalized_threshold = row['median_interval_days'] * churn_threshold_multiplier
            return 1 if row['days_since_last_order'] > personalized_threshold else 0
    
    customer_features['is_churned'] = customer_features.apply(calculate_dynamic_churn, axis=1)
    
    # Объединяем полученные признаки с основным датафреймом
    # Выбираем только нужные столбцы
    #churn_data_to_merge = customer_features[['customer_unique_id', 'is_churned', 'days_since_last_order', 'median_interval_days']]
    
    merge1 = pd.merge(
        merge1,
        customer_features[['customer_unique_id', 'is_churned', 'days_since_last_order', 'median_interval_days']],
        on='customer_unique_id',
        how='left'
    )
##########################################################################################################################################################
    # Обработка данных о продуктах и категориях
    products = pd.merge(
        products, 
        product_category, 
        on='product_category_name', 
        how='left'
    )
    products.drop('product_category_name', axis=1, inplace=True)
    products = products.rename(columns={'product_category_name_english': 'product_category_name'})
    products['product_category_name'] = products['product_category_name'].str.strip().str.lower()
    products.dropna(subset=['product_category_name'], inplace=True)

    category_corrections = {
        'home_confort': 'home_comfort',
        'home_appliances_2': 'home_appliances',
    }
    products['product_category_name'] = products['product_category_name'].replace(category_corrections)

    # Объединение продуктов с элементами заказа
    merge2 = pd.merge(
        products,
        order_items[['order_id', 'product_id', 'seller_id', 'price']],
        on='product_id',
        how='left'
    ).drop_duplicates()

    # Объединение с данными о продавцах
    merge3 = pd.merge(
        merge2,
        sellers[['seller_id', 'seller_city', 'seller_state']],
        on='seller_id',
        how='left'
    )

    # Объединение с основным DataFrame (merge1)
    merge4 = pd.merge(
        merge1,
        merge3,
        on='order_id',
        how='left'
    )

    # Объединение с данными об оплате
    merge5 = pd.merge(
        merge4,
        order_payments[['order_id', 'payment_type', 'payment_installments', 'payment_value']],
        on='order_id',
        how='left'
    ).drop_duplicates()

    merge5 = merge5.dropna(subset=['product_id', 'payment_type'])

    # Расчет временных метрик доставки и обработки заказа
    merge5['approval_delay'] = (merge5['order_approved_at'] - merge5['order_purchase_timestamp']).dt.total_seconds()
    merge5['delivery_time_to_customer_days'] = (merge5['order_delivered_customer_date'] - merge5['order_purchase_timestamp']).dt.days
    merge5['delivery_delay'] = (merge5['order_delivered_customer_date'] - merge5['order_estimated_delivery_date']).dt.days
    merge5['estimated_delivery_time'] = (merge5['order_estimated_delivery_date'] - merge5['order_purchase_timestamp']).dt.days

    # Фильтрация некорректных временных интервалов
    merge5 = merge5[
        (merge5['order_approved_at'] < merge5['order_delivered_customer_date']) &
        (merge5['order_approved_at'] < merge5['order_delivered_carrier_date']) &
        (merge5['order_delivered_carrier_date'] < merge5['order_delivered_customer_date'])
    ]

    # Расчет RFM метрик (Recency, Frequency, MonetaryValue)
    rfm = merge5.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': 'max', # Recency (дата последней покупки)
        'order_id': 'nunique',            # Frequency (количество заказов)
        'payment_value': 'sum'            # MonetaryValue (сумма покупок)
    }).reset_index()

    rfm.columns = ['customer_unique_id', 'last_order', 'Frequency', 'MonetaryValue']
    rfm['Recency'] = (analysis_date - rfm['last_order']).dt.days
    rfm.drop('last_order', axis=1, inplace=True)

    quantiles = rfm[['Recency', 'Frequency', 'MonetaryValue']].quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()

    def RScore(x, p, d) -> int:
        """Присваивает R-скор на основе значения Recency и квантилей."""
        if x <= d[p][0.2]: return 5
        elif x <= d[p][0.4]: return 4
        elif x <= d[p][0.6]: return 3
        elif x <= d[p][0.8]: return 2
        else: return 1

    def FMScore(x, p, d) -> int:
        """Присваивает F и M скоры на основе значений Frequency/MonetaryValue и квантилей."""
        if x <= d[p][0.2]: return 1
        elif x <= d[p][0.4]: return 2
        elif x <= d[p][0.6]: return 3
        elif x <= d[p][0.8]: return 4
        else: return 5

    rfm['R'] = rfm['Recency'].apply(RScore, args=('Recency', quantiles))
    rfm['F'] = rfm['Frequency'].apply(FMScore, args=('Frequency', quantiles))
    rfm['M'] = rfm['MonetaryValue'].apply(FMScore, args=('MonetaryValue', quantiles))
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)

    # Объединение RFM метрик с основным DataFrame
    merge6 = pd.merge(
        merge5,
        rfm[['customer_unique_id', 'Frequency', 'MonetaryValue', 'Recency', 'R', 'F', 'M', 'RFM_Score']],
        on='customer_unique_id',
        how='left'
    )

    features = ['customer_unique_id', 'customer_city', 'customer_state', 'order_status', 'payment_type',
        'payment_installments', 'payment_value', 'review_score', 'price',
        'product_weight_g', 'seller_city', 'seller_state', 'product_category_name',
        'approval_delay', 'delivery_time_to_customer_days', 'estimated_delivery_time',
        'delivery_delay', 'RFM_Score', 'is_churned'
    ]

    merge6.dropna(subset=features, inplace=True)

    return merge6[features], rfm, merge6

@st.cache_data(show_spinner="🌎 загружаем geo …")
def load_geo():
    geo = pd.read_csv("data/geolocation.csv", dtype={"geolocation_zip_code_prefix": str})
    for n in range(1, 5):
        geo[f"geolocation_zip_code_prefix_{n}_digits"] = geo["geolocation_zip_code_prefix"].str[:n]
    geo = geo.query(
        "geolocation_lat <= 5.27438888 and geolocation_lng >= -73.98283055 and "
        "geolocation_lat >= -33.75116944 and geolocation_lng <= -34.79314722"
    )
    geo["x"], geo["y"] = webm(geo.geolocation_lng, geo.geolocation_lat)
    geo[[c for c in geo.columns if "prefix" in c]] = geo.filter(like="prefix").astype(int)
    return geo

@st.cache_data(show_spinner="📦 загружаем orders …")
def load_orders():
    o   = pd.read_csv("data/orders.csv")
    it  = pd.read_csv("data/orders_items.csv")
    rv  = pd.read_csv("data/order_reviews.csv")
    cu  = pd.read_csv("data/customers.csv", dtype={"customer_zip_code_prefix": str})
    cu["customer_zip_code_prefix_3_digits"] = cu["customer_zip_code_prefix"].str[:3].astype(int)
    return (
        o.merge(it, on="order_id").merge(cu, on="customer_id").merge(rv, on="order_id")
    )

# ==============================================
# НАСТРОЙКИ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ
# ==============================================

numerical_features = ['payment_value','review_score','price','product_weight_g',
                  'approval_delay', 'RFM_Score']
categorical_features = ['customer_city','customer_state','order_status','payment_type',
                      'payment_installments','seller_city','seller_state','product_category_name']

def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Рассчитывает специфичность (True Negative Rate) модели.

    Args:
        y_true (np.ndarray): Истинные метки классов.
        y_pred (np.ndarray): Предсказанные метки классов.

    Returns:
        float: Значение специфичности.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def extended_classification_report(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str) -> dict:
    """
    Рассчитывает расширенный набор метрик классификации.

    Args:
        y_true (np.ndarray): Истинные метки классов.
        y_pred (np.ndarray): Предсказанные метки классов.
        y_proba (np.ndarray): Вероятности принадлежности к классу 1.
        model_name (str): Название модели (для информации).

    Returns:
        dict: Словарь с рассчитанными метриками.
    """
    metrics = {
        'specificity': calculate_specificity(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred), # Recall
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    return metrics

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # Масштабирование числовых признаков
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # One-Hot кодирование категориальных
    ])

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Создает пайплайн с препроцессором и моделью, обучает его.

    Args:
        model: Экземпляр модели машинного обучения (например, LogisticRegression).
        X_train (pd.DataFrame): Обучающие признаки.
        y_train (pd.Series): Обучающие целевые метки.

    Returns:
        Pipeline: Обученный пайплайн Scikit-learn.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor), # Шаг предобработки
        ('classifier', model)          # Шаг классификации
    ])
    pipeline.fit(X_train, y_train) # Обучение пайплайна
    return pipeline

def predict_churn(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Делает предсказания классов и вероятностей с помощью обученного пайплайна.

    Args:
        model (Pipeline): Обученный пайплайн.
        X (pd.DataFrame): Данные для предсказания.

    Returns:
        tuple[np.ndarray, np.ndarray]: Кортеж из предсказанных меток и вероятностей класса 1.
    """
    y_pred = model.predict(X) # Предсказание классов
    y_proba = model.predict_proba(X)[:, 1] # Предсказание вероятностей класса 1
    return y_pred, y_proba

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обертка для кэширования функции загрузки и предобработки данных.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Результат выполнения load_and_preprocess_data().
    """
    return load_and_preprocess_data()

# ==============================================
# ОСНОВНАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА ПРИЛОЖЕНИЯ
# ==============================================
def main():
    """
    Основная функция, запускающая Streamlit приложение для анализа оттока.
    Включает загрузку данных, фильтрацию, визуализацию и моделирование.
    """

    from pathlib import Path
    import base64

    logo_path = Path("assets/logo_white_font.png")
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode()

    st.markdown(
    f"""
    <style>
        .logo-corner {{
            position: fixed;
            right: 1rem;
            bottom: 1rem;
            width: 100px;
            cursor: pointer;
            opacity: 0.4;
            z-index: 100;
            transition: all 0.3s ease;
            }}
        
        .logo-corner:hover {{
            transform: scale(1.1);
            opacity: 1;
        }}

        .logo-corner:active {{
            transform: scale(0.9);
            opacity: 0.65;
            }}
    </style>

    <img class="logo-corner"
        src="data:image/png;base64,{logo_base64}">
    """,
    unsafe_allow_html=True,
    )

    st.title("Анализ оттока клиентов")

    try:
        df = load_data()[0]
        
        required_columns = ['customer_state', 'payment_value', 'review_score', 
                          'RFM_Score', 'is_churned']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            st.error(f"Отсутствуют ключевые колонки: {missing_cols}")
            st.stop()
        
        st.sidebar.header("🔍 Фильтры")
        
        # получаем все штаты
        all_states = sorted(df['customer_state'].unique())
        # инициализируем session_state единожды
        if 'selected_states' not in st.session_state:
            st.session_state.selected_states = all_states.copy()

        if st.sidebar.button("Выбрать все", key="select_all"):
            st.session_state.selected_states = all_states.copy()
            st.rerun()

        # сам multiselect теперь ниже
        selected_states = st.sidebar.multiselect(
            "Выберите штаты:",
            options=all_states,
            key="selected_states"
        )
        
        rfm_range = st.sidebar.slider(
            "Диапазон RFM Score:",
            min_value=int(df['RFM_Score'].min()),
            max_value=int(df['RFM_Score'].max()),
            value=(int(df['RFM_Score'].min()), int(df['RFM_Score'].max()))
        )
        
        filtered_df = df[
            (df['customer_state'].isin(selected_states)) &
            (df['RFM_Score'].between(rfm_range[0], rfm_range[1]))
        ]

        # Это штука делает текст вкладок больше
        st.markdown(
        """
        <style>
        /* Target the tab buttons and their inner text */
        div[data-baseweb="tab-list"] button[role="tab"] > div {
            font-size: 20px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )

        st.markdown(
        """
        <style>
        /* Target the tab list container to enable flexbox and take full width */
        div[data-baseweb="tab-list"] {
            display: flex;
            width: 100%;
        }

        /* Target individual tab buttons to make them grow equally and center text */
        div[data-baseweb="tab-list"] button[role="tab"] {
            flex-grow: 1;
            justify-content: center; /* Center the text inside the tab button */
        }

        /* Target the inner div containing the text for font size */
        div[data-baseweb="tab-list"] button[role="tab"] > div {
            font-size: 20px !important; /* Keep existing font size rule */
        }
        </style>
        """,
        unsafe_allow_html=True,
        )
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Обзор", "Древовидная карта", "Распределения", "Детали", "Модели", "Карта", "Power BI"])
        
        # Вкладка 1: Обзор
        with tab1:
            st.markdown("""
            <style>
            * {
                scroll-behavior: smooth;
            }
            .toc-box {
                border: 1px solid #FFFFFF;
                border-radius: 8px;
                padding: 24px; 
                margin-top: 24px;
                margin-bottom: 36px;
                background-color: #0f1116;
            }
            .toc-box h4 {
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 18px;
            }
            .toc-box ul {
                padding-left: 0;
                margin-bottom: 0;
            }
            .toc-box li {
                margin-bottom: 10px;
                font-size: 16px;
            }
            .toc-box a {
                text-decoration: none;
                color: #549ecd;
                font-weight: 600;
            }
            .toc-box a:hover {
                text-decoration: underline; /* Underline on hover */
                color: #1c83e1; /* Darker blue on hover */
            }
            </style>

            <div class="toc-box">
                <h4>Содержание</h4>
                <ul>
                    <li><a href="#top-churn-states">Топ штатов по оттоку</a></li>
                    <li><a href="#customer-distribution-by-state">Распределение клиентов по штатам</a></li>
                    <li><a href="#churn-by-state-and-category">Уровень оттока по штатам и категориям товаров</a></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Всего клиентов", filtered_df['customer_unique_id'].nunique())
            col2.metric("Уровень оттока", 
                    f"{filtered_df['is_churned'].mean():.1%}")
            col3.metric("Средний RFM Score", 
                    f"{filtered_df['RFM_Score'].mean():.1f}")

            st.markdown("<a name='top-churn-states'></a>", unsafe_allow_html=True)
            # График топ регионов по оттоку
            st.subheader("Топ штатов по оттоку")

            # Подготовка данных
            churn_by_state = (filtered_df.groupby('customer_state')['is_churned']
                            .mean()
                            .sort_values(ascending=False)
                            .reset_index())
            churn_by_state.columns = ['Штат', 'Уровень оттока']

            # Создание интерактивного графика Plotly
            fig = px.bar(
                churn_by_state,
                x='Штат',
                y='Уровень оттока',
                color='Уровень оттока',
                color_continuous_scale='Blues',
                text='Уровень оттока'
            )

            # Настройка внешнего вида
            fig.update_traces(
                texttemplate='%{text:.1%}',
                textposition='outside'
            )
            
            fig.update_layout(
                xaxis_title='Штат',
                yaxis_title='',
                yaxis=dict(showticklabels=False, showgrid=False),
                yaxis_tickformat='.0%',
                coloraxis_showscale=False,
                hovermode='x',
                height=500
            )

            # Отображение графика в Streamlit
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<a name='customer-distribution-by-state'></a>", unsafe_allow_html=True)
            # Дополнительный график с распределением клиентов по штатам
            st.subheader("Распределение клиентов по штатам")

            # Подготовка данных
            customers_by_state = (filtered_df['customer_state']
                                .value_counts()
                                .sort_values(ascending=False)
                                .reset_index())
            customers_by_state.columns = ['Штат', 'Количество клиентов']

            # Создание графика
            fig2 = px.bar(
                customers_by_state,
                x='Штат',
                y='Количество клиентов',
                color='Количество клиентов',
                color_continuous_scale='Blues',
                text='Количество клиентов',
            )

            # Настройка внешнего вида
            fig2.update_traces(
                textposition='outside'
            )
            fig2.update_layout(
                xaxis_title='Штат',
                yaxis_title='',
                yaxis=dict(showticklabels=False, showgrid=False),
                coloraxis_showscale=False,
                hovermode='x',
                height=500
            )

            st.plotly_chart(fig2, use_container_width=True)

            merge6 = load_data()[2]

            heatmap_data = merge6.groupby(['customer_state', 'product_category_name'])['is_churned'].mean().reset_index()

            fig = px.density_heatmap(
            heatmap_data,
            x='customer_state',
            y='product_category_name',
            z='is_churned',
            color_continuous_scale='Blues',
            labels={'is_churned': '% оттока'},
            width=1400,
            height=1000
            )

            fig.update_layout(
                xaxis_title="Штат",
                yaxis_title="Категория товара",
                hovermode='closest',

                font=dict(size=18),

                xaxis=dict(
                    title_font=dict(size=20),
                    tickfont=dict(size=16)
                ),
                yaxis=dict(
                    title_font=dict(size=20),
                    tickfont=dict(size=16)
                )
            )

            st.markdown("<a name='churn-by-state-and-category'></a>", unsafe_allow_html=True)
            st.subheader("Уровень оттока по штатам и категориям товаров")
            st.plotly_chart(fig, use_container_width=True)

        # Вкладка 2: Древовидная карта по сегментам
        with tab2:
            st.subheader("RFM сегменты")

            segt_map = {
                r'[1-2][1-2]': 'Спящие',
                r'[1-2][3-4]': 'В зоне риска',
                r'[1-2]5': 'Нельзя потерять',
                r'3[1-2]': 'На грани ухода',
                r'33': 'Требуют внимания',
                r'[3-4][4-5]': 'Лояльные клиенты',
                r'41': 'Перспективные',
                r'51': 'Новые клиенты',
                r'[4-5][2-3]': 'Potential loyalists',
                r'5[4-5]': 'Чемпионы'
            }

            rfm = load_data()[1]
            rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
            rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)

            fig3 = rfm.groupby('Segment').agg({'customer_unique_id': lambda x: len(x)}).reset_index()

            fig3.rename(columns={'customer_unique_id': 'Count'}, inplace=True)
            fig3['percent'] = (fig3['Count'] / fig3['Count'].sum()) * 100
            fig3['percent'] = fig3['percent'].round(1)

            fig3['display_text'] = fig3['Segment'] + '<br>Количество: ' + fig3['Count'].astype(str) + '<br>' + fig3['percent'].astype(str) + '%'

            colors = [
                '#1a4b7d',
                '#3a6ea5',
                '#5d8fc7',
                '#89b0d9',
                '#b5d0e8',
                '#d9e6f2',
                '#e6eef7',
                '#f2f7fb'
            ]

            fig = px.treemap(fig3, path=['Segment'], values='Count',
                            width=800, height=800,
                            )

            fig.update_traces(text=fig3['display_text'],
                            textinfo='text',  
                            textposition='middle center',
                            textfont_size=18,
                            hovertemplate=(
                    "<b>%{label}</b><br>" +
                    "Количество: %{value:,}<br>" +
                    "Процент: %{customdata[0]:.1f}%" +
                    "<extra></extra>"),
                customdata=fig3[['percent']])  

            fig.update_layout(
                treemapcolorway = colors, 
                margin=dict(t=50, l=25, r=25, b=25))

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Дендрограмма клиентских сегментов")

            # нормализация метрик
            customers_fix = pd.DataFrame()
            customers_fix["Recency"] = boxcox1p(rfm['Recency'], stats.boxcox_normmax(rfm['MonetaryValue'] + 1))
            customers_fix["Frequency"] = stats.boxcox(rfm['Frequency'])[0]
            customers_fix["Frequency"] = rfm['Frequency']
            customers_fix["MonetaryValue"] = stats.boxcox(rfm['MonetaryValue'])[0]

            scaler = StandardScaler()
            scaler.fit(customers_fix)
            customers_normalized = scaler.transform(customers_fix)

            # Создаем модель с 4 кластерами (выбрано по Elbow-методу) и добавляем метку кластера в rfm
            model = KMeans(n_clusters=4, random_state=42)
            model.fit(customers_normalized)
            rfm["Cluster"] = model.labels_

            # Определяем сегмент (Segment) и ценность клиента (Score).
            rfm['Score'] = 'Стандарт'
            rfm.loc[rfm['RFM_Score']>5,'Score'] = 'Бронза'
            rfm.loc[rfm['RFM_Score']>7,'Score'] = 'Серебро'
            rfm.loc[rfm['RFM_Score']>9,'Score'] = 'Золото'
            rfm.loc[rfm['RFM_Score']>10,'Score'] = 'Платина'

            # дендрограмма
            fig5 = rfm.groupby(['Cluster', 'Segment', 'Score']).agg({'customer_unique_id': lambda x: len(x)}).reset_index()

            fig5.rename(columns={'customer_unique_id': 'Count'}, inplace=True)
            fig5['percent'] = (fig5['Count'] / fig5['Count'].sum()) * 100
            fig5['percent'] = fig5['percent'].round(1)

            colors = [
                '#3a6ea5',
                '#5d8fc7',
                '#89b0d9',
                '#b5d0e8',
                '#d9e6f2',
                '#e6eef7',
                '#f2f7fb'
            ]

            fig = px.treemap(fig5, path=['Cluster', 'Segment', 'Score'], values='Count',
                            width=800, height=1000,
                            hover_data={'Segment': True, 'Count': True, 'percent': True})

            fig.update_traces(textinfo='label+text+value+percent root', textfont_size=18)
            fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

            fig.update_layout(
                treemapcolorway = colors,
                margin = dict(t=50, l=25, r=25, b=25))

            fig.data[0].textinfo = 'label+text+value+percent root'
            
            st.plotly_chart(fig, use_container_width=True)

        # Вкладка 3: Распределения
        with tab3:
            st.subheader("Распределение RFM Score")
            fig1 = px.histogram(filtered_df, x='RFM_Score', color='is_churned',
                              nbins=20, barmode='overlay')
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("Корреляция признаков")
            num_cols = ['payment_value', 'review_score', 'RFM_Score', 'is_churned']
            corr_matrix = filtered_df[num_cols].corr()
            fig2 = px.imshow(corr_matrix, text_auto=True,
                            width=800, height=600)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Вкладка 4: Детали по клиентам
        with tab4:
            st.subheader("Детали по клиентам")
            
            selected_customer = st.selectbox(
                "Выберите клиента:",
                options=filtered_df['customer_unique_id'].unique()
            )
            
            customer_data = filtered_df[filtered_df['customer_unique_id'] == selected_customer].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Регион", customer_data['customer_state'])
                st.metric("RFM Score", customer_data['RFM_Score'])
                st.metric("Сумма покупок", f"{customer_data['payment_value']:.2f}")
            
            with col2:
                st.metric("Оценка отзыва", customer_data['review_score'])
                is_churned = customer_data['is_churned'] == 1
                churn_label = "Да" if is_churned else "Нет"
                delta_text = "Высокий риск" if is_churned else "Низкий риск"
                st.metric("Статус оттока", 
                         churn_label,
                         delta=delta_text,
                         delta_color="inverse" if is_churned else "normal"
                         )
    
        # Вкладка 5: Модели
        with tab5:
            st.header("Прогнозирование оттока клиентов")
            
            X = filtered_df[numerical_features + categorical_features]
            y = filtered_df['is_churned']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            model_option = st.selectbox(
                "Выберите модель:",
                ("Logistic Regression", "Decision Tree", "XGBoost")
            )
            
            if st.button("Обучить модель"):
                with st.spinner("Обучение модели..."):
                    if model_option == "Logistic Regression":
                        model = LogisticRegression(
                            class_weight='balanced',
                            max_iter=1000,
                            random_state=42
                        )
                    elif model_option == "Decision Tree":
                        model = DecisionTreeClassifier(
                            max_depth=5,
                            min_samples_leaf=50,
                            class_weight='balanced',
                            random_state=42
                        )
                    elif model_option == "XGBoost":
                        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
                        model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            n_estimators=150,
                            max_depth=5,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            scale_pos_weight=scale_pos_weight,
                            random_state=42,
                            n_jobs=-1
                        )
                    
                    pipeline = train_model(model, X_train, y_train)
                    st.session_state.trained_pipeline = pipeline
                    st.session_state.model_option = model_option

                    y_pred, y_proba = predict_churn(pipeline, X_test)
                    
                    metrics = extended_classification_report(y_test, y_pred, y_proba, model_option)
                    
                    st.subheader("Метрики модели")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    col2.metric("F1 Score", f"{metrics['f1']:.4f}")
                    col3.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
                    col5.metric("Specificity", f"{metrics['specificity']:.4f}")
                    col6.metric("Precision", f"{metrics['precision']:.4f}")
                    
                    st.subheader("ROC-кривая")
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig = px.line(x=fpr, y=tpr, 
                                 labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                 title=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Матрица ошибок")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, 
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=['Not Churned', 'Churned'],
                                       y=['Not Churned', 'Churned'],
                                       text_auto=True)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    st.success(f"Модель {model_option} успешно обучена!")

            if 'trained_pipeline' in st.session_state:
                st.subheader("Скачать обученную модель")
                pipeline_to_save = st.session_state.trained_pipeline
                model_option_to_save = st.session_state.model_option
                model_filename = f'{model_option_to_save.replace(" ", "_").lower()}_model.pkl'
                
                from io import BytesIO
                model_bytes = BytesIO()
                joblib.dump(pipeline_to_save, model_bytes)
                model_bytes.seek(0)

                st.download_button(
                    label=f"Скачать {model_option_to_save.replace(" ", "_")}.pkl",
                    data=model_bytes,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )

            st.subheader("Прогноз для конкретного клиента")
            customer_id_predict = st.selectbox(
                "Выберите клиента для прогноза:",
                options=filtered_df['customer_unique_id'].unique(),
                key="customer_predict_select"
            )
            
            if st.button("Сделать прогноз"):
                if 'trained_pipeline' not in st.session_state:
                    st.error("Модель не обучена. Пожалуйста, сначала обучите модель на этой вкладке.")
                else:
                    try:
                        pipeline = st.session_state.trained_pipeline
                        current_model_option = st.session_state.model_option
                        
                        customer_data = filtered_df[filtered_df['customer_unique_id'] == customer_id_predict].iloc[0:1]
                        X_customer = customer_data[numerical_features + categorical_features]
                        
                        pred, proba = predict_churn(pipeline, X_customer)
                        
                        st.subheader("Результат прогноза")
                        col1, col2 = st.columns(2)
                        col1.metric("Вероятность оттока", f"{proba[0]:.2%}")

                        is_high_risk = pred[0] == 1
                        prognosis_label = "Высокий риск оттока" if is_high_risk else "Низкий риск оттока"
                        delta_display_text = "⚠️ Внимание" if is_high_risk else "✅ Норма"

                        col2.metric("Прогноз",
                                   prognosis_label,
                                   delta=delta_display_text,
                                   delta_color="inverse" if is_high_risk else "normal"
                                   )

                        if current_model_option in ["Decision Tree", "XGBoost"]:
                            st.subheader("Важность признаков")
                            try:
                                classifier = pipeline.named_steps['classifier']
                                
                                preprocessor_step = pipeline.named_steps['preprocessor']
                                
                                ohe_feature_names = preprocessor_step.transformers_[1][1]\
                                    .get_feature_names_out(categorical_features)
                                
                                feature_names = numerical_features + list(ohe_feature_names)
                                
                                importances = classifier.feature_importances_
                                
                                importance_df = pd.DataFrame({
                                    'feature': feature_names,
                                    'importance': importances
                                }).sort_values(by='importance', ascending=False)
                                
                                top_n = 15
                                fig_importance = px.bar(
                                    importance_df.head(top_n),
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title=f'Топ-{top_n} важных признаков для {current_model_option}'
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                            except AttributeError:
                                st.warning(f"Не удалось получить важность признаков для {current_model_option}.")
                            except Exception as e:
                                st.error(f"Ошибка при расчете важности признаков: {str(e)}")

                    except Exception as e:
                        st.error(f"Ошибка при прогнозировании: {str(e)}")

            # Вкладка 6: Карта
            with tab6:
                geo    = load_geo()
                orders = load_orders()

                T, PX = 0.05, 1

                # ─── HoloViews → Bokeh opts (FULL‑WIDTH, RESPONSIVE) ───────────────────
                hv.extension("bokeh", logo=False)

                gv.opts.defaults(
                    gv.opts.Overlay(
                        height   = 700,
                        toolbar  = "above",
                        xaxis    = None,
                        yaxis    = None,
                        responsive=True               # это важно
                    ),
                    gv.opts.QuadMesh(
                        tools      = ["hover"],
                        colorbar   = True,
                        alpha      = 0,
                        hover_alpha= 0.2,
                    ),
                )

                # Вспомогательные функции

                def make_map(df, col, title, cmap):
                    """
                    returns a holoviews overlay (bokeh renderable) showing df[col]
                    """
                    url = "https://server.arcgisonline.com/ArcGIS/rest/services/" \
                        "Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"
                    tiles  = gv.WMTS(WMTSTileSource(url=url))
                    pts    = hv.Points(gv.Dataset(df, kdims=["x", "y"], vdims=[col]))
                    shade  = datashade(pts, element_type=gv.Image, aggregator=ds.mean(col), cmap=cmap)
                    spread = dynspread(shade, threshold=T, max_px=PX)
                    hover  = hv.util.Dynamic(
                        rasterize(pts, aggregator=ds.mean(col), width=50, height=25, streams=[RangeXY]),
                        operation=hv.QuadMesh,
                    ).opts(cmap=cmap)
                    return (tiles * spread * hover).relabel(title)

                def clip_outliers(df):
                    return df[
                        df.x.between(df.x.quantile(0.001), df.x.quantile(0.999)) &
                        df.y.between(df.y.quantile(0.001), df.y.quantile(0.999))
                    ]

                def build_revenue_df(geo, orders):
                    geo3 = geo.set_index("geolocation_zip_code_prefix_3_digits").copy()
                    gp   = orders.groupby("customer_zip_code_prefix_3_digits")["price"].sum()
                    df   = geo3.join(gp)
                    df["revenue"] = df["price"].fillna(0) / 1_000          # k R$
                    return df

                def build_avg_ticket_df(geo, orders):
                    tickets = (
                        orders.groupby("order_id")
                            .agg({"price": "sum",
                                    "customer_zip_code_prefix_3_digits": "first"})
                    )
                    gp = tickets.groupby("customer_zip_code_prefix_3_digits")["price"].mean()
                    geo3 = geo.set_index("geolocation_zip_code_prefix_3_digits").copy()
                    df = geo3.join(gp)
                    df["avg_ticket"] = df["price"]
                    return df

                @st.cache_data(show_spinner=True)
                def build_freight_ratio_df(geo, orders):
                    tmp = (
                        orders.groupby("order_id")
                            .agg({"price": "sum",
                                    "freight_value": "sum",
                                    "customer_zip_code_prefix_3_digits": "first"})
                    )

                    tmp = tmp[tmp["price"] > 0]
                    tmp["freight_ratio"] = tmp["freight_value"] / tmp["price"]

                    gp = tmp.groupby("customer_zip_code_prefix_3_digits")["freight_ratio"].mean()

                    geo3 = geo.set_index("geolocation_zip_code_prefix_3_digits").copy()
                    return geo3.join(gp)

                # ───────────────────────────── sidebar selectors ─────────────────────────────
                st.header("📍 Фильтр карты")

                states = ["<вся Бразилия>"] + sorted(geo["geolocation_state"].unique())
                state  = st.selectbox("Штат (UF)", states, index=0)

                if state == "<вся Бразилия>":
                    sub_geo    = geo.copy()
                    sub_orders = orders.copy()
                    city       = "<все>"
                else:
                    sub_geo    = geo[geo["geolocation_state"] == state]
                    sub_orders = orders[orders["customer_state"] == state]
                    cities     = ["<все>"] + sorted(sub_geo["geolocation_city"].str.lower().unique())
                    city       = st.selectbox("Город", cities, index=0)
                    if city != "<все>":
                        sub_geo    = sub_geo[sub_geo["geolocation_city"].str.lower() == city]
                        sub_orders = sub_orders[sub_orders["customer_city"].str.lower() == city]

                sub_geo = clip_outliers(sub_geo)

                # ───────────────────────────── revenue map ─────────────────────────────
                st.subheader("💰 Доход от заказов (тыс. R$)")

                revenue_df = build_revenue_df(sub_geo, sub_orders)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("ZIP‑префиксов", len(revenue_df))
                with col2:
                    st.metric("Всего (k R$)", int(revenue_df["revenue"].sum()))

                with st.spinner("🖼️ рендер дохода …"):
                    fig_rev = hv.render(make_map(revenue_df, "revenue",
                                                "Orders Revenue (k R$)", fire),
                                        backend="bokeh")
                fig_rev.sizing_mode = "stretch_width"
                streamlit_bokeh(fig_rev, use_container_width=True)

                # ───────────────────────────── avg ticket map ─────────────────────────────
                st.subheader("🎟️ Средний чек заказа (R$)")

                avg_df = build_avg_ticket_df(sub_geo, sub_orders)

                st.metric("Средний чек (R$)", f"{avg_df['avg_ticket'].mean():.2f}")

                with st.spinner("🖼️ рендер среднего чека …"):
                    fig_avg = hv.render(make_map(avg_df, "avg_ticket",
                                                "Orders Average Ticket (R$)", bgy),
                                        backend="bokeh")
                fig_avg.sizing_mode = "stretch_width"
                streamlit_bokeh(fig_avg, use_container_width=True)

                # ───────────────────────────── freight‑ratio map ─────────────────────────────
                freight_df = build_freight_ratio_df(sub_geo, sub_orders)

                st.subheader("🚚 Среднее отношение доставки к цене")

                st.metric("Среднее соотношение доставка/чек", f"{freight_df['freight_ratio'].mean():.2%}")

                with st.spinner("🖼️ рендер freight-ratio map …"):
                    fig_freight = hv.render(
                        make_map(freight_df, "freight_ratio",
                                "Orders Average Freight Ratio", kr),
                        backend="bokeh"
                    )
                fig_freight.sizing_mode = "stretch_width"
                streamlit_bokeh(fig_freight, use_container_width=True)

            with tab7:
                import streamlit.components.v1 as components

                PBI_EMBED_URL = "https://app.powerbi.com/reportEmbed?reportId=67889dcb-3cc3-4b1c-9a22-14472a156917&autoAuth=true&ctid=dfe014b9-885d-4e4a-8eb4-597464b165c5"

                components.html(f"""
                    <iframe 
                        width="100%" 
                        height="800" 
                        src="{PBI_EMBED_URL}" 
                        frameborder="0" 
                        allowFullScreen="true">
                    </iframe>
                """, height=800)


    except Exception as e:
        st.error(f"Ошибка при загрузке или обработке данных: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()