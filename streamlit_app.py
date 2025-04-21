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
from colorcet import rainbow, fire
from bokeh.models import WMTSTileSource
from streamlit_bokeh import streamlit_bokeh

import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from scipy import stats


st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка и предобработка данных
@st.cache_data
def load_and_preprocess_data():
    customers = pd.read_csv('customers.csv')
    orders = pd.read_csv('orders.csv')
    order_payments = pd.read_csv('order_payments.csv')
    order_reviews = pd.read_csv('order_reviews.csv')
    products = pd.read_csv('products.csv')
    product_category = pd.read_csv('product_category_name_translation.csv')
    sellers = pd.read_csv('sellers.csv')
    order_items = pd.read_csv('orders_items.csv')

    # Удаление лишних колонок
    for df in [order_payments, order_reviews, product_category, sellers, order_items]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

    # Преобразование дат в orders
    date_cols = [
        'order_purchase_timestamp', 
        'order_approved_at',
        'order_delivered_carrier_date', 
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Проверка на пропуски после конвертации
        if orders[col].isnull().any():
            st.warning(f"Обнаружены некорректные даты в колонке {col}")

    # После загрузки order_reviews.csv
    order_reviews = pd.read_csv('order_reviews.csv')
    
    # Преобразование временных меток отзывов
    order_reviews['review_creation_date'] = pd.to_datetime(
        order_reviews['review_creation_date'], 
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')

    order_reviews['review_answer_timestamp'] = pd.to_datetime(
        order_reviews['review_answer_timestamp'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')
    
    # Обе колонки с датами (оригинальная и дублирующая)
    order_items['shipping_limit_date'] = pd.to_datetime(
        order_items['shipping_limit_date'],
        format='%Y-%m-%d %H:%M:%S', 
        errors='coerce')

    order_items['shipping_limit_date.1'] = pd.to_datetime(
        order_items['shipping_limit_date.1'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')
    
    # Объединение orders и customers
    orders = pd.merge(
        orders, 
        customers[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']], 
        on='customer_id', 
        how='inner'
    )
    orders.drop('customer_id', axis=1, inplace=True)

    # Добавление информации о количестве заказов клиента
    orders['quantity'] = orders.groupby('customer_unique_id')['order_id'].transform('nunique')

    # Объединение с отзывами
    merge1 = pd.merge(
        orders,
        order_reviews[['order_id', 'review_creation_date', 'review_score']],
        on='order_id',
        how='left'
    )
    
    # Удаление NA в ключевых столбцах
    merge1.dropna(subset=['review_creation_date', 'review_score'], inplace=True)
    merge1['review_creation_date'] = pd.to_datetime(merge1['review_creation_date'])

    # Расчет признаков оттока (ваша оригинальная логика)
    analysis_date = merge1['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # Сортируем по клиенту и дате заказа
    df_sorted = merge1.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    
    # Вычисляем разницу между заказами
    df_sorted['time_diff_days'] = df_sorted.groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    
    # Медиана интервалов между заказами
    customer_intervals = df_sorted.groupby('customer_unique_id')['time_diff_days'].median().reset_index()
    customer_intervals.rename(columns={'time_diff_days': 'median_interval_days'}, inplace=True)
    
    # Статистика по клиентам
    customer_stats = merge1.groupby('customer_unique_id').agg(
        last_order_date=('order_purchase_timestamp', 'max'),
        order_count=('order_id', 'nunique')
    ).reset_index()
    
    # Дней с последнего заказа
    customer_stats['days_since_last_order'] = (analysis_date - customer_stats['last_order_date']).dt.days
    
    # Объединяем с интервалами
    customer_features = pd.merge(customer_stats, customer_intervals, on='customer_unique_id', how='left')
    
    # Динамический порог оттока (ваша оригинальная логика)
    churn_threshold_multiplier = 2.0
    default_churn_days = customer_features['days_since_last_order'].mean()  # ~77 дней
    
    def calculate_dynamic_churn(row):
        if row['order_count'] == 1:
            return 1 if row['days_since_last_order'] > default_churn_days else 0
        elif pd.isna(row['median_interval_days']) or row['median_interval_days'] <= 0:
            return 1 if row['days_since_last_order'] > default_churn_days else 0
        else:
            personalized_threshold = row['median_interval_days'] * churn_threshold_multiplier
            return 1 if row['days_since_last_order'] > personalized_threshold else 0
    
    customer_features['is_churned'] = customer_features.apply(calculate_dynamic_churn, axis=1)
    
    # Объединяем с основными данными
    merge1 = pd.merge(
        merge1,
        customer_features[['customer_unique_id', 'is_churned', 'days_since_last_order', 'median_interval_days']],
        on='customer_unique_id',
        how='left'
    )

    # Обработка продуктов
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
    
    # Исправление похожих категорий
    category_corrections = {
        'home_confort': 'home_comfort',
        'home_appliances_2': 'home_appliances',
    }
    products['product_category_name'] = products['product_category_name'].replace(category_corrections)

    # Объединение с товарами и продавцами
    merge2 = pd.merge(
        products,
        order_items[['order_id', 'product_id', 'seller_id', 'price']],
        on='product_id',
        how='left'
    ).drop_duplicates()

    merge3 = pd.merge(
        merge2,
        sellers[['seller_id', 'seller_city', 'seller_state']],
        on='seller_id',
        how='left'
    )

    # Объединение всех данных
    merge4 = pd.merge(
        merge1,
        merge3,
        on='order_id',
        how='left'
    )

    # Добавление платежей
    merge5 = pd.merge(
        merge4,
        order_payments[['order_id', 'payment_type', 'payment_installments', 'payment_value']],
        on='order_id',
        how='left'
    ).drop_duplicates()

    # Удаление NA в ключевых столбцах
    merge5 = merge5.dropna(subset=['product_id', 'payment_type'])

    # Расчет дополнительных признаков
    merge5['approval_delay'] = (merge5['order_approved_at'] - merge5['order_purchase_timestamp']).dt.total_seconds()
    merge5['delivery_time_to_customer_days'] = (merge5['order_delivered_customer_date'] - merge5['order_purchase_timestamp']).dt.days
    merge5['delivery_delay'] = (merge5['order_delivered_customer_date'] - merge5['order_estimated_delivery_date']).dt.days
    merge5['estimated_delivery_time'] = (merge5['order_estimated_delivery_date'] - merge5['order_purchase_timestamp']).dt.days

    # Удаление нелогичных строк
    merge5 = merge5[
        (merge5['order_approved_at'] < merge5['order_delivered_customer_date']) &
        (merge5['order_approved_at'] < merge5['order_delivered_carrier_date']) &
        (merge5['order_delivered_carrier_date'] < merge5['order_delivered_customer_date'])
    ]

    # RFM анализ
    rfm = merge5.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': 'max',
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).reset_index()
    
    rfm.columns = ['customer_unique_id', 'last_order', 'Frequency', 'MonetaryValue']
    rfm['Recency'] = (analysis_date - rfm['last_order']).dt.days
    rfm.drop('last_order', axis=1, inplace=True)
    
    # RFM Scores
    quantiles = rfm[['Recency', 'Frequency', 'MonetaryValue']].quantile(q=[0.2, 0.4, 0.6, 0.8]).to_dict()
    
    def RScore(x, p, d):
        if x <= d[p][0.2]: return 5
        elif x <= d[p][0.4]: return 4
        elif x <= d[p][0.6]: return 3
        elif x <= d[p][0.8]: return 2
        else: return 1
    
    def FMScore(x, p, d):
        if x <= d[p][0.2]: return 1
        elif x <= d[p][0.4]: return 2
        elif x <= d[p][0.6]: return 3
        elif x <= d[p][0.8]: return 4
        else: return 5
    
    rfm['R'] = rfm['Recency'].apply(RScore, args=('Recency', quantiles))
    rfm['F'] = rfm['Frequency'].apply(FMScore, args=('Frequency', quantiles))
    rfm['M'] = rfm['MonetaryValue'].apply(FMScore, args=('MonetaryValue', quantiles))
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
    
    # Объединение RFM с основными данными
    merge6 = pd.merge(
        merge5,
        rfm[['customer_unique_id', 'Frequency', 'MonetaryValue', 'Recency', 'R', 'F', 'M', 'RFM_Score']],
        on='customer_unique_id',
        how='left'
    )

    
    # Выбор финальных признаков
    features = ['customer_unique_id',        'customer_city', 'customer_state', 'order_status', 'payment_type',
        'payment_installments', 'payment_value', 'review_score', 'price',
        'product_weight_g', 'seller_city', 'seller_state', 'product_category_name',
        'approval_delay', 'delivery_time_to_customer_days', 'estimated_delivery_time',
        'delivery_delay', 'RFM_Score', 'is_churned'
    ]
    merge6.dropna(subset=features, inplace=True)
    
    return merge6[features]

@st.cache_data(show_spinner=True)
def load_and_prepare_geo():
    geo = pd.read_csv("geolocation.csv", dtype={'geolocation_zip_code_prefix': str})

    for n in range(1, 5):
        geo[f'geolocation_zip_code_prefix_{n}_digits'] = geo['geolocation_zip_code_prefix'].str[:n]

    geo = geo.query(
        "geolocation_lat <= 5.27438888 and geolocation_lng >= -73.98283055 "
        "and geolocation_lat >= -33.75116944 and geolocation_lng <= -34.79314722"
    )

    geo['x'], geo['y'] = webm(geo.geolocation_lng, geo.geolocation_lat)

    int_cols = [c for c in geo.columns if 'prefix' in c]
    geo[int_cols] = geo[int_cols].astype(int)

    return geo

@st.cache_data(show_spinner=True)
def load_orders():
    orders_df      = pd.read_csv("orders.csv")
    order_items    = pd.read_csv("orders_items.csv")
    order_reviews  = pd.read_csv("order_reviews.csv")
    customers      = pd.read_csv("customers.csv",
                                 dtype={'customer_zip_code_prefix': str})

    # customer zip prefix – 3 digits
    customers['customer_zip_code_prefix_3_digits'] = (
        customers['customer_zip_code_prefix'].str[:3].astype(int)
    )

    # merge the four tables into one big fat orders frame
    orders = (
        orders_df
        .merge(order_items, on='order_id')
        .merge(customers,  on='customer_id')
        .merge(order_reviews, on='order_id')
    )

    return orders



# Загрузка предобработанных данных с фичами
# Добавим числовые и категориальные признаки (замените на ваши реальные признаки)
numerical_features = ['payment_value','review_score','price','product_weight_g',
                  'approval_delay', 'RFM_Score']
categorical_features = ['customer_city','customer_state','order_status','payment_type',
                      'payment_installments','seller_city','seller_state','product_category_name']

# Функция для расчета специфичности
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Функция для расширенного отчета
def extended_classification_report(y_true, y_pred, y_proba, model_name):
    metrics = {
        'specificity': calculate_specificity(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    return metrics

# Создание пайплайна для обработки данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Функция для обучения модели
def train_model(model, X_train, y_train):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Функция для предсказания
def predict_churn(model, X):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba

@st.cache_data
def load_data():
    return load_and_preprocess_data()
# Добавляем вкладку для моделей в main()

def main():
    st.title("📊 Анализ оттока клиентов")
    
    # Загрузка данных
    try:
        df = load_data()
        
        # Проверка ключевых колонок
        required_columns = ['customer_state', 'payment_value', 'review_score', 
                          'RFM_Score', 'is_churned']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            st.error(f"Отсутствуют ключевые колонки: {missing_cols}")
            st.stop()
        
        # Боковая панель с фильтрами
        st.sidebar.header("🔍 Фильтры")
        
        # Фильтр по региону
        selected_states = st.sidebar.multiselect(
            "Выберите регионы:",
            options=df['customer_state'].unique(),
            default=df['customer_state'].unique()
        )
        
        # Фильтр по RFM Score
        rfm_range = st.sidebar.slider(
            "Диапазон RFM Score:",
            min_value=int(df['RFM_Score'].min()),
            max_value=int(df['RFM_Score'].max()),
            value=(int(df['RFM_Score'].min()), int(df['RFM_Score'].max()))
        )
        
        # Применение фильтров
        filtered_df = df[
            (df['customer_state'].isin(selected_states)) &
            (df['RFM_Score'].between(rfm_range[0], rfm_range[1]))
        ]
        
        # Основная панель - добавляем вкладку для моделей
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Обзор", "📊 Распределения", "🔍 Детали", "🤖 Модели", "🗺️ Карта"])
        
        # ... остальные вкладки ...
        with tab1:
            # Ключевые метрики
            col1, col2, col3 = st.columns(3)
            col1.metric("Всего клиентов", filtered_df['customer_unique_id'].nunique())
            col2.metric("Уровень оттока", 
                       f"{filtered_df['is_churned'].mean():.1%}",
                       help="Доля клиентов с is_churned=1")
            col3.metric("Средний RFM Score", 
                       f"{filtered_df['RFM_Score'].mean():.1f}")
            
            # Топ регионов по оттоку
            st.subheader("Топ регионов по оттоку")
            churn_by_state = filtered_df.groupby('customer_state')['is_churned'].mean().sort_values(ascending=False)
            st.bar_chart(churn_by_state)
        
        with tab2:
            # Распределение RFM Score
            st.subheader("Распределение RFM Score")
            fig1 = px.histogram(filtered_df, x='RFM_Score', color='is_churned',
                              nbins=20, barmode='overlay')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Корреляция признаков
            st.subheader("Корреляция признаков")
            num_cols = ['payment_value', 'review_score', 'RFM_Score', 'is_churned']
            corr_matrix = filtered_df[num_cols].corr()
            fig2 = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            # Детализация по клиентам
            st.subheader("Детали по клиентам")
            
            # Выбор клиента для анализа
            selected_customer = st.selectbox(
                "Выберите клиента:",
                options=filtered_df['customer_unique_id'].unique()
            )
            
            # Данные выбранного клиента
            customer_data = filtered_df[filtered_df['customer_unique_id'] == selected_customer].iloc[0]
            
            # Отображение информации
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
    
  
        with tab4:
            st.header("Прогнозирование оттока клиентов")
            
            # Разделяем данные
            X = filtered_df[numerical_features + categorical_features]
            y = filtered_df['is_churned']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Выбор модели
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
                    
                    # Обучение модели
                    pipeline = train_model(model, X_train, y_train)
                    st.session_state.trained_pipeline = pipeline  # Сохраняем в сессию
                    st.session_state.model_option = model_option # Сохраняем имя модели

                    # Предсказания
                    y_pred, y_proba = predict_churn(pipeline, X_test)
                    
                    # Оценка модели
                    metrics = extended_classification_report(y_test, y_pred, y_proba, model_option)
                    
                    # Отображение метрик
                    st.subheader("Метрики модели")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    col2.metric("F1 Score", f"{metrics['f1']:.4f}")
                    col3.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
                    col5.metric("Specificity", f"{metrics['specificity']:.4f}")
                    col6.metric("Precision", f"{metrics['precision']:.4f}")
                    
                    # Визуализация ROC-кривой
                    st.subheader("ROC-кривая")
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig = px.line(x=fpr, y=tpr, 
                                 labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                 title=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Матрица ошибок
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
                
                # Сериализуем модель для скачивания
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

            # Предсказание для конкретного клиента
            st.subheader("Прогноз для конкретного клиента")
            customer_id_predict = st.selectbox(
                "Выберите клиента для прогноза:",
                options=filtered_df['customer_unique_id'].unique(),
                key="customer_predict_select" # Добавляем уникальный ключ
            )
            
            if st.button("Сделать прогноз"):
                # Проверяем, есть ли обученная модель в сессии
                if 'trained_pipeline' not in st.session_state:
                    st.error("Модель не обучена. Пожалуйста, сначала обучите модель на этой вкладке.")
                else:
                    try:
                        # Загружаем модель из сессии
                        pipeline = st.session_state.trained_pipeline
                        current_model_option = st.session_state.model_option
                        
                        # Получаем данные клиента
                        customer_data = filtered_df[filtered_df['customer_unique_id'] == customer_id_predict].iloc[0:1]
                        X_customer = customer_data[numerical_features + categorical_features]
                        
                        # Делаем предсказание
                        pred, proba = predict_churn(pipeline, X_customer)
                        
                        # Отображаем результат
                        st.subheader("Результат прогноза")
                        col1, col2 = st.columns(2)
                        col1.metric("Вероятность оттока", f"{proba[0]:.2%}")

                        is_high_risk = pred[0] == 1
                        prognosis_label = "Высокий риск оттока" if is_high_risk else "Низкий риск оттока"
                        delta_display_text = "⚠️ Внимание" if is_high_risk else "✅ Норма"

                        col2.metric("Прогноз",
                                   prognosis_label,
                                   delta=delta_display_text,
                                   # Set color explicitly: red for high risk, green for low risk
                                   delta_color="inverse" if is_high_risk else "normal"
                                   )

                        # Важность признаков (для tree-based моделей)
                        if current_model_option in ["Decision Tree", "XGBoost"]:
                            st.subheader("Важность признаков")
                            try:
                                # Получаем обученный классификатор из пайплайна
                                classifier = pipeline.named_steps['classifier']
                                
                                # Получаем препроцессор
                                preprocessor_step = pipeline.named_steps['preprocessor']
                                
                                # Получаем имена признаков после OneHotEncoding
                                ohe_feature_names = preprocessor_step.transformers_[1][1]\
                                    .get_feature_names_out(categorical_features)
                                
                                # Объединяем числовые и категориальные имена
                                feature_names = numerical_features + list(ohe_feature_names)
                                
                                # Получаем важность признаков
                                importances = classifier.feature_importances_
                                
                                # Создаем DataFrame для визуализации
                                importance_df = pd.DataFrame({
                                    'feature': feature_names,
                                    'importance': importances
                                }).sort_values(by='importance', ascending=False)
                                
                                # Отображаем топ-N признаков
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

            with tab5:
                st.subheader("Доход от заказов (тыс. реалов)")

                hv.extension('bokeh', logo=False)

                overlay_opts = dict(width=800, height=600, toolbar='above', xaxis=None, yaxis=None)
                quad_opts = dict(tools=['hover'], colorbar=True, alpha=0, hover_alpha=0.2)
                gv.opts.defaults(
                    gv.opts.Overlay(**overlay_opts),
                    gv.opts.QuadMesh(**quad_opts)
                )

                brazil = load_and_prepare_geo()

                # column the aggregation will use
                agg_name = 'geolocation_zip_code_prefix'

                # ---------------------------------------------------------------------------------
                # 2. holoviews + datashader plot builder
                # ---------------------------------------------------------------------------------
                T  = 0.05      # dynspread threshold
                PX = 1         # dynspread px growth

                def plotted_zipcodes(df, agg_name=agg_name, cmap=rainbow):
                    # background tiles (dark grey canvas)
                    url = (
                        "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"
                    )
                    geomap  = gv.WMTS(WMTSTileSource(url=url))

                    # points in projected coords
                    points  = hv.Points(gv.Dataset(df, kdims=['x', 'y'], vdims=[agg_name]))

                    # datashader aggregation
                    agg     = datashade(points,
                                        element_type=gv.Image,
                                        aggregator=ds.min(agg_name),   # same as the legacy call
                                        cmap=cmap)

                    # tweak visibility
                    zipcodes = dynspread(agg, threshold=T, max_px=PX)

                    # interactive hover layer (the small one keeps perf ok)
                    hover = hv.util.Dynamic(
                        rasterize(points,
                                aggregator=ds.min(agg_name),
                                width=50,
                                height=25,
                                streams=[RangeXY]
                        ),
                        operation=hv.QuadMesh
                    ).opts(cmap=cmap)

                    return geomap * zipcodes * hover

                with st.spinner('Загружаем карту...'):

                    bokeh_fig = hv.render(plotted_zipcodes(brazil), backend='bokeh')

                # streamlit_bokeh(bokeh_fig, use_container_width=True)

                def filter_data(df, level, name):

                    df = df[df[level] == name]
                    df = df[(df.x <= df.x.quantile(0.999)) & (df.x >= df.x.quantile(0.001))]
                    df = df[(df.y <= df.y.quantile(0.999)) & (df.y >= df.y.quantile(0.001))]
                    return df
                

                def build_revenue_df(geo, orders):

                    geo3 = geo.set_index('geolocation_zip_code_prefix_3_digits').copy()


                    gp = orders.groupby('customer_zip_code_prefix_3_digits')['price'].sum()

                    revenue = geo3.join(gp)
                    revenue['revenue'] = revenue['price'].fillna(0) / 1_000

                    return revenue
                
                def map_plot(df, agg_name='revenue', cmap=fire):
 
                    T, PX = 0.05, 1
                    url = (
                        "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"
                    )
                    geomap = gv.WMTS(WMTSTileSource(url=url))

                    points = hv.Points(gv.Dataset(df, kdims=['x', 'y'], vdims=[agg_name]))

                    agg = datashade(points,
                                    element_type=gv.Image,
                                    aggregator=ds.mean(agg_name),
                                    cmap=cmap)
                    img = dynspread(agg, threshold=T, max_px=PX)

                    hover = hv.util.Dynamic(
                        rasterize(points,
                                aggregator=ds.mean(agg_name),
                                width=50, height=25,
                                streams=[RangeXY]),
                        operation=hv.QuadMesh
                    ).opts(cmap=cmap)

                    return geomap * img * hover

                geo = load_and_prepare_geo()
                orders = load_orders()

                st.sidebar.header("📍 Карта")
                states = ['<all brazil>'] + sorted(geo['geolocation_state'].unique())
                state_choice = st.sidebar.selectbox("state (uf)", states, index=0)

                cities_in_state = sorted(geo.query("geolocation_state == @state_choice")['geolocation_city'].str.lower().unique())

                sub_geo = filter_data(geo, 'geolocation_state', state_choice)

                if state_choice == '<all brazil>':
                    sub_geo    = geo.copy()                     
                    sub_orders = orders.copy()                  
                else:
                    sub_geo    = filter_data(geo, 'geolocation_state', state_choice)
                    sub_orders = orders[orders['customer_state'] == state_choice]

                    cities_in_state = sorted(
                        sub_geo['geolocation_city'].str.lower().unique()
                    )
                    city_choice = st.sidebar.selectbox(
                        "city (optional)", ['<all>'] + cities_in_state
                    )

                    if city_choice != '<all>':
                        sub_geo = filter_data(sub_geo, 'geolocation_city', city_choice)
                        sub_orders = sub_orders[
                            sub_orders['customer_city'].str.lower() == city_choice
                        ]

                revenue_df = build_revenue_df(sub_geo, sub_orders)

                st.sidebar.markdown("### Общая статистика")
                st.sidebar.metric("nº zip prefixes", len(revenue_df))
                st.sidebar.metric("total revenue (k R$)", int(revenue_df['revenue'].sum()))

                if st.sidebar.checkbox("Показать `describe()`", value=False):
                    st.sidebar.dataframe(revenue_df['revenue'].describe().to_frame(), use_container_width=True)

                with st.spinner("Рендеринг datashader tiles"):
                    fig = hv.render(map_plot(revenue_df), backend='bokeh')
                streamlit_bokeh(fig, use_container_width=True)



    except Exception as e:
        st.error(f"Ошибка при загрузке или обработке данных: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()