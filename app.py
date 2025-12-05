import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Модель с параметрами в кэш
@st.cache_resource
def load_data():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_data()

model = data['model']
scaler = data['scaler']
encoder = data['encoder']
medians = data['medians']
df_train = data['train_data']

# Предобработка данных
def preprocess_input(df_input: pd.DataFrame):
    df = df_input.copy()
    
    # Удаление единиц измерения
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            
    # Заполнение пропусков (медианами из трейна)
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
            
    # Приведение к типу int
    if 'engine' in df.columns: df['engine'] = df['engine'].astype(int)
    if 'seats' in df.columns: df['seats'] = df['seats'].astype(int)
            
    # Удаление лишних столбцов (как в ноутбке)
    drop_cols = ['name', 'selling_price', 'torque']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Разделение на числовые и категориальные
    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    
    # Проверка целостности набора данных 
    missing_cols = set(cat_cols + num_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"В данных не хватает: {missing_cols}")

    # OneHotEncoding
    try:
        encoded_cats = encoder.transform(df[cat_cols])
    except Exception as e:
        raise ValueError(f"Ошибка кодирования категорий. Проверьте значения: {e}")
        
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    
    # Сборка (Числа + OHE)
    df_final = pd.concat([df[num_cols], encoded_df], axis=1)
    
    # Стандартизация
    df_scaled = pd.DataFrame(scaler.transform(df_final), columns=df_final.columns, index=df_final.index)
    
    return df_scaled

# ИНТЕРФЕЙС
st.title('🚘 Car Price Prediction Service')
st.markdown('#### **Интерактивный сервис для прогноза стоимости автомобилей**')

# Боковая панель для навигации
page = st.sidebar.selectbox("Выберите режим:", ["EDA", "Интерпретация модели", "Прогноз цены"])

# СТРАНИЦА 1: EDA 
if page == "EDA":
    st.header("Разведочный анализ данных")
    st.write("Здесь представлены основные распределения и зависимости в данных обучающей выборки.")
    
    # Распределение цены
    st.subheader("1.1 – Распределение целевой переменной")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.histplot(df_train['selling_price'], kde=True, ax=ax, color='blue')
    ax.set_ylabel('count')
    st.pyplot(fig)

    # Распределение цены (логарифмированной)
    st.subheader("1.2 – Распределение целевой переменной (логарифм)")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.histplot(np.log(df_train['selling_price']), kde=True, ax=ax, color='green')
    ax.set_xlabel('log(selling_price)')
    ax.set_ylabel('count')
    st.pyplot(fig)
    
    # Тепловая карта
    st.subheader("2 – Корреляция числовых признаков")
    fig, ax = plt.subplots(figsize=(14, 10))
    numeric_df = df_train.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    # Boxplot
    st.subheader("3.1 – Зависимость цены от типа трансмиссии")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.boxplot(x='transmission', y='selling_price', hue='transmission', data=df_train, ax=ax)
    ax.set_yscale('log')
    st.pyplot(fig)

    # Boxplot
    st.subheader("3.2 – Зависимость цены от типа топлива")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.boxplot(x='fuel', y='selling_price', hue='fuel', data=df_train, ax=ax)
    ax.set_yscale('log')
    st.pyplot(fig)

# СТРАНИЦА 2: ВЕСА МОДЕЛИ
elif page == "Интерпретация модели":
    st.header("Интерпретация весов модели")
    st.write("Какие признаки значительно влияют на стоимость?")
    
    # Название фич берем у скейлера
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_
    else:
        feature_names = [f"Feature {i}" for i in range(len(model.coef_))]

        
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': model.coef_
    })
    
    # Сортировка весов по модулю
    weights_df['Abs_Weight'] = weights_df['Weight'].abs()
    weights_df = weights_df.sort_values(by='Abs_Weight', ascending=False).head(10)
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='Weight', y='Feature', data=weights_df, ax=ax)
    ax.set_title('Top 10 Important Features')
    st.pyplot(fig)

# СТРАНИЦА 3: ПРОГНОЗ
elif page == "Прогноз цены":
    st.header("Прогноз стоимости")
    
    input_type = st.radio("Как вы хотите ввести данные?", ["Ввести вручную", "Загрузить CSV"])
    
    if input_type == "Ввести вручную":
        # Текущий год
        current_year = datetime.datetime.now().year

        # Форма ввода
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input(
                "Год выпуска", 
                min_value=1975, 
                # Год модели может быть на 1 больше текущего
                max_value=current_year + 1, 
                value=2015
            )
            km_driven = st.number_input(
                "Пробег (км)", 
                min_value=0, 
                max_value=1000000, 
                value=50000
            )
            mileage = st.number_input(
                "Расход/Пробег на ед. топлива (км/л)", 
                min_value=0.5, 
                max_value=50.0, 
                value=20.0
            )
            engine = st.number_input(
                "Объем двигателя (CC)", 
                min_value=200, 
                max_value=10000, 
                value=1248
            )
        with col2:
            max_power = st.number_input(
                "Мощность (bhp)", 
                min_value=100,
                max_value=2500, 
                value=150
            )
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
            fuel = st.selectbox("Тип топлива", ['Diesel', 'Petrol', 'CNG', 'LPG'])
            seller_type = st.selectbox("Продавец", ['Individual', 'Dealer', 'Trustmark Dealer'])
            transmission = st.selectbox("Коробка передач", ['Manual', 'Automatic'])
            owner = st.selectbox("Владелец", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
            
        if st.button("Рассчитать цену"):
            # DataFrame из введенных данных
            data_dict = {
                'year': [year], 'km_driven': [km_driven], 'mileage': [mileage],
                'engine': [engine], 'max_power': [max_power], 'seats': [seats],
                'fuel': [fuel], 'seller_type': [seller_type], 
                'transmission': [transmission], 'owner': [owner]
            }
            input_df = pd.DataFrame(data_dict)
            
            try:
                processed_df = preprocess_input(input_df)
                prediction_log = model.predict(processed_df)
                prediction_real = np.exp(prediction_log)[0]
                
                st.success(f"💰 Рекомендованная цена: {prediction_real:,.2f}")
            except Exception as e:
                st.error(f"Ошибка при обработке данных: {e}")

    else: # Загрузка CSV
        uploaded_file = st.file_uploader("Загрузите CSV файл с характеристиками автомобиля", type=["csv"])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Первые 5 строк вашего файла:")
            st.dataframe(input_df.head())
            
            if st.button("Прогнозировать цены"):
                try:
                    processed_df = preprocess_input(input_df)
                    predictions_log = model.predict(processed_df)
                    predictions_real = np.exp(predictions_log)
                    
                    input_df['predicted_price'] = predictions_real
                    st.write("Результаты:")
                    st.dataframe(input_df[['name', 'predicted_price']].head() if 'name' in input_df.columns else input_df.head())
                    
                    # Кнопка скачивания
                    csv = input_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Скачать результат CSV", csv, "predictions.csv", "text/csv")
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {e}")
                    st.warning("Убедитесь, что формат данных совпадает с тренировочным датасетом (mileage, engine и т.д.)")