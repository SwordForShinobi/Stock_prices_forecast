import streamlit as st
import pandas as pd
import pickle
import zipfile
import os
from io import BytesIO
import plotly.express as px

# Заголовок приложения
st.title("📊 Прогноз цен на нефтепродукты")
st.write("Выберите нефтепродукт, введите текущие значения признаков и получите прогноз на 30 дней вперёд.")

# Словарь: нефтепродукт → имя архива и признаки
PRODUCT_CONFIG = {
    "АИ-98": {
        "zip": "models/ai98_models.zip",
        "features": [
            'Цена Brent', 'Откр. Brent', 'Макс. Brent', 'Мин. Brent',
            'Курс доллара', 'Ключевая ставка', 'Инфляция', 'Год', 'Месяц',
            'Цена биржа АИ-98'
        ]
    },
    "АИ-95": {
        "zip": "models/ai95_models.zip",
        "features": [
            'Цена Brent', 'Откр. Brent', 'Макс. Brent', 'Мин. Brent',
            'Курс доллара', 'Ключевая ставка', 'Инфляция', 'Год', 'Месяц',
            'Цена биржа АИ-95'
        ]
    },
    "АИ-92": {
        "zip": "models/ai92_models.zip",
        "features": [
            'Цена Brent', 'Откр. Brent', 'Макс. Brent', 'Мин. Brent',
            'Курс доллара', 'Ключевая ставка', 'Инфляция', 'Год', 'Месяц',
            'Цена биржа АИ-92'
        ]
    },
    "ДТ реализация": {
        "zip": "models/dt_models.zip",
        "features": [
            'ДТ реализация', 'Цена Brent', 'Откр. Brent', 'Макс. Brent',
            'Мин. Brent', 'Объем, К Brent', 'Курс доллара', 'Цена Urals',
            'Ключевая ставка', 'Инфляция', 'Год', 'Месяц'
        ]
    }
}


# Функция загрузки модели из архива по горизонту
def load_model_from_zip(zip_path, horizon):
    if not os.path.exists(zip_path):
        st.error(f"❌ Архив не найден: {zip_path}")
        return None
    with open(zip_path, 'rb') as f:
        archive = zipfile.ZipFile(BytesIO(f.read()))
        model_name = f"model_{horizon}.pkl"
        try:
            with archive.open(model_name) as model_file:
                return pickle.load(model_file)
        except KeyError:
            st.error(f"❌ Модель для {horizon} дня не найдена в архиве: {model_name}")
            return None


# Функция прогноза
def make_a_forecast(product_key, input_data):
    config = PRODUCT_CONFIG[product_key]
    predictions = {}

    data = pd.DataFrame([input_data])

    for horizon in range(1, 31):
        model = load_model_from_zip(config["zip"], horizon)
        if model is None:
            return None
        pred = model.predict(data)[0]
        predictions[f"{horizon} дн"] = round(float(pred), 2)

    return pd.Series(predictions)


# --- Интерфейс Streamlit ---
product = st.selectbox(
    "⛽ Выберите нефтепродукт",
    options=list(PRODUCT_CONFIG.keys())
)

config = PRODUCT_CONFIG[product]
st.subheader(f"Введите данные для прогноза: {product}")

input_data = {}
cols = st.columns(2)

# Год и месяц — отдельно
year = st.slider("Год", 2020, 2030, 2025)
month = st.slider("Месяц", 1, 12, 6)

for i, param in enumerate(config["features"]):
    if param == "Год":
        input_data[param] = year
    elif param == "Месяц":
        input_data[param] = month
    else:
        value = st.number_input(f"{param}", value=0.0, step=0.1, format="%.3f")
        input_data[param] = value

# Кнопка прогноза
if st.button("🚀 Получить прогноз"):
    with st.spinner("Прогнозируем..."):
        predictions = make_a_forecast(product, input_data)

        if predictions is not None:
            # Таблица
            st.subheader("📈 Прогноз на 30 дней")
            df_pred = pd.DataFrame([predictions]).T
            df_pred.columns = ["Цена"]
            df_pred.index.name = "День"
            st.dataframe(df_pred.style.format({"Цена": "{:.2f}"}))

            # График
            fig = px.line(
                df_pred,
                y="Цена",
                title=f"Прогноз цены на {product} (на 30 дней вперёд)",
                labels={"index": "День", "value": "Цена"},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Подготовка файлов для скачивания
            csv = df_pred.to_csv()

            # Создание Excel-файла в памяти
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_pred.to_excel(writer, sheet_name="Прогноз")
            excel_buffer.seek(0)

            # Кнопки скачивания
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Скачать как CSV",
                    data=csv,
                    file_name=f"{product}_forecast_30_days.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="📘 Скачать как XLSX",
                    data=excel_buffer,
                    file_name=f"{product}_forecast_30_days.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:

            st.error("Не удалось получить прогноз. Проверьте наличие моделей.")

