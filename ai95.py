import streamlit as st
import pandas as pd
import pickle
import zipfile
import os
from io import BytesIO
import plotly.graph_objects as go

# --- Фичи ---
features = [
    'Цена Brent', 'Откр. Brent', 'Макс. Brent', 'Мин. Brent',
    'Курс доллара', 'Ключевая ставка', 'Инфляция', 'Год', 'Месяц',
    'Цена биржа АИ-95'
]


def run_ai95():
    st.title("🔮 Прогноз цены на АИ-95")
    st.markdown("Введите данные для прогноза:")

    # --- Поля ввода ---
    col1, col2 = st.columns(2)

    with col1:
        price_brent = st.number_input("Цена Brent (USD/барр)", value=70.0, step=0.1)
        open_brent = st.number_input("Откр. Brent", value=69.0, step=0.1)
        high_brent = st.number_input("Макс. Brent", value=71.0, step=0.1)
        low_brent = st.number_input("Мин. Brent", value=68.0, step=0.1)
        usd_rate = st.number_input("Курс доллара", value=85.0, step=0.1)
        year = st.number_input("Год", value=2026, step=1)
        month = st.slider("Месяц", 1, 12, 6)

    with col2:
        key_rate = st.number_input("Ключевая ставка", value=0.155, step=0.001)
        inflation = st.number_input("Инфляция", value=0.085, step=0.001)
        price_aiton = st.number_input("Цена биржа АИ-95 (руб/тонн)", value=85000, step=100)


    if st.button("Рассчитать и сделать прогноз"):
        # --- Сбор данных ---
        input_data = pd.DataFrame([{
            'Цена Brent': price_brent,
            'Откр. Brent': open_brent,
            'Макс. Brent': high_brent,
            'Мин. Brent': low_brent,
            'Курс доллара': usd_rate,
            'Ключевая ставка': key_rate,
            'Инфляция': inflation,
            'Год': year,
            'Месяц': month,
            'Цена биржа АИ-95': price_aiton
        }])

        # --- Извлечение моделей из ZIP ---
        zip_path = 'models/ai95_models.zip'
        if not os.path.exists(zip_path):
            st.error(f"ZIP-архив с моделями не найден: {zip_path}")
            return

        with zipfile.ZipFile(zip_path, 'r') as z:
            model_files = [f for f in z.namelist() if f.endswith('.pkl') and 'model_' in f]
            # Сортировка по номеру модели: model_1.pkl → model_30.pkl
            model_files.sort(key=lambda x: int(x.split('_')[1].replace('.pkl', '')))

            predictions = {}
            for file in model_files:
                try:
                    with z.open(file) as f:
                        model = pickle.load(BytesIO(f.read()))
                        horizon = int(file.split('_')[1].replace('.pkl', ''))

                        data = input_data[features]

                        pred = model.predict(data)[0]
                        predictions[f'{horizon} days'] = round(float(pred), 2)
                except Exception as e:
                    st.warning(f"Ошибка при прогнозировании на {horizon} день: {e}")
                    predictions[f'{horizon} days'] = None

        # --- Сохранение результата ---
        forecast_df = pd.DataFrame([predictions])
        st.session_state['forecast_df'] = forecast_df
        st.session_state['input_data'] = input_data

        # --- Отображение результатов ---
        st.subheader("Прогноз на 30 дней вперёд")
        styled_df = forecast_df.style.format({col: "{:.2f}" for col in forecast_df.columns})
        st.dataframe(styled_df)

        # --- График через Plotly ---
        st.subheader("График прогноза")

        # Преобразуем в длинный формат
        forecast_long = forecast_df.T.reset_index()
        forecast_long.columns = ['Дни', 'Цена']
        forecast_long['День'] = forecast_long['Дни'].str.extract(r'(\d+)').astype(int)
        forecast_long = forecast_long.sort_values('День').reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_long['День'],
            y=forecast_long['Цена'],
            mode='lines+markers',
            name='АИ-95',
            line=dict(color='purple'),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Прогноз цены на АИ-95",
            xaxis_title="Горизонт прогноза (дни)",
            yaxis_title="Цена (руб/тонн)",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(tickmode='linear', dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Скачивание в Excel ---
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            forecast_df.to_excel(writer, sheet_name='Прогноз', index=False)
            input_data.to_excel(writer, sheet_name='Входные данные', index=False)
        output.seek(0)

        st.download_button(
            label="📥 Скачать результаты в Excel",
            data=output,
            file_name="прогноз_АИ-95.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Показать входные данные ---
        st.subheader("Введённые данные")

        st.write(input_data)
