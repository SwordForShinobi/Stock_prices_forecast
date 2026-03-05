import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import zipfile
from io import BytesIO
import plotly.graph_objects as go

# --- Данные ---
indicativ = {
    'year': [2021, 2022, 2023, 2024, 2025, 2026],
    'AI-92': [56300, 55200, 58650, 58650, 60450, 62300],
    'DT': [50700, 52250, 53850, 55450, 57150, 58900]
}
df_indicativ = pd.DataFrame(indicativ)

transit = {
    'year': [2021, 2022, 2023, 2024, 2025, 2026],
    'per_tonn_usd': [23, 90, 50, 40, 40, 35]
}
df_transit = pd.DataFrame(transit)


def run_ai98():
    st.title("🔮 Прогноз цены на АИ-98")
    st.markdown("Введите данные для прогноза:")

    # --- Поля ввода ---
    col1, col2 = st.columns(2)

    with col1:
        moex10 = st.number_input("MOEX10", min_value=4000, max_value=15000, value=5500)
        moexog = st.number_input("MOEXOG", min_value=4000, max_value=15000, value=7500)
        usd_rate = st.number_input("Курс доллара", min_value=60.0, max_value=120.0, value=85.0, step=0.1)
        year =  st.number_input("Год", min_value=1990, max_value=2050, value=2026, step=1)
        month = st.slider("Месяц", 1, 12, 6)

    with col2:
        price_aiton = st.number_input("Цена биржа АИ-98 (руб/тонн)", min_value=10000, max_value=120000, value=90000, step=100)
        price_brent = st.number_input("Цена Brent (USD/барр)", min_value=40.0, max_value=100.0, value=70.0, step=0.1)
        price_urals = st.number_input("Цена Urals (USD/барр)", min_value=30.0, max_value=90.0, value=50.0, step=0.1)
        key_rate = st.number_input("Ключевая ставка", min_value=0.05, max_value=0.25, value=0.155, step=0.001)
        inflation = st.number_input("Инфляция", min_value=0.01, max_value=0.20, value=0.085, step=0.001)

    if st.button("Рассчитать и сделать прогноз"):
        # --- Формируем DataFrame ---
        input_data = pd.DataFrame([{
            'MOEX10': moex10,
            'MOEXOG': moexog,
            'Курс доллара': usd_rate,
            'Ключевая ставка': key_rate,
            'Инфляция': inflation,
            'Год': year,
            'Месяц': month,
            'Цена биржа АИ-98': price_aiton,
            'Цена Brent': price_brent,
            'Цена Urals': price_urals
        }])

        # --- Расчёт Indicativ ---
        indicativ_val = df_indicativ.loc[df_indicativ['year'] == year, 'AI-92'].values[0]
        input_data = input_data.merge(df_indicativ[['year', 'AI-92']], left_on='Год', right_on='year', how='left') \
                               .rename(columns={'AI-92': 'Indicatif'}) \
                               .drop(columns=['year'], errors='ignore')

        # --- Расчёт Transit_price ---
        transit_val = df_transit.loc[df_transit['year'] == year, 'per_tonn_usd'].values[0]
        input_data = input_data.merge(df_transit[['year', 'per_tonn_usd']], left_on='Год', right_on='year', how='left') \
                               .rename(columns={'per_tonn_usd': 'Transit_price'}) \
                               .drop(columns=['year'], errors='ignore')

        # --- Price_exp ---
        input_data['Price_exp'] = 1.2 * (
            (input_data['Цена Brent'] * 7.33 * 1.15 * input_data['Курс доллара']) -
            ((input_data['Цена Brent'] - input_data['Цена Urals']) * 7.33 * input_data['Курс доллара']) -
            (input_data['Transit_price'] * input_data['Курс доллара'])
        )
        input_data['Price_exp'] = round(input_data['Price_exp'], 2)

        # --- Dempfer ---
        input_data['Dempfer'] = (input_data['Price_exp'] - input_data['Indicatif']) * 0.68
        input_data['Dempfer'] = input_data['Dempfer'].apply(lambda x: x if x > 0 else 0)

        # --- Финальные фичи ---
        final_features = input_data[[
            'MOEX10', 'MOEXOG', 'Курс доллара', 'Ключевая ставка',
            'Инфляция', 'Год', 'Месяц', 'Indicatif', 'Transit_price', 'Price_exp', 'Цена биржа АИ-98'
        ]].rename(columns={'Indicatif': 'Indicativ'})

        # --- Извлечение моделей из ZIP ---
        zip_path = 'models/ai98_models.zip'
        if not os.path.exists(zip_path):
            st.error(f"ZIP-архив с моделями не найден: {zip_path}")
            st.stop()

        with zipfile.ZipFile(zip_path, 'r') as z:
            model_files = [f for f in z.namelist() if f.endswith('.pkl') and 'model_' in f]
            # Сортировка по номеру модели: model_1.pkl, model_2.pkl, ..., model_30.pkl
            model_files.sort(key=lambda x: int(x.split('_')[1].replace('.pkl', '')))

            predictions = {}
            for file in model_files:
                try:
                    with z.open(file) as f:
                        model_data = BytesIO(f.read())
                        model = pickle.load(model_data)
                        horizon = int(file.split('_')[1].replace('.pkl', ''))
                        pred = model.predict(final_features)[0]
                        predictions[f'{horizon} days'] = round(pred, 2)
                except Exception as e:
                    st.warning(f"Ошибка при загрузке модели {file}: {e}")
                    predictions[f'{horizon} days'] = np.nan

        # --- Сохранение результата ---
        forecast_df = pd.DataFrame([predictions])
        st.session_state['forecast_df'] = forecast_df
        st.session_state['input_data'] = input_data

        # --- Отображение результатов ---
        st.subheader("Прогноз на 30 дней вперёд")
        st.dataframe(forecast_df.style.format("{:.2f}"))

        # --- График через Plotly ---
        st.subheader("График прогноза")

        # Преобразуем прогноз в длинный формат
        forecast_long = forecast_df.T.reset_index()
        forecast_long.columns = ['Дни', 'Цена']
        forecast_long['День'] = forecast_long['Дни'].str.extract(r'(\d+)').astype(int)

        # Сортируем по дню, чтобы график шёл от 1 до 30
        forecast_long = forecast_long.sort_values('День').reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_long['День'],
            y=forecast_long['Цена'],
            mode='lines+markers',
            name='Прогноз АИ-98',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Прогноз цены на АИ-98",
            xaxis_title="Горизонт прогноза (дни)",
            yaxis_title="Цена (руб/тонн)",
            hovermode="x unified",
            template="plotly_white",
            xaxis=dict(
                tickmode='linear',
                dtick=1
            )
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
            file_name="прогноз_АИ-98.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Показать входные данные ---
        st.subheader("Введённые данные и расчётные параметры")

        st.write(final_features)
