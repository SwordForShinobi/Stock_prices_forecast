import streamlit as st
import pandas as pd
import pickle
import zipfile
import os
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

# --- Фичи ---
features_1_11 = [
    'Цена Brent', 'Откр. Brent', 'Макс. Brent', 'Мин. Brent',
    'Курс доллара', 'Ключевая ставка', 'Инфляция', 'Год', 'Месяц', 'Цена биржа АИ-92'
]

features_12_30 = [
    'Курс доллара', 'Ключевая ставка', 'MOEX10', 'MOEXOG',
    'Год', 'Месяц', 'Indicativ', 'Price_exp', 'Dempfer', 'Цена биржа АИ-92'
]


def run_ai92():
    st.title("🔮 Прогноз цены на АИ-92")
    st.markdown("Введите данные для прогноза:")

    # --- Поля ввода ---
    col1, col2 = st.columns(2)

    with col1:
        price_brent = st.number_input("Цена Brent (USD/барр)", min_value=40.0, max_value=100.0, value=70.0, step=0.1)
        open_brent = st.number_input("Откр. Brent", min_value=40.0, max_value=100.0, value=69.0, step=0.1)
        high_brent = st.number_input("Макс. Brent", min_value=40.0, max_value=100.0, value=71.0, step=0.1)
        low_brent = st.number_input("Мин. Brent", min_value=40.0, max_value=100.0, value=68.0, step=0.1)
        usd_rate = st.number_input("Курс доллара", min_value=60.0, max_value=120.0, value=85.0, step=0.1)
        year = st.number_input("Год", min_value=1990, max_value=2050, value=2026, step=1)
        month = st.slider("Месяц", 1, 12, 6)

    with col2:
        price_aiton = st.number_input("Цена биржа АИ-92 (руб/тонн)", min_value=60000, max_value=100000, value=65000, step=100)
        moex10 = st.number_input("MOEX10", min_value=4000, max_value=15000, value=5500)
        moexog = st.number_input("MOEXOG", min_value=4000, max_value=15000, value=7500)
        price_urals = st.number_input("Цена Urals (USD/барр)", min_value=30.0, max_value=90.0, value=50.0, step=0.1)
        key_rate = st.number_input("Ключевая ставка", min_value=0.05, max_value=0.25, value=0.155, step=0.001)
        inflation = st.number_input("Инфляция", min_value=0.01, max_value=0.20, value=0.085, step=0.001)

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
            'Цена биржа АИ-92': price_aiton,
            'MOEX10': moex10,
            'MOEXOG': moexog,
            'Цена Urals': price_urals
        }])

        # --- Расчёт Indicativ ---
        try:
            indicativ_val = df_indicativ.loc[df_indicativ['year'] == year, 'AI-92'].values[0]
        except IndexError:
            st.error(f"Не найдено значение Indicativ для года {year}")
            return

        # --- Расчёт Transit_price ---
        try:
            transit_usd = df_transit.loc[df_transit['year'] == year, 'per_tonn_usd'].values[0]
        except IndexError:
            st.error(f"Не найдено значение Transit для года {year}")
            return
        transit_rub = transit_usd * usd_rate

        # --- Price_exp ---
        price_exp = (price_brent * 7.33 * 1.15 * usd_rate) - \
                    ((price_brent - price_urals) * 7.33 * usd_rate) - \
                    transit_rub
        price_exp = round(price_exp, 2)

        # --- Dempfer ---
        dumper = max((price_exp - indicativ_val) * 0.68, 0)

        # --- Загрузка моделей из ZIP ---
        zip_path = 'models/ai92_models.zip'
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

                        if horizon <= 11:
                            data = input_data[features_1_11]
                        else:
                            data = input_data.assign(
                                Indicativ=indicativ_val,
                                Price_exp=price_exp,
                                Dempfer=dumper
                            )[features_12_30]

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
            name='АИ-92',
            line=dict(color='green'),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Прогноз цены на АИ-92",
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
            file_name="прогноз_АИ-92.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Показать входные данные ---
        st.subheader("Введённые данные")

        st.write(input_data)
