import streamlit as st
import pandas as pd
import pickle
import zipfile
import os
from io import BytesIO
import plotly.graph_objects as go

# --- Справочники ---
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
features_1_20 = [
    'Ключевая ставка', 'Год', 'Месяц', 'Indicativ', 'ДТ реализация'
]

features_21_30 = [
    'ДТ реализация', 'Цена Brent', 'Курс доллара',
    'Ключевая ставка', 'Инфляция', 'MOEX10', 'MOEXOG',
    'Год', 'Месяц', 'Indicativ', 'Transit_price', 'Price_exp',
    'Dempfer', 'Brent/WTI_spred'
]

def run_dt():
    st.title("🔮 Прогноз цены на ДТ реализация")
    st.markdown("Введите данные для прогноза:")

    # --- Поля ввода ---
    col1, col2 = st.columns(2)

    with col1:
        price_brent = st.number_input("Цена Brent (USD/барр)", value=70.0, step=0.1)
        usd_rate = st.number_input("Курс доллара", value=78.0, step=0.1)
        moex10 = st.number_input("MOEX10", value=5100)
        moexog = st.number_input("MOEXOG", value=7500)
        year = st.number_input("Год", value=2026)
        month = st.slider("Месяц", 1, 12, 6)

    with col2:
        key_rate = st.number_input("Ключевая ставка", value=0.155, step=0.001)
        inflation = st.number_input("Инфляция", value=0.085, step=0.001)
        dt_price = st.number_input("ДТ реализация (руб/тонн)", value=71000, step=100)
        price_urals = st.number_input("Цена Urals (USD/барр)", value=50.0, step=0.1)
        price_wti = st.number_input("Цена WTI (USD/барр)", value=65.0, step=0.1)

    if st.button("Рассчитать и сделать прогноз"):
        # --- Сбор данных ---
        input_data = pd.DataFrame([{
            'Ключевая ставка': key_rate,
            'Год': year,
            'Месяц': month,
            'ДТ реализация': dt_price,
            'Цена Brent': price_brent,
            'Курс доллара': usd_rate,
            'Инфляция': inflation,
            'MOEX10': moex10,
            'MOEXOG': moexog,
            'Цена Urals': price_urals,
            'Цена WTI': price_wti
        }])
        # --- Объединение данных: получение индикатива и цен на транзит ---
        input_data = pd.merge(input_data, df_indicativ[['DT', 'year']],
                              left_on='Год', right_on='year').rename(columns={'DT': 'Indicativ'})
        input_data = pd.merge(input_data, df_transit, left_on='Год', right_on='year').rename(columns={
            'per_tonn_usd': 'Transit_price'}).drop(columns=['year_x', 'year_y'])

        # --- Price_exp --- прикольный коэффициент в начале формулы, выравнивает погрешность
        # при переходе с одной модели на другую
        input_data['Price_exp'] = 1 * (
                (input_data['Цена Brent'] * 7.45 * 1.5 * input_data['Курс доллара']) -
                ((input_data['Цена Brent'] - input_data['Цена Urals']) * 7.45 * input_data['Курс доллара']) -
                (input_data['Transit_price'] * input_data['Курс доллара'])
        )
        input_data['Price_exp'] = round(input_data['Price_exp'], 2)

        # --- Dempfer ---
        input_data['Dempfer'] = (input_data['Price_exp'] - input_data['Indicativ']) * 0.65
        input_data['Dempfer'] = [i if i>0 else 0 for i in input_data['Dempfer']]

        # --- Brent/WTI_spred ---
        input_data['Brent/WTI_spred'] = input_data['Цена Brent'] - input_data['Цена WTI']

        # --- Извлечение моделей из ZIP ---
        zip_path = 'models/dt_models.zip'
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

                        if horizon >= 21:
                            data = input_data[features_21_30]
                        else:
                            data = input_data[features_1_20]

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
            name='ДТ реализация',
            line=dict(color='orange'),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Прогноз цены на ДТ реализация",
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
            file_name="прогноз_ДТ_реализация.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # --- Показать входные данные ---
        st.subheader("Введённые данные")

        st.write(input_data)

