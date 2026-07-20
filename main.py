import streamlit as st

# Импортируем функции запуска для каждого продукта
from ai92 import run_ai92
from ai95 import run_ai95
from ai98 import run_ai98
from dt import run_dt

# Боковое меню для выбора продукта
st.sidebar.title("⛽ Выбор нефтепродукта")
product = st.sidebar.radio(
    "Выберите продукт:",
    ("АИ-92", "АИ-95", "АИ-98", "ДТ реализация")
)

# Запуск соответствующего интерфейса
if product == "АИ-92":
    run_ai92()
elif product == "АИ-95":
    run_ai95()
elif product == "АИ-98":
    run_ai98()
elif product == "ДТ реализация":
    run_dt()
# else:
#     if product == "ДТ тестирование":
#         run_dt_test()
else:
    st.error("Выберите нефтепродукт для запуска интерфейса.")