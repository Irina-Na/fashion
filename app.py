# app.py
import os
import streamlit as st
import pandas as pd
from stylist_core import generate_look, filter_dataset

st.set_page_config(page_title="Fashion Look Finder", layout="wide")
st.title("👗 Total-Look Stylist")

# --- загрузка датасета ---
@st.cache_data(show_spinner="Загружаем датасет…")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)  # или CSV/SQL — замените на свой формат

DATA_PATH = st.sidebar.text_input(
    "Путь к датасету (parquet/CSV)", value="df_enriched.parquet"
)
if DATA_PATH and os.path.isfile(DATA_PATH):
    df_enriched = load_data(DATA_PATH)
    st.sidebar.success("Датасет загружен")
else:
    st.sidebar.error("Файл не найден")
    st.stop()

# --- ввод запроса пользователя ---
user_query = st.text_area(
    "Опишите образ (любой свободный текст)",
    "Мне нужен образ на выпускной в нежных пастельных тонах, лето, женский.",
    height=120,
)

api_key = st.sidebar.text_input(
    "OpenAI API key (если не в ENV/Secrets)",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

model_choice = st.sidebar.selectbox(
    "LLM модель", ["gpt-4o-mini", "gpt-4o", "gpt-4"], index=0
)

if st.button("Сгенерировать лук"):
    with st.spinner("Запрашиваем стилиста-ИИ…"):
        look = generate_look(user_query, model=model_choice)
    st.success("Образ сгенерирован")
    st.write("### Структура полученного лука")
    st.json(look.model_dump(), expanded=False)

    # Фильтрация датасета
    with st.spinner("Подбираем вещи из каталога…"):
        results = filter_dataset(df_enriched, look)

    # Вывод таблиц
    for part, df_part in results.items():
        if not df_part.empty:
            st.subheader(part.capitalize())
            st.dataframe(df_part, use_container_width=True)
        else:
            st.write(f"_{part}: подходящих вещей не найдено_")
