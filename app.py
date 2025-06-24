# app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import ast
# --- ваш бизнес-код ---
from stylist_core import generate_look, filter_dataset

# ──────────────────────────────────────────────────────────────
# Константы (можно переопределить через переменные окружения)
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATA_PATH = Path(
    os.getenv("DATA_PATH", DATA_DIR / "clothes_enriched_new_cat1_only.csv")
).expanduser()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SUPPORTED_EXT = {".parquet", ".csv"}
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fashion Look Finder", layout="wide")
st.title("👗 Total-Look Stylist")

def to_list(val):
    """
    Преобразует строку-представление списка в настоящий список.
    Оставляет без изменений NaN и уже готовые списки.
    """
    if pd.isna(val) or isinstance(val, list):
        return val
    return ast.literal_eval(val)   # безопасный eval для литералов

df_enriched = pd.read_csv(
    DEFAULT_DATA_PATH,
    converters={'category_id': to_list}
)
df_enriched = df_enriched.fillna("")
# --- ввод запроса пользователя ---
user_query = st.text_area(
    "Опишите образ (любой свободный текст)",
    "Мне нужен образ на выпускной в нежных пастельных тонах, лето, женский.",
    height=120,
)


model_choice = st.sidebar.selectbox(
    "LLM-модель", [ "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"], index=0
)

# --- обработка запроса ---
if st.button("Сгенерировать лук"):
    with st.spinner("Запрашиваем стилиста-ИИ…"):
        look = generate_look(user_query, model=model_choice)

    st.success("Образ сгенерирован")
    st.write("### Структура полученного лука")
    st.json(look.model_dump(), expanded=False)

    # --- фильтрация датасета ---
    with st.spinner("Подбираем вещи из каталога…"):
        results = filter_dataset(df_enriched, look, max_per_item=100)

    # --- вывод таблиц ---
    for part, df_part in results.items():
        if df_part.empty:
            st.write(f"_{part}: подходящих вещей не найдено_")
        else:
            st.subheader(part.capitalize())
            st.dataframe(df_part, use_container_width=True)
