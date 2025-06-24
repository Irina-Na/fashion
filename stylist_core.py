# stylist_core.py
from __future__ import annotations
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import pandas as pd

import openai
import prompts
from prompts import OneTotalLook, Item




# ---------- LLM call ----------
def generate_look(user_text: str, model: str = "gpt-4o-mini") -> OneTotalLook:
    """
    Запрашивает LLM и возвращает структурированный OneTotalLook.
    Ключ API можно передать напрямую или через переменную окружения OPENAI_API_KEY.
    """
    
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": prompts.TOTAL_CREATIONLOOK_PROMPT.format(request=user_text)},
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        max_completion_tokens=1000,
        response_format=OneTotalLook,
    )

    # .parse() возвращает специальный объект, сама модель в .choices[0].message.parsed
    return response.choices[0].message.parsed


# ---------- DF utilities ----------
def match_item(df: pd.DataFrame, itm: Item) -> pd.DataFrame:
    """
    Оставляет строки c совпадением по category_id[0] и (необязательно) другим признакам.
    Раскомментируйте фильтры, как только заполните соответствующие столбцы датасета.
    """
    df_f = df[df["category_id"].str[0] == itm.category]
    df_2 = df[df["name"].str.contains(itm.category)]
    df_f = pd.concat([df_f, df_2])

    # if itm.color:
    #     df_f = df_f[df_f["color_hex"].isin(itm.color)]
    # if itm.pattern:
    #     df_f = df_f[df_f["pattern_id"].isin(itm.pattern)]
    # if itm.fabric:
    #     df_f = df_f[df_f["fabric"].isin(itm.fabric)]
    # if itm.fit:
    #     df_f = df_f[df_f["fit"] == itm.fit]

    return df_f


def filter_dataset(df: pd.DataFrame, look: OneTotalLook,
                   max_per_item: int = 1) -> Dict[str, pd.DataFrame]:
    """
    Возвращает словарь {part: dataframe} с подходящими позициями.
    """

    if look.sex:
        df_base = df[df["gender"].str.lower() == "unisex"]
        df_sex = df[df["gender"].str.lower() == look.sex.lower()]
        df_base = pd.concat([df_base, df_sex])

    parts = ['top', 'bottom', 'full', 'shoes', 'outerwear', 'accessories']
    results: Dict[str, pd.DataFrame] = {}

    for part in parts:
        items = getattr(look, part) or []
        selections = []
        for itm in items:
            sub = match_item(df_base, itm)
            if not sub.empty:
                selections.append(sub.head(max_per_item))
        results[part] = pd.concat(selections) if selections else pd.DataFrame()

    return results
