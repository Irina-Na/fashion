# stylist_core.py
from __future__ import annotations
import os
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
import openai

# ---------- Pydantic models ----------
class Item(BaseModel):
    category: str = Field(..., description="название категории одежды")
    fabric:  Optional[List[str]] = None
    color:   Optional[List[str]] = None
    pattern: Optional[List[str]] = None
    fit:     Optional[str]       = None


class OneTotalLook(BaseModel):
    sex:        str  = Field(..., description="женский, мужской, унисекс")
    season:     Optional[str]      = Field(None, description="зимний, летний и т.п.")
    top:        Optional[List[Item]] = None
    bottom:     Optional[List[Item]] = None
    full:       Optional[List[Item]] = None
    shoes:      List[Item]
    outerwear:  Optional[List[Item]] = None
    accessories: Optional[List[Item]] = None


_SYSTEM_PROMPT = """\
Вы - профессиональный стилист, создающий total look.
Вот запрос пользователя: {request}.
Проанализируй запрос и собери лук, который бы удовлетворял всем его требованиям.
Выбери пол, сезон, укажи обувь. При необходимости добавь верхнюю одежду/аксессуары.
"""


# ---------- LLM call ----------
def generate_look(user_text: str, model: str = "gpt-4o-mini") -> OneTotalLook:
    """
    Запрашивает LLM и возвращает структурированный OneTotalLook.
    Требуется переменная окружения OPENAI_API_KEY или ключ в streamlit secrets.
    """
    client = openai.OpenAI()                 # openai.api_key берётся из env/secrets
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(request=user_text)},
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1000,
        response_format=OneTotalLook,
    )

    # parse() уже возвращает объект модели
    return response


# ---------- DF utilities ----------
def match_item(df: pd.DataFrame, itm: Item) -> pd.DataFrame:
    """
    Оставляет строки c совпадением по category_id[0] и (необязательно) другим признакам.
    Раскомментируйте фильтры, как только заполните соответствующие столбцы датасета.
    """
    df_f = df[df["category_id"].str[0] == itm.category]

    # if itm.color:
    #     df_f = df_f[df_f["color_hex"].isin(itm.color)]
    # if itm.pattern:
    #     df_f = df_f[df_f["pattern_id"].isin(itm.pattern)]
    # if itm.fabric:
    #     df_f = df_f[df_f["fabric"].isin(itm.fabric)]
    # if itm.fit:
    #     df_f = df_f[df_f["fit"] == itm.fit]

    return df_f


def filter_dataset(df_enriched: pd.DataFrame, look: OneTotalLook,
                   max_per_item: int = 1) -> Dict[str, pd.DataFrame]:
    """
    Возвращает словарь {part: dataframe} с подходящими позициями.
    """
    df_base = df_enriched.copy()
    df_base = df_base[df_base.category_pieces == 1]

    if look.sex:
        df_base = df_base[df_base["gender"].str.lower() == look.sex.lower()]

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
