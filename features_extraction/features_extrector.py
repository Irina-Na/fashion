"""Batch‑by‑meta LLM extraction with prompt caching.

• CSV input must contain: name, category_id, meta_category.
• TEMPLATES comes from previous section (one entry per meta).
• We process dataframe meta‑by‑meta to keep each prompt‑cache "hot" (TTL ≈ 5–10 min).
• Cache IDs are persisted in cache_ids.json so repeated runs skip re‑warming.

This module exposes two enrichment functions:
* :func:`enrich_csv` – uses textual descriptions (``name`` column).
* :func:`enrich_csv_from_images` – uses remote images referenced by
  ``image_external_url``.
"""

from __future__ import annotations

import json, os, time, requests
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Dict, Any
import math
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError, APIError
import httpx, time, uuid


# ---------- 0. settings ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY env var"

from prompts import GENERAL_PROMPT, TEMPLATES  # schemas & placeholders filled

CACHE_FILE = Path("cache_ids.json")
CACHE_FILE.touch(exist_ok=True)
cache_ids: Dict[str, str] = json.loads(CACHE_FILE.read_text() or "{}")


# ---------- 1. helpers ---------
DATA_DIR = Path(__file__).parent.parent / "data"

client = OpenAI( api_key=OPENAI_API_KEY,
     # можно и тут, но ниже покажу per-call override
    # max_retries=2 по умолчанию
    timeout=90.0,  # разумный верх для vision-задач
)

def infer_item(name: str, meta: str,  model: str) -> dict[str, Any]: #cache_id: str,
    tpl = TEMPLATES[meta]
    resp = client.beta.chat.completions.parse(
        model=model, #tpl["model"],
        #cache_control={"prefix_cache_ids": [cache_id]},
        #prompt_cache_key=f"{hash(GENERAL_PROMPT)}",
        messages=[
            {"role": "system", "content": GENERAL_PROMPT},
            {"role": "user", "content": name},
        ],
        response_format=tpl["class"],
        temperature=0.0,
        max_completion_tokens=20000,
    )
    return resp.choices[0].message.parsed.model_dump()
"""
def infer_item_from_image(image_url: str, description: str, meta: str, model: str) -> dict[str, Any]:
    tpl = TEMPLATES[meta]
    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": GENERAL_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ],
            },
        ],
        response_format=tpl["class"],
        #temperature=0,
        max_completion_tokens=15000,
    )
    return resp.choices[0].message.parsed.model_dump()
"""
def _with_retry(call, tries=6, base_delay=0.5):
    for attempt in range(1, tries + 1):
        try:
            return call()
        except (APIConnectionError, httpx.RemoteProtocolError, httpx.ConnectError):
            if attempt == tries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))

def infer_item_from_image(img_url: str, description:str, meta: str, model: str) -> dict[str, Any]:
    tpl = TEMPLATES[meta]
    content = [
        {"type": "text", "text": description},
        {"type": "image_url", "image_url": {"url": img_url}},  # Chat Completions синтаксис
    ]

    do = lambda: client.with_options(
        max_retries=5,            # поддерживается
        timeout=120.0,            # поддерживается
    ).beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": GENERAL_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format=tpl["class"],  # Pydantic-модель
        # temperature НЕ передавать для reasoning-моделей вроде gpt-5
        max_completion_tokens=15000,

        # <-- если нужно добавить нестандартный заголовок (например, для прокси),
        # делай это здесь, а не в with_options:
        extra_headers={"X-Request-ID": str(uuid.uuid4())},
        # есть также extra_query / extra_body
    )

    resp = _with_retry(do)
    return resp.choices[0].message.parsed.model_dump()


def is_image_accessible(url: str, timeout: float = 5.0) -> bool:
    """Return True if the image URL responds with HTTP 200."""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            return True
        # Some servers may not support HEAD; fall back to GET
        resp = requests.get(url, timeout=timeout, stream=True)
        status = resp.status_code
        resp.close()
        return status == 200
    except Exception:
        return False

# ---------- 2. main text‑based enrichment ----------
def enrich_csv(csv_in: str, model: str, csv_out: str) -> None:
    """Enrich dataset using item names (text)."""
    df = pd.read_csv(csv_in)
    df = df.fillna("")
    df = df.drop_duplicates(['image_external_url']).drop_duplicates(['good_id', 'store_id'])
    assert {
        "name",
        "meta_category",
    }.issubset(df.columns), "CSV must have 'name' and 'meta_category' columns"

    records: list[dict[str, Any]] = []
    row_index = []

    #cache_id = client.caches.create(model=model, content=GENERAL_PROMPT).id
    
    # Persist immediately to survive crashes
    #CACHE_FILE.write_text(json.dumps(cache_id))
    #time.sleep(0.1)  # gentle rate‑limit
    
    # groupby meta for long‑lived prompt cache hit‑rate
    for meta, group in df.groupby("meta_category", sort=False):
        if meta not in TEMPLATES:
            print(f"[warn] Unknown meta '{meta}', skipping {len(group)} rows")
            continue
        if meta != 'bag':
            pass
        else:
                   
            print(f"➡ Processing {len(group)} items of meta '{meta}' …")
            for row in tqdm(group.itertuples(index=False), total=len(group), leave=False):
                if isinstance(row.name, float) and math.isnan(row.name):
                    pass
                  # or `"<unknown>"` if you prefer
                else:
                    name = str(row.name)         # guarantees plain text for the API

                    time.sleep(0.1)  # gentle rate‑limit
                    item = infer_item(name, meta, model) #cache_id)
                    item["good_id"] = row.good_id
                    records.append(item)

# concat and save
    df_rec = pd.DataFrame(records) # now join / concat will align exactly to the indexes
    df_rec.to_csv(DATA_DIR / f"extracted_products_exclude_bags.csv")  
    # df_enriched = pd.concat([df.reset_index(drop=True), df_rec], axis=1)
    #df_enriched = df.join(df_rec)         # same as concat([df, df_rec], axis=1)
    df_enriched = df.merge(df_rec, on="good_id", how="left")
    df_enriched.to_csv(csv_out, index=False)
    #print(f"✅ Saved → {csv_out}  (rows: {len(df_enriched)})")
    #df_rec.to_csv(csv_out)
    print(f"✅ Saved → {csv_out}  (rows: {len(df_rec)})")

# ---------- 3. image‑based enrichment ----------
def enrich_csv_from_images(csv_in: str, model: str, csv_out: str) -> None:
    """Enrich dataset using images referenced by 'image_external_url'."""
    df = pd.read_csv(csv_in)
    df = df.fillna("")
    df = df.drop_duplicates(["image_external_url"]).drop_duplicates(["good_id", "store_id"])
    assert {"image_external_url", "meta_category"}.issubset(
        df.columns
    ), "CSV must have 'image_external_url' and 'meta_category' columns"

    records: list[dict[str, Any]] = []

    for meta, group in df.groupby("meta_category", sort=False):
        if meta not in TEMPLATES:
            print(f"[warn] Unknown meta '{meta}', skipping {len(group)} rows")
            continue
        if meta != 'bag':
            pass
        else:
            print(f"➡ Processing {len(group)} items of meta '{meta}' …")
            for row in tqdm(group.itertuples(index=False), total=len(group), leave=False):
                if not row.image_external_url:
                    continue
                if not is_image_accessible(row.image_external_url):
                    print(f"[warn] Unreachable image {row.image_external_url}, skipping")
                    continue
                time.sleep(0.1)
                item = infer_item_from_image(row.image_external_url, str(row.name), meta, model)
                item["good_id"] = row.good_id
                records.append(item)

    df_rec = pd.DataFrame(records)
    df_rec.to_csv(DATA_DIR / "extracted_products_from_images_bags.csv", index=False)
    df_enriched = df.merge(df_rec, on="good_id", how="left")
    df_enriched.to_csv(csv_out, index=False)
    print(f"✅ Saved → {csv_out}  (rows: {len(df_rec)})")


if __name__ == "__main__":
    in_path = DATA_DIR / "items_with_meta_small.csv"
    out_path_text = DATA_DIR / "extracted_features_bags.csv"
    out_path_img = DATA_DIR / "extracted_features_from_images_bags.csv"

    # Example usage (uncomment the desired call):
    # enrich_csv(in_path, "gpt-4.1", out_path_text)
    enrich_csv_from_images(in_path, "gpt-5", out_path_img)