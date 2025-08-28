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
from langfuse.openai import openai
from openai import OpenAI, APIConnectionError, BadRequestError, RateLimitError, APIError, BadRequestError
import httpx, time, uuid
import base64, tempfile



# ---------- 0. settings ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY env var"

from prompts import GENERAL_PROMPT, TEMPLATES, MetaCategory, META_CATEGORY_DETECTION_PROMPT  # schemas & placeholders filled

from langfuse import Langfuse, observe, get_client
 
@observe()
def process_request():
    # Get the client
    langfuse = get_client()
 
    # Add to the current trace
    langfuse.update_current_trace(session_id=SESSION_ID, tags=["feature_extraction", "tag-2"])
 
    # ...your processing logic...
    return 0 #result

langfuse = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST"),
)

CACHE_FILE = Path("cache_ids.json")
CACHE_FILE.touch(exist_ok=True)
cache_ids: Dict[str, str] = json.loads(CACHE_FILE.read_text() or "{}")

SESSION_ID = str(uuid.uuid4())

# ---------- 1. helpers ---------
DATA_DIR = Path(__file__).parent.parent / "data"

client = OpenAI( api_key=OPENAI_API_KEY,
     # можно и тут, но ниже покажу per-call override
    # max_retries=2 по умолчанию
    timeout=90.0,  # разумный верх для vision-задач
)

def infer_item(name: str, response_format: Any,  model: str, max_completion_tokens: int, prompt: str = GENERAL_PROMPT, ) -> dict[str, Any]: #cache_id: str,
    resp = client.beta.chat.completions.parse(
        model=model, #tpl["model"],
        #cache_control={"prefix_cache_ids": [cache_id]},
        #prompt_cache_key=f"{hash(GENERAL_PROMPT)}",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": name},
        ],
        response_format=response_format,
        temperature=0.0,
        max_completion_tokens=max_completion_tokens,
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

def to_base64(url: str) -> str:
    resp = requests.get(url, timeout=15)          # download on your side
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode()

def fetch_as_data_url(img_url: str, timeout=15) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",           # некоторые CDN режут «ботов»
        "Referer": img_url.rsplit("/", 1)[0],  # помогает против hotlink-защиты
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    }
    r = requests.get(img_url, timeout=timeout, headers=headers)
    r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg")
    b64 = base64.b64encode(r.content).decode("ascii")
    return f"data:{mime};base64,{b64}"


def infer_item_from_image_file(b64:str, meta: str, model: str, name: str) -> dict[str, Any]:
    tpl = TEMPLATES[meta]
    resp = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": GENERAL_PROMPT.replace('*NAME*', name)},
                    {"type": "input_image", "image_base64": b64},
                ],
            }
        ],
        text_format=tpl["class"],
        temperature=0,
    )
    return resp.output[0].content[0].parsed


@observe()    
def infer_item_from_image(img_url: str, description: str, meta: str, model: str) -> dict[str, Any]:
    tpl = TEMPLATES[meta]
    content = [
        {"type": "image_url", "image_url": {"url": img_url}},  # Chat Completions синтаксис
    ]
    prompt = GENERAL_PROMPT.replace('**META_CATEGORY_NAME**', tpl['metacategory_name']).replace('*NAME*', description).replace('**CATEGORY_EXAMPLES**', tpl['fewshots_categories']).replace('**MODEL_EXAMPLES**', tpl['fewshots_silhouette'])
    # Get the client
    langfuse = get_client()
 
    # Add to the current trace
    langfuse.update_current_trace(session_id=SESSION_ID,  tags=["feature_extraction", meta])
    try:
        do = lambda: client.with_options(
            max_retries=5,            # поддерживается
            timeout=120.0,            # поддерживается
        ).beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            response_format=tpl["class"],  # Pydantic-модель
            # temperature НЕ передавать для reasoning-моделей вроде gpt-5
            max_completion_tokens=15000,

            # <-- если нужно добавить нестандартный заголовок (например, для прокси),
            # делай это здесь, а не в with_options:
            extra_headers={"X-Request-ID": SESSION_ID},
            # есть также extra_query / extra_body
        )
        resp = _with_retry(do)
        return resp.choices[0].message.parsed.model_dump()
    except BadRequestError as e:
        #b64 = to_base64(img_url)
        #return infer_item_from_image_file(b64, meta, model, description)
    
    #return resp.choices[0].message.parsed.model_dump()

        # 2) Если ошибка вида invalid_image_url / timeout — фолбэк на data URL + Responses API
        msg = str(getattr(e, "message", e))
        if "invalid_image_url" in msg or "Timeout while downloading" in msg:
            data_url = fetch_as_data_url(img_url)
            resp2 = client.responses.parse(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text",  "text": GENERAL_PROMPT.replace("{NAME}", description)},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
                text_format=tpl["class"],
            )
            return resp2.output_parsed.model_dump()
        # 3) Иначе пробрасываем
        raise


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
                    item = infer_item(name=name, model=model, prompt=GENERAL_PROMPT, response_format= TEMPLATES[meta]["class"]) #cache_id)
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
        if meta != 'fullbody':
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
                item = infer_item_from_image(img_url=row.image_external_url, description=str(row.name), meta=meta, model=model)
                item["good_id"] = row.good_id
                records.append(item)

    df_rec = pd.DataFrame(records)
    df_rec.to_csv(DATA_DIR / "extracted_products_from_images_fullbody.csv", index=False)
    df_enriched = df.merge(df_rec, on="good_id", how="left")
    df_enriched.to_csv(csv_out, index=False)
    print(f"✅ Saved → {csv_out}  (rows: {len(df_rec)})")


def get_category(csv_in: str, csv_out: str, model: str = 'gpt-4.1-mini') -> None:
    """Classify items to metacategories by names (text)."""
    df = pd.read_csv(csv_in)
    df = df.fillna("")
    df = df.drop_duplicates(['image_external_url']).drop_duplicates(['good_id', 'store_id'])
    assert {
        "name",
    }.issubset(df.columns), "CSV must have 'name' columns"

    records: list[dict[str, Any]] = []

    print(f"➡ Classify {csv_in} items to meta-categories")
    for row in tqdm(df.itertuples(index=False), total=len(df), leave=False):
        if isinstance(row.name, float) and math.isnan(row.name):
            # Skip items with no name
            item = {"good_id": row.good_id, "category": None, "img_accessible": None}
        else:
            name = str(row.name)         # guarantees plain text for the API

            #time.sleep(0.1)  # gentle rate‑limit
            access = is_image_accessible(row.image_external_url)
            if access:
                item = infer_item(name=name, model=model, prompt=META_CATEGORY_DETECTION_PROMPT, response_format=MetaCategory, max_completion_tokens=100) #cache_id)
            else:
                item = dict()
            item["good_id"] = row.good_id
            item['img_accessible'] = access
            records.append(item)

    # concat and save
    df_rec = pd.DataFrame(records) # now join / concat will align exactly to the indexes
    df_rec.rename(columns={'category': 'meta_category_ai'}, inplace=True)

    df_rec.to_csv(DATA_DIR / f"extracted_products_categories.csv") 
    print(f"✅ Saved → {DATA_DIR / f"extracted_products_categories.csv"}  (rows: {len(df_rec)})")
 
    df_enriched = df.merge(df_rec, on="good_id", how="left")
    df_enriched.to_csv(csv_out, index=False)
    print(f"✅ Saved → {csv_out}  (rows: {len(df_enriched)})")


if __name__ == "__main__":
    in_path = DATA_DIR / 'items_with_ai_category_small_manual_check.csv' #"items_with_meta_small.csv"
    out_path_text = DATA_DIR / "extracted_features_full.csv"
    out_path_img = DATA_DIR / "extracted_features_from_images_full.csv"

    # Example usage (uncomment the desired call):
    # enrich_csv(in_path, "gpt-4.1", out_path_text)
    enrich_csv_from_images(in_path, "gpt-5-mini", out_path_img)
    #get_category(csv_in=DATA_DIR / "items_with_meta_small.csv", csv_out=DATA_DIR / "items_with_category.csv", model = 'gpt-4.1-mini') 
