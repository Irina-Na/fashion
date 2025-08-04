"""Batch‑by‑meta LLM extraction with prompt caching.

• CSV input must contain: name, category_id, meta_category.
• TEMPLATES comes from previous section (one entry per meta).
• We process dataframe meta‑by‑meta to keep each prompt‑cache "hot" (TTL ≈ 5–10 min).
• Cache IDs are persisted in cache_ids.json so repeated runs skip re‑warming.
"""

from __future__ import annotations

import json, os, time
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Dict, Any
import math
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------- 0. settings ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY env var"

from prompts import GENERAL_PROMPT, TEMPLATES  # <- your canvas file with schemas & placeholders filled

CACHE_FILE = Path("cache_ids.json")
CACHE_FILE.touch(exist_ok=True)
cache_ids: Dict[str, str] = json.loads(CACHE_FILE.read_text() or "{}")


# ---------- 1. helpers ---------
DATA_DIR = Path(__file__).parent.parent / "data"

client = OpenAI(api_key=OPENAI_API_KEY)


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
        temperature=0.1,
        max_completion_tokens=20000,
    )
    return resp.choices[0].message.parsed.model_dump()  # OpenAI SDK v1.14+ returns Pydantic instance

# ---------- 2. main ----------

def enrich_csv(csv_in: str, model:str, csv_out: str) -> None:
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
        if meta == 'bag':
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


if __name__ == "__main__":
    in_path = DATA_DIR / "items_with_meta_small.csv"
    out_path = DATA_DIR / "extracted_features_bags.csv"

    enrich_csv(in_path, 'gpt-4.1', out_path )