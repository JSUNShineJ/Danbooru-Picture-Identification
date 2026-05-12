"""根据自然语言查询,从 FAISS 索引里搜出最相似的图片。"""

import pickle
from pathlib import Path

import numpy as np
import faiss
from openai import OpenAI

from config import (
    OPENAI_API_KEY, EMBED_MODEL,
    INDEX_PATH, META_PKL,
)


client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# 加载索引和元数据(模块级,导入时只跑一次)
# ─────────────────────────────────────────────

print("📦 加载 FAISS 索引...")
_index = faiss.read_index(str(INDEX_PATH))

with open(META_PKL, "rb") as f:
    _records = pickle.load(f)

print(f"   ✅ {_index.ntotal} 条向量,{len(_records)} 条元数据")


# ─────────────────────────────────────────────
# 搜索函数
# ─────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """把自然语言查询转成向量(归一化后)。"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    vec = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


from config import CHAT_MODEL   # config 里加 CHAT_MODEL = "gpt-4o-mini"


REWRITE_PROMPT = """You are a danbooru tag expert. Convert the user's natural language query into danbooru tags.

Rules:
- Output ONLY the tags, separated by spaces, no other text
- Use underscores for multi-word tags (e.g., "from_behind" not "from behind")
- Use lowercase
- Common tag patterns:
  - person count: 1girl, 1boy, 2girls, ...
  - viewpoint: from_behind, from_above, from_side, looking_back
  - hair: pink_hair, long_hair, short_hair, twintails
  - fantasy: dragon_girl, monster_girl, kemonomimi
  - composition: solo, multiple_girls

Examples:
User: a girl with pink hair seen from behind
Output: 1girl pink_hair from_behind

User: 龙娘从背后看
Output: 1girl dragon_girl from_behind

User: two girls smiling
Output: 2girls smile

Now convert this query:
"""


def rewrite_query(natural_query: str) -> str:
    """把自然语言转成 tag 串。"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user",   "content": natural_query},
        ],
        temperature=0,    # 关闭随机性
    )
    return resp.choices[0].message.content.strip()


def search(query: str, top_k: int = 5, use_rewrite: bool = True) -> list[dict]:
    """搜索。use_rewrite=True 时先用 GPT 改写查询。"""
    if use_rewrite:
        rewritten = rewrite_query(query)
        print(f"  🔄 改写: '{query}' → '{rewritten}'")
        embed_input = rewritten
    else:
        embed_input = query
    
    q_vec = embed_query(embed_input)
    
    scores, indices = _index.search(q_vec, top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        meta = _records[idx].copy()
        meta["_rank"]  = rank
        meta["_score"] = float(score)
        results.append(meta)
    return results
