"""把 metadata.jsonl 转成 FAISS 索引(带 embedding 缓存)。"""

import json
import pickle
import time

import numpy as np
import faiss
from openai import OpenAI

from config import (
    EMBED_CACHE,
    OPENAI_API_KEY, EMBED_MODEL,
    BATCH_SIZE, SLEEP_BETWEEN,
    META_PATH, INDEX_PATH, META_PKL, DATA_DIR,
)


client = OpenAI(api_key=OPENAI_API_KEY)



# ─────────────────────────────────────────────
# 数据读取 & 文本拼接
# ─────────────────────────────────────────────

def load_records() -> list[dict]:
    records = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def build_text(record: dict) -> str:
    parts = []
    if record.get("tag_string_general"):
        parts.append(f"general: {record['tag_string_general']}")
    if record.get("tag_string_character"):
        parts.append(f"character: {record['tag_string_character']}")
    if record.get("tag_string_copyright"):
        parts.append(f"copyright: {record['tag_string_copyright']}")
    if record.get("tag_string_artist"):
        parts.append(f"artist: {record['tag_string_artist']}")
    return " | ".join(parts)


# ─────────────────────────────────────────────
# Embedding 缓存
# ─────────────────────────────────────────────

def load_embed_cache() -> dict[int, list[float]]:
    """读已缓存的 embedding。
    返回 {post_id: vector} 字典。
    """
    if not EMBED_CACHE.exists():
        return {}
    with open(EMBED_CACHE, "rb") as f:
        return pickle.load(f)


def save_embed_cache(cache: dict[int, list[float]]):
    """把 embedding 缓存写回磁盘。"""
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(cache, f)


# ─────────────────────────────────────────────
# OpenAI Embedding API
# ─────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def embed_new_records(new_records: list[dict]) -> dict[int, list[float]]:
    """对新增记录算 embedding,返回 {id: vector} 字典。"""
    new_embeddings = {}
    total = len(new_records)
    
    for start in range(0, total, BATCH_SIZE):
        batch = new_records[start : start + BATCH_SIZE]
        texts = [build_text(r) for r in batch]
        
        for attempt in range(3):
            try:
                vecs = embed_batch(texts)
                break
            except Exception as e:
                print(f"  ⚠️ 第 {attempt+1} 次失败: {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        
        for record, vec in zip(batch, vecs):
            new_embeddings[record["id"]] = vec
        
        done = start + len(batch)
        print(f"  embedding 进度: {done} / {total}")
        
        if done < total:
            time.sleep(SLEEP_BETWEEN)
    
    return new_embeddings


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def build_index():
    """增量构建 FAISS 索引。"""
    
    print("📂 读取 metadata...")
    records = load_records()
    print(f"   共 {len(records)} 条记录")
    
    if not records:
        print("❌ 没有数据,先跑爬虫")
        return
    
    # ─── 增量逻辑 ──────────────────────────
    print("\n📦 读取 embedding 缓存...")
    cache = load_embed_cache()
    print(f"   已缓存 {len(cache)} 条")
    
    # 找出还没算过 embedding 的记录
    new_records = [r for r in records if r["id"] not in cache]
    print(f"   需要新算 {len(new_records)} 条")
    
    if new_records:
        print("\n🧠 计算新 embedding...")
        new_embeddings = embed_new_records(new_records)
        cache.update(new_embeddings)
        
        save_embed_cache(cache)
        print(f"💾 缓存已更新,共 {len(cache)} 条")
    else:
        print("✅ 所有数据都已有缓存,无需调 API")
    
    # ─── 按 records 顺序拿出向量,建 FAISS ──
    print("\n🔧 构建 FAISS 索引...")
    vectors = np.array(
        [cache[r["id"]] for r in records],
        dtype=np.float32,
    )
    
    faiss.normalize_L2(vectors)
    
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    print(f"   索引大小: {index.ntotal} 条向量,维度 {dim}")
    
    # ─── 保存 ─────────────────────────────
    print("\n💾 保存到磁盘...")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PKL, "wb") as f:
        pickle.dump(records, f)
    
    print(f"   ✅ FAISS 索引: {INDEX_PATH}")
    print(f"   ✅ 元数据:    {META_PKL}")
    print(f"\n🎉 完成,共索引 {index.ntotal} 条记录")


if __name__ == "__main__":
    build_index()