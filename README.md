# CopyPicSearch

A image search project built with OpenAI Embeddings and FAISS.

The goal of this project is to explore Danbooru-style image retrieval using natural language queries

## Project Structure

```text
DANBOORU-PICTURE-IDENTIFICATION/
├── src/
│   ├── config.py
│   ├── crawler.py
│   ├── indexer.py
│   └── search.py
├── experimental/
│   ├── run_wd14_batch.py
│   └── tagger.py
├── README.md
├── searchresult.ipynb
├── .gitignore
└── .gitattributes
```

## Overview

The system has three components:

1. Crawler (crawler.py)
An incremental crawler built on the Danbooru API. Supports resuming from checkpoints, tag combinations, and filtering by rating and score. Re-running only fetches new images — no duplicates.

3. Indexer (indexer.py)
For each image, tags are concatenated into a text string, embedded using text-embedding-3-small, and stored in a FAISS index. Embeddings are cached, so re-running only calls the API for new images.

5. Search (search.py)
A natural language query is first passed to a GPT model, which rewrites it into Danbooru-style tags. Those tags are then embedded with text-embedding-3-small and matched against the FAISS index by cosine similarity, returning the top-K most relevant images.

## Usage
Set your OpenAI API key in config.py (or as an OPENAI_API_KEY environment variable). Danbooru credentials are optional.
Then run each stage:

bashpython crawler.py     # Crawl images into data/
python indexer.py     # Build the FAISS index

Once the index is built, open search.ipynb to run queries and view results inline.


## Roadmap
Planned next steps to make the search system more capable:

WD14 auto-tagging integration — Use the WD14 Tagger (a ViT-based vision model trained on Danbooru-style tags) to automatically tag images that don't come with metadata. This would extend the system beyond Danbooru to any image source. Inference scripts are already in place; threshold tuning and A/B evaluation against human tags are next.
Negative semantic search — Embedding models don't understand negation: querying "girls without exposed skin" tends to pull in the opposite of what the user wants. Plan to add a GPT-based query parser that splits queries into positive tags (for retrieval) and negative conditions (applied as a rerank-stage filter), so exclusion constraints actually work.
