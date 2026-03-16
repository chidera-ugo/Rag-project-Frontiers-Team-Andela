## Improvements

### 1. Fix retrieval count (k=3 → k=20)
Baseline returned only 3 docs, capping MRR/nDCG at positions 4–10. Replaced with 20 docs then rerank to 10.

### 2. Smaller, smarter chunks (1000 → 500 chars)
Reduce chunk size to reduce noise and then switched to RecursiveTextSplitter for more semantic chunking

### 3. Stamp document title onto every chunk
We append document title to each chunk to serve as a pointer, preserving context

### 4. LLM reranking (20 candidates → top 10)
Vector similarity ranks by broad semantic match. After retrieving 20 candidates, `gpt-4.1` reorders them by relevance to the specific question. Top 10 kept.

### 5. Rephrase user query + dual search
`gpt-4.1` rewrites the question into a targeted KB query, then both original and updated are searched, merged, deduplicated, and reranked together.

### 6. Upgrade embeddings model (all-MiniLM-L6-v2 → text-embedding-3-large)


## Testing Guide

```bash
uv run ev.py -r              # v2 retrieval eval
uv run ev.py -a              # v2 answer eval
uv run logs.py               # view full history + comparison table
```
