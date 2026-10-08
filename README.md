# Word_Counter

Word frequency counter for the English Wikipedia XML dump.

```bash
clang++ -O3 -std=c++17 -o wiki_analyzer wiki_analyzer.cpp
./wiki_analyzer enwiki-latest-pages-articles.xml 100
```

## Performance (M3, 8 cores)

| Version | 4 GB sample | Full dump (107 GB) |
|---|---|---|
| Original | ~20 s | ~2.5 min |
| Current | ~3 s | ~73 s |

Main changes: workers parse their own byte ranges, a flat hash map with
inline short keys, a NEON tokenizer, and per-shard batched merging.
