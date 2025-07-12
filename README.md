# Research Challenge - Entity Search

submission_06_21_09.32.csv => 0.43883 => expand_query_sbert
submission_test.csv        => 0.43705 => (tfidf ** pl2 ** bm25)
submission_06_20_22.37.csv => 0.43703 => variantes dos parâmetros 43705 => hits, test_size parametros dos retrievers
submission_06_19_19.35.csv => 0.43638 => variantes dos parâmetros 43705 => parametros campos
submission_06_19_19.11.csv => 0.43287 => variantes dos parâmetros 43705 => parametros bm 25

## Run
---
To execute best score run

```bash
 python3 -B src/index/index.py
 python3 -B src/query_processor/query_processor.py
```