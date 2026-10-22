# Text-RAG Pipeline

This class turns retrieval into a complete, source-grounded question-answering pipeline. It follows the presentation's sequence:

```text
load -> chunk -> embed -> store -> retrieve -> inspect -> generate -> evaluate
```

The implementation uses the OpenAI SDK and Chroma directly. It intentionally does not use LangChain, notebooks, routers, hybrid search, report frameworks, or hidden chains.

## Start here

- Students: read [`STUDENT_GUIDE.md`](STUDENT_GUIDE.md).
- Trainers: read [`TRAINER_README.md`](TRAINER_README.md).
- Main classroom script: `rag_demo.py`.

## Setup

From this folder in Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

This folder already has a local `.env` copied from the existing course setup. Never display, share, or commit it. The supported variables are documented in `.env.sample`.

## Recommended run order

```powershell
.venv\Scripts\python.exe test_code.py
.venv\Scripts\python.exe reset_local_db.py
.venv\Scripts\python.exe rag_demo.py
.venv\Scripts\python.exe rag_demo.py
.venv\Scripts\python.exe evaluate.py
```

- `test_code.py` is offline and free.
- The first `rag_demo.py` run embeds and stores the corpus.
- The second run reuses stored document embeddings.
- `evaluate.py` measures retrieval separately from generation.

## Files

| File | Purpose |
|---|---|
| `rag_demo.py` | Main indexing, retrieval, and grounded-generation lesson. |
| `utils.py` | Plain Python functions for every RAG stage. |
| `evaluate.py` | Precision@K, Recall@K, Answer F1, and retrieval evaluation. |
| `test_code.py` | Offline deterministic checks using fixed vectors. |
| `reset_local_db.py` | Removes only this project's local Chroma store. |
| `data/*.md` | Small teaching corpus with meaningful Markdown sections. |
| `data/gold_questions.json` | Questions with expected answers and source IDs. |
| `STUDENT_GUIDE.md` | Detailed student explanation of every file and function. |
| `TRAINER_README.md` | Teaching sequence, slide mapping, and rehearsal checklist. |

## Important behavior

- Retrieved evidence is printed before the generated answer.
- The prompt requires the model to use only supplied context and admit insufficient evidence.
- Source file and section labels are included in the context.
- Chroma is persisted in `chroma_store/` to avoid repeated document embedding.
- Changing source documents or chunk settings requires `reset_local_db.py`.
