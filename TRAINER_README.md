# Trainer Runbook - Text-RAG Pipeline

## Main classroom command

```powershell
.venv\Scripts\python.exe rag_demo.py
```

Run it twice during rehearsal. The first run should embed and store chunks. The second should reuse the saved Chroma collection.

## Before class

```powershell
.venv\Scripts\python.exe test_code.py
.venv\Scripts\python.exe reset_local_db.py
.venv\Scripts\python.exe rag_demo.py
.venv\Scripts\python.exe rag_demo.py
.venv\Scripts\python.exe evaluate.py
```

Confirm:

- offline checks pass;
- three Markdown documents load successfully;
- chunks retain source and section metadata;
- the second demo avoids re-embedding documents;
- retrieved evidence includes the domestic travel section;
- the generated answer cites sources;
- evaluation prints Precision@4 and Recall@4.

Reset once more before class if students should see first-time indexing.

## Presentation-to-code teaching flow

### Slides 3-8 - Why RAG exists

Open `rag_demo.py`. Explain that vector search returns passages, while `answer_question()` adds controlled generation. Emphasize that RAG retrieves at query time and does not retrain the model.

### Slides 9-15 - Indexing-time pipeline and chunking

Show these calls:

```python
sections = load_markdown_sections(DATA_DIR)
chunks = chunk_sections(sections, chunk_size=300, overlap=50)
```

Open one data file and identify the heading, text, source file, and stable `source_id`. Show `split_text()` only after students understand why chunks exist.

### Slides 16-21 - Storage and retrieval

Show `open_collection()`, `index_chunks()`, and `retrieve_chunks()`. Run the script and pause before generation to inspect the retrieved evidence.

Talking points:

- Chroma stores IDs, text, metadata, and vectors.
- `top_k=4` is a design choice, not a universal best value.
- The project uses dense retrieval only, matching the slide recommendation.
- There is deliberately no router in the first working RAG version.

### Slides 22-24 - Grounded generation

Show `format_context()` and `generate_answer()`. Point out three prompt requirements:

1. answer only from context;
2. admit insufficient evidence;
3. finish with sources used.

Then display the generated answer.

### Slides 25-30 - Evaluation and diagnosis

Run:

```powershell
.venv\Scripts\python.exe evaluate.py
```

Explain that retrieval is evaluated separately from generation. Use `answer_f1()` only as a teaching example: it is simple word overlap and cannot prove grounding.

## Code call path

```text
rag_demo.main
├─ load_markdown_sections
├─ chunk_sections -> split_text
├─ open_collection
├─ index_chunks -> create_embeddings
└─ answer_question
   ├─ retrieve_chunks -> create_embeddings
   └─ generate_answer -> format_context -> Responses API
```

## Questions to ask students

- What happens if a useful sentence is split across two chunks?
- Why retain both `source_id` and `chunk_id`?
- Why inspect evidence before generation?
- What is the risk of increasing top-k too far?
- What does high recall but low precision mean?
- Why can Answer F1 disagree with human judgment?

## If time is short

Teach loading, chunking, persisted indexing, retrieval inspection, one generated answer, and the meaning of Precision@K versus Recall@K. Skip editing chunk settings live.

## Safety and cost

- The included corpus is synthetic teaching material.
- Never project the `.env` file.
- The first demo embeds the corpus; later demos reuse it.
- Each demo still embeds the question and makes one generation request.
- `reset_local_db.py` deletes only `chroma_store/` in this project.
