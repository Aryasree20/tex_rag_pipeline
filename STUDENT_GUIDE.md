# Text-RAG Pipeline - Detailed Student Guide

## 1. High-level overview

Retrieval-Augmented Generation, or RAG, combines search with text generation. Instead of asking a model to answer only from its training, the application first retrieves relevant evidence from a controlled knowledge base and then supplies that evidence to the model.

This project contains two connected flows.

### Indexing-time flow

```text
Markdown files
    -> heading-based sections
    -> overlapping chunks
    -> OpenAI embeddings
    -> persistent Chroma collection
```

Indexing prepares the knowledge base before a user asks a question.

### Query-time flow

```text
question
    -> question embedding
    -> nearest Chroma chunks
    -> labelled context
    -> OpenAI Responses API
    -> grounded answer with sources
```

The main lesson is that a generated answer is only as reliable as the evidence retrieved for it. The demo therefore prints retrieved chunks before it prints the final answer.

## 2. Project design

The code is divided by responsibility:

- `rag_demo.py` tells the complete classroom story in order.
- `utils.py` implements each indexing and query stage as a small function.
- `evaluate.py` measures retrieval separately from generation.
- `test_code.py` checks important behavior without an API call.
- `reset_local_db.py` resets only the generated local database.

The project uses the OpenAI SDK and Chroma directly. There is no LangChain chain, router, notebook state, or hidden orchestration layer.

## 3. Setup and execution

From the `text_rag_pipeline` folder:

```powershell
.venv\Scripts\python.exe test_code.py
.venv\Scripts\python.exe reset_local_db.py
.venv\Scripts\python.exe rag_demo.py
.venv\Scripts\python.exe rag_demo.py
.venv\Scripts\python.exe evaluate.py
```

What each execution proves:

1. `test_code.py` proves the local logic works without the network.
2. `reset_local_db.py` creates a clean starting point.
3. The first `rag_demo.py` run embeds and stores source chunks.
4. The second run reuses persisted document embeddings.
5. `evaluate.py` tests the retriever on six labelled questions.

## 4. Explanation of every file

### `rag_demo.py`

The main classroom program. It loads documents, chunks them, opens Chroma, indexes new chunks, retrieves evidence for one question, and generates a source-grounded answer.

### `utils.py`

The reusable RAG functions. Every major arrow in the indexing-time and query-time diagrams has a corresponding function here.

### `evaluate.py`

Retrieval evaluation. It calculates Precision@K and Recall@K for the gold questions. It also includes Answer F1 as a transparent example of generation scoring.

### `test_code.py`

An offline confidence check. It uses fixed two-dimensional vectors, so the expected nearest result is obvious and no OpenAI request is needed.

### `reset_local_db.py`

Deletes only the `chroma_store` directory created by this project. Run it after changing source content, chunk size, overlap, or embedding model.

### `data/company_policy.md`

A synthetic policy document covering travel, expenses, remote work, and laptop replacement. It provides factual questions for the RAG system.

### `data/applied_rag_notes.md`

Technical notes about RAG, chunking, grounding, retrieval evaluation, and failure modes.

### `data/support_handbook.md`

Operational notes about API setup, Chroma persistence, empty retrieval, and cost control.

### `data/gold_questions.json`

Six evaluation records. Each record has an ID, question, expected answer, and the `source_id` values known to contain evidence.

### `requirements.txt`

Pins only three direct dependencies: Chroma, the OpenAI SDK, and python-dotenv.

### `.env`

Contains the real local API key and selected chat model. It must not be displayed, shared, or committed.

### `.env.sample`

Shows the supported variable names using safe placeholder values.

### `.gitignore`

Prevents secrets, the virtual environment, Python caches, and the generated Chroma store from being committed.

### `README.md`

Short setup and run reference.

### `TRAINER_README.md`

Instructor-only run order, slide mapping, discussion prompts, and rehearsal checklist.

## 5. `rag_demo.py` in detail

### Constants

- `BASE_DIR` is the folder containing the script.
- `DATA_DIR` points to the source Markdown files.
- `STORE_PATH` points to the generated persistent Chroma database.

Using paths relative to the script makes the program work even when the terminal starts elsewhere.

### `main()`

This function coordinates five visible steps:

1. `load_markdown_sections()` loads structured source sections.
2. `chunk_sections()` creates chunks and stable chunk IDs.
3. `open_collection()` opens persistent Chroma storage.
4. `index_chunks()` embeds and stores documents only when required.
5. `answer_question()` retrieves evidence and generates an answer.

It calls `print_retrieved_chunks()` before displaying the answer so retrieval remains inspectable.

The `if __name__ == "__main__"` block runs `main()` when this file is executed directly.

## 6. Every function in `utils.py`

### `slugify(text)`

Converts a heading such as `Domestic Travel Reimbursement` into `domestic_travel_reimbursement`. This creates predictable source IDs used by retrieval evaluation.

### `load_markdown_sections(data_dir)`

Finds all Markdown files, reads them as UTF-8, detects headings with a regular expression, and makes one dictionary per section. Each dictionary contains:

- `text`;
- `source` file name;
- `section_title`;
- stable `source_id`;
- Markdown `heading_level`.

It raises `FileNotFoundError` when the data folder contains no Markdown documents.

### Nested `save_section()`

An internal helper inside `load_markdown_sections()`. It saves the current heading and accumulated lines only when the section contains readable text. Keeping it nested shows that it is used only during one file-loading operation.

### `split_text(text, chunk_size=300, overlap=50)`

Splits a long section into overlapping character windows. Before cutting, it prefers a paragraph boundary, then a sentence ending, then a space. This reduces awkward mid-sentence chunks while keeping the algorithm visible.

The classroom corpus uses a small 300-character limit so students can observe splitting on short files. Real projects should tune chunk size using their document structure and retrieval evaluation rather than copying this number unchanged.

It validates that chunk size is positive and overlap is smaller than chunk size.

### `chunk_sections(sections, chunk_size=300, overlap=50)`

Calls `split_text()` for every section, copies the section metadata into every resulting chunk, and creates stable IDs such as:

```text
company_policy#domestic_travel_reimbursement::chunk_01
```

### `get_openai_client()`

Loads this folder's `.env`, checks `OPENAI_API_KEY`, and returns an authenticated `OpenAI` client. It raises a clear error instead of allowing a confusing authentication failure later.

### `create_embeddings(texts)`

Converts an iterable to a list, verifies that it is non-empty, reads the embedding model, and sends the strings in one batched embeddings request. Returned vectors stay in the same order as the input text.

### `open_collection(store_path, collection_name='text_rag_chunks')`

Creates a local `PersistentClient` and gets or creates the named Chroma collection. It returns both the client and collection so the caller can close the database cleanly on Windows.

### `index_chunks(collection, chunks)`

Implements persisted indexing:

- An empty collection causes all chunk text to be embedded and stored.
- A collection with the expected count is reused.
- A mismatched count raises an error asking for a reset, preventing old and new chunk layouts from being mixed.

Chroma receives parallel lists of chunk IDs, document text, metadata, and embeddings.

### `retrieve_chunks(collection, question, top_k=4)`

Embeds the question, asks Chroma for the nearest chunks, and converts Chroma's nested response into simple ranked dictionaries containing chunk ID, text, metadata, and distance.

`top_k` controls how many evidence chunks are returned. Lower Chroma distance means a closer vector match.

### `format_context(retrieved)`

Converts retrieved rows into one labelled context string. Every block starts with its source filename and section, allowing the generator to cite the evidence.

### `generate_answer(question, retrieved)`

Uses the OpenAI Responses API. The instructions require three behaviors:

1. use only supplied context;
2. admit when evidence is insufficient;
3. end with a sources-used line.

The formatted question and retrieved context are passed as the request input. The function returns `response.output_text`.

### `answer_question(collection, question, top_k=4)`

Combines query-time retrieval and generation. It returns one dictionary containing the original question, retrieved evidence, and generated answer.

### `print_retrieved_chunks(retrieved)`

Prints rank, Chroma distance, source, section, and a short text preview. It contains display logic only.

## 7. Every function in `evaluate.py`

### `load_gold_questions(path)`

Reads `gold_questions.json` and returns its list of labelled evaluation records.

### `precision_at_k(retrieved_ids, relevant_ids, k)`

Answers: "How clean were the first k results?"

```text
Precision@K = relevant retrieved results / K
```

### `recall_at_k(retrieved_ids, relevant_ids, k)`

Answers: "How much known evidence did retrieval find?"

```text
Recall@K = relevant source IDs found / all known relevant source IDs
```

### `answer_f1(prediction, expected)`

Measures word overlap between a generated answer and a reference answer. It calculates word precision and word recall, then combines them using the F1 formula.

Answer F1 is educational but limited: a correct paraphrase can receive a low score, and word overlap does not prove faithfulness.

### Nested `tokens(text)`

Lowercases text and extracts simple letter-and-number tokens for `answer_f1()`.

### `evaluate_retrieval(collection, gold_questions, top_k=4)`

Retrieves chunks for each labelled question, extracts their `source_id` metadata, calculates Precision@K and Recall@K, and returns one metric row per question. It does not call the generation model.

### `main()`

Loads the gold set and existing Chroma collection, verifies that indexing has already occurred, runs retrieval evaluation, prints each question's metrics, and prints mean precision and recall.

## 8. `reset_local_db.py` in detail

### `STORE_PATH`

Points only to this project's `chroma_store` directory.

### `main()`

Deletes that directory if it exists; otherwise it prints that there is nothing to remove. It does not delete source data or contact a cloud service.

## 9. `test_code.py` in detail

### `BASE_DIR`

Locates the project and data files reliably.

### `main()`

The offline test checks:

- Markdown section loading;
- chunk creation and metadata;
- Precision@K and Recall@K arithmetic;
- Answer F1 arithmetic;
- Chroma insertion and similarity search;
- context formatting;
- `answer_question()` orchestration with a fixed local answer function.

It temporarily replaces `create_embeddings()` with a fixed vector function. That substitution is test isolation, not production routing logic. The real function is restored before the test exits.

The temporary Chroma client is closed before Windows removes its files.

## 10. What to observe during the demo

- The system loads sections rather than treating every file as one giant document.
- Each chunk retains enough metadata to trace it to a source.
- The first run performs document embedding; the second does not.
- Retrieved evidence is visible before generation.
- The answer should reflect the policy amount and identify its source section.
- Evaluation can reveal retrieval problems even when an answer sounds fluent.

## 11. Common mistakes

- Changing chunk settings without resetting the persisted collection.
- Using different embedding models for stored chunks and questions.
- Increasing top-k without inspecting the extra noise.
- Losing source metadata during chunking.
- Judging only the final answer instead of checking retrieved evidence.
- Treating Answer F1 as a faithfulness score.
- Expecting RAG to retrain the language model.

## 12. Suggested experiments

- Change `top_k=4` to `top_k=2` and compare retrieval.
- Change chunk size, reset the store, and compare metrics.
- Ask an unanswerable question and inspect the refusal.
- Add a new Markdown section and a matching gold question.
- Intentionally remove a relevant section and observe recall.
