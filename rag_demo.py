"""Run the complete Text-RAG Pipeline classroom demonstration.

The script follows the presentation: load, chunk, embed, store, retrieve,
inspect evidence, generate a grounded answer, and display sources.
"""

from pathlib import Path

from utils import (
    answer_question,
    chunk_sections,
    index_chunks,
    load_markdown_sections,
    open_collection,
    print_retrieved_chunks,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = BASE_DIR / "chroma_store"


def main() -> None:
    """Build or reuse the knowledge base and answer one sample question."""
    print("STEP 1 - Load source documents by Markdown section")
    sections = load_markdown_sections(DATA_DIR)
    print(f"Loaded {len(sections)} sections.")

    print("\nSTEP 2 - Split sections into retrievable chunks")
    chunks = chunk_sections(sections, chunk_size=300, overlap=50)
    print(f"Created {len(chunks)} chunks with source metadata.")

    print("\nSTEP 3 - Open persistent Chroma storage")
    client, collection = open_collection(STORE_PATH)
    if index_chunks(collection, chunks):
        print("Embedded and stored the chunks.")
    else:
        print("Reused the saved collection; document embeddings were not rebuilt.")

    print("\nSTEP 4 - Retrieve evidence before asking the generator")
    question = "What is the domestic travel reimbursement limit?"
    result = answer_question(collection, question, top_k=4)
    print(f"Question: {question}\n")
    print_retrieved_chunks(result["retrieved"])

    print("\nSTEP 5 - Generate an answer from the retrieved evidence")
    print(result["answer"])
    client.close()


if __name__ == "__main__":
    main()
