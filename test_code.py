"""Run offline checks for loading, chunking, Chroma, and evaluation metrics."""

from pathlib import Path
from tempfile import TemporaryDirectory

import utils
from evaluate import answer_f1, precision_at_k, recall_at_k

BASE_DIR = Path(__file__).parent


def main() -> None:
    """Run deterministic checks without calling the OpenAI API."""
    sections = utils.load_markdown_sections(BASE_DIR / "data")
    chunks = utils.chunk_sections(sections, chunk_size=300, overlap=50)
    assert sections
    assert chunks
    assert all("source_id" in chunk and "chunk_id" in chunk for chunk in chunks)

    assert precision_at_k(["A", "B", "C"], ["A", "C"], 3) == 2 / 3
    assert recall_at_k(["A", "B", "C"], ["A", "C"], 3) == 1.0
    assert answer_f1("travel limit is 3000", "travel limit is 3000") == 1.0

    with TemporaryDirectory() as temporary_directory:
        client, collection = utils.open_collection(
            Path(temporary_directory) / "chroma", "offline_test"
        )
        collection.add(
            ids=["travel", "security"],
            documents=["travel reimbursement limit", "remote work VPN"],
            metadatas=[
                {"source": "policy.md", "section_title": "Travel", "source_id": "policy#travel"},
                {"source": "policy.md", "section_title": "Security", "source_id": "policy#security"},
            ],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        original_embedding_function = utils.create_embeddings
        original_generation_function = utils.generate_answer
        utils.create_embeddings = lambda texts: [[1.0, 0.0] for _ in texts]
        utils.generate_answer = lambda question, retrieved: "Grounded test answer."
        retrieved = utils.retrieve_chunks(collection, "travel question", top_k=1)
        result = utils.answer_question(collection, "travel question", top_k=1)
        utils.create_embeddings = original_embedding_function
        utils.generate_answer = original_generation_function
        assert retrieved[0]["chunk_id"] == "travel"
        assert "policy.md" in utils.format_context(retrieved)
        assert result["answer"] == "Grounded test answer."
        client.close()

    print("All Text-RAG Pipeline offline checks passed.")


if __name__ == "__main__":
    main()
