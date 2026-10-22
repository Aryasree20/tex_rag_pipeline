"""Beginner-friendly building blocks for the Text-RAG Pipeline class.

The functions follow the presentation's two flows. Indexing functions load,
chunk, embed, and store documents. Query functions retrieve evidence, format
context, and ask an OpenAI model to answer only from that evidence.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


def slugify(text: str) -> str:
    """Convert a heading into a stable lowercase identifier.

    Args:
        text: Markdown heading such as ``Domestic Travel Reimbursement``.

    Returns:
        A value such as ``domestic_travel_reimbursement``.
    """
    lowercase = text.strip().lower()
    without_punctuation = re.sub(r"[^a-z0-9]+", "_", lowercase)
    return without_punctuation.strip("_")


def load_markdown_sections(data_dir: Path) -> list[dict[str, Any]]:
    """Load Markdown files and split them at headings.

    Each section keeps source metadata so retrieved evidence can be displayed
    and evaluated later.

    Args:
        data_dir: Folder containing the source ``.md`` files.

    Returns:
        Section dictionaries containing text and source metadata.
    """
    markdown_files = sorted(data_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No Markdown files found in {data_dir}")

    sections: list[dict[str, Any]] = []
    for file_path in markdown_files:
        heading = file_path.stem.replace("_", " ").title()
        heading_level = 1
        lines: list[str] = []

        def save_section() -> None:
            """Save the current section when it contains readable text."""
            text = "\n".join(lines).strip()
            if not text:
                return
            sections.append(
                {
                    "text": text,
                    "source": file_path.name,
                    "section_title": heading,
                    "source_id": f"{file_path.stem}#{slugify(heading)}",
                    "heading_level": heading_level,
                }
            )

        for line in file_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                save_section()
                heading_level = len(match.group(1))
                heading = match.group(2).strip()
                lines = []
            else:
                lines.append(line)
        save_section()

    return sections


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split long text into overlapping chunks at readable boundaries.

    The function first looks for a paragraph break, then a sentence ending,
    then a space. This is easier to inspect than a framework abstraction while
    still avoiding most mid-sentence cuts.

    Args:
        text: Section text to split.
        chunk_size: Maximum approximate characters in one chunk.
        overlap: Approximate characters repeated in the next chunk.

    Returns:
        Ordered text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size.")
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            candidates = [
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            readable_end = max(candidates)
            if readable_end > start + chunk_size // 2:
                end = readable_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break

        next_start = max(0, end - overlap)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start if next_start < end else end

    return chunks


def chunk_sections(
    sections: list[dict[str, Any]],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """Create retrievable chunks while preserving section metadata.

    Args:
        sections: Values returned by :func:`load_markdown_sections`.
        chunk_size: Maximum approximate characters per chunk.
        overlap: Approximate repeated characters between adjacent chunks.

    Returns:
        Chunk dictionaries with stable ``chunk_id`` values.
    """
    chunks: list[dict[str, Any]] = []
    for section in sections:
        text_chunks = split_text(section["text"], chunk_size, overlap)
        for number, text in enumerate(text_chunks, start=1):
            chunk = dict(section)
            chunk["text"] = text
            chunk["chunk_id"] = f"{section['source_id']}::chunk_{number:02d}"
            chunks.append(chunk)
    return chunks


def get_openai_client():
    """Load this project's ``.env`` and create an authenticated OpenAI client.

    Returns:
        An ``OpenAI`` client.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is missing.
    """
    from openai import OpenAI

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Add OPENAI_API_KEY to this folder's .env file.")
    return OpenAI(api_key=api_key)


def create_embeddings(texts: Iterable[str]) -> list[list[float]]:
    """Create embeddings for multiple strings in one API request.

    Args:
        texts: Document chunks or questions to embed.

    Returns:
        Embedding vectors in the same order as the input strings.
    """
    text_list = [str(text) for text in texts]
    if not text_list:
        raise ValueError("At least one text is required for embedding.")

    load_dotenv(Path(__file__).with_name(".env"))
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    response = get_openai_client().embeddings.create(
        model=model,
        input=text_list,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


def open_collection(store_path: Path, collection_name: str = "text_rag_chunks"):
    """Open or create a persistent Chroma collection.

    Args:
        store_path: Directory used for local Chroma persistence.
        collection_name: Stable collection name.

    Returns:
        A ``(client, collection)`` pair. Close the client when finished.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(store_path))
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def index_chunks(collection: Any, chunks: list[dict[str, Any]]) -> bool:
    """Embed and store chunks when the Chroma collection is empty.

    Args:
        collection: Chroma collection returned by :func:`open_collection`.
        chunks: Chunk dictionaries returned by :func:`chunk_sections`.

    Returns:
        ``True`` if indexing occurred or ``False`` if persisted data was reused.
    """
    existing_count = collection.count()
    if existing_count:
        if existing_count != len(chunks):
            raise RuntimeError(
                "The saved collection does not match the current chunks. "
                "Run reset_local_db.py and then rerun rag_demo.py."
            )
        return False

    embeddings = create_embeddings(chunk["text"] for chunk in chunks)
    metadatas = [
        {
            "source": chunk["source"],
            "section_title": chunk["section_title"],
            "source_id": chunk["source_id"],
            "heading_level": chunk["heading_level"],
        }
        for chunk in chunks
    ]
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return True


def retrieve_chunks(
    collection: Any, question: str, top_k: int = 4
) -> list[dict[str, Any]]:
    """Retrieve the nearest chunks for a question.

    Args:
        collection: Populated Chroma collection.
        question: User question.
        top_k: Maximum number of chunks to retrieve.

    Returns:
        Ranked dictionaries containing text, metadata, and distance.
    """
    if collection.count() == 0:
        raise RuntimeError("The collection is empty. Run rag_demo.py to index the corpus.")

    query_embedding = create_embeddings([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
            }
        )
    return retrieved


def format_context(retrieved: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as labelled evidence for the generator.

    Args:
        retrieved: Rows returned by :func:`retrieve_chunks`.

    Returns:
        One text block containing source labels and chunk text.
    """
    blocks = []
    for item in retrieved:
        metadata = item["metadata"]
        label = f"{metadata['source']} | {metadata['section_title']}"
        blocks.append(f"[Source: {label}]\n{item['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, retrieved: list[dict[str, Any]]) -> str:
    """Generate an answer using only the retrieved evidence.

    Args:
        question: User question.
        retrieved: Evidence returned by :func:`retrieve_chunks`.

    Returns:
        Generated answer text with a sources line.
    """
    load_dotenv(Path(__file__).with_name(".env"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    context = format_context(retrieved)
    response = get_openai_client().responses.create(
        model=model,
        instructions=(
            "Answer only from the supplied context. If the context does not contain "
            "enough evidence, say that the provided sources do not contain enough "
            "information. End with a 'Sources used:' line listing source file and section."
        ),
        input=f"Question:\n{question}\n\nContext:\n{context}",
    )
    return response.output_text


def answer_question(collection: Any, question: str, top_k: int = 4) -> dict[str, Any]:
    """Run retrieval and generation for one question.

    Args:
        collection: Populated Chroma collection.
        question: User question.
        top_k: Maximum evidence chunks supplied to the generator.

    Returns:
        A dictionary containing the question, evidence, and generated answer.
    """
    retrieved = retrieve_chunks(collection, question, top_k)
    answer = generate_answer(question, retrieved)
    return {"question": question, "retrieved": retrieved, "answer": answer}


def print_retrieved_chunks(retrieved: list[dict[str, Any]]) -> None:
    """Print retrieved evidence before generation.

    Args:
        retrieved: Rows returned by :func:`retrieve_chunks`.
    """
    for item in retrieved:
        metadata = item["metadata"]
        print(
            f"{item['rank']}. distance={item['distance']:.3f} | "
            f"{metadata['source']} > {metadata['section_title']}"
        )
        print(f"   {item['text'][:180].replace(chr(10), ' ')}")
