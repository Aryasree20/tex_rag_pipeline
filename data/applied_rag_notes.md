# Applied RAG Design Notes

## What RAG Adds

Retrieval-Augmented Generation connects a search system to a language model. The retriever finds relevant evidence from external knowledge, and the generator uses that evidence to produce a grounded answer.

## Chunking Guidelines

For policy documents, product manuals, and reports, recursive chunking with moderate overlap is a reliable starting point. A chunk size between 500 and 900 tokens with 10 to 20 percent overlap often preserves local context while keeping retrieved context manageable. Section titles should be preserved as metadata because they help interpret short chunks.

## Grounded Generation Policy

A RAG answer should only use facts present in the retrieved context. If the context does not contain enough evidence, the answer should say that the information is not available in the provided sources. Source titles or section names should be returned with the answer.

## Retrieval Evaluation Metrics

Precision@K measures how many of the top K retrieved chunks are relevant. Recall@K measures how many known relevant evidence chunks were found within the top K results. Retrieval metrics should be reviewed before judging the language model's answer quality.

## Common Failure Modes

RAG systems often fail because chunks are too small, chunks are too large, metadata is missing, retrieval returns semantically similar but wrong evidence, or the prompt allows the model to answer beyond the retrieved context.
