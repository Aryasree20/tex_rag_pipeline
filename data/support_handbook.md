# AI Assistant Support Handbook

## API Key Setup

The OpenAI API key must be stored in a local .env file as OPENAI_API_KEY. Keys must never be committed to source control or pasted into notebooks. Missing-key errors should explain how to create the .env file.

## Chroma Persistence

Chroma can persist a local collection to disk. Persisting the collection avoids re-embedding the same documents during repeated demos. When changing chunk size or overlap, reset the collection so the index reflects the new chunking strategy.

## Troubleshooting Empty Retrieval

If retrieval returns irrelevant or empty results, inspect the chunk text, metadata, query wording, collection path, and embedding model. A common classroom issue is querying an old persisted collection after changing the input documents.

## Demo Cost Control

A small classroom RAG demo can stay below ₹50 or well under one US dollar when the corpus is small, embeddings are generated once, and only a few LLM calls are made. Reusing a persisted Chroma collection is the most important cost-control step.
