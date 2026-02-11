# Mistral RAG (Streamlit + Qdrant)

A small RAG app using Mistral models, Streamlit UI, Langchain and Qdrant as vector database.

## Prerequisites

- Docker
- Docker Compose
- A Mistral API key

## Quick Start

1. Create an env file:

```bash
cp app/.env.example app/.env
```
Then replace de key with your actual Mistral API KEY
You can get your free mistral API key here -> [Mistral Console](https://console.mistral.ai/home).


2. Start the stack:

```bash
docker compose up --build -d
```

3. Open the app:

- http://localhost:8501

## How to Use

1. Upload a PDF or TXT file in the sidebar.
2. Click **Indexer le document**.
3. Ask questions in the chat input.

## Useful Commands

Start without rebuild:

```bash
docker compose up -d
```

Stop services:

```bash
docker compose down
```
