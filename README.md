# YouTube RAG Chatbot

A notebook-based chatbot that answers questions about a YouTube video using its transcript. It retrieves relevant transcript chunks with semantic search and passes them to an LLM to generate an answer.

## How It Works

1. **Transcript extraction** — fetches English captions using `youtube-transcript-api`.
2. **Chunking** — splits the transcript into overlapping chunks with `RecursiveCharacterTextSplitter` (400 characters, 200-character overlap).
3. **Embeddings** — converts chunks into vectors using Hugging Face `all-MiniLM-L6-v2`.
4. **Vector search** — stores and searches vectors with FAISS; retrieves the top 4 similar chunks.
5. **Answer generation** — sends the retrieved context and question to Groq's `openai/gpt-oss-120b` model through LangChain.

## Tech Stack

- Python and Google Colab / Jupyter
- YouTube Transcript API
- LangChain
- Hugging Face sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- FAISS (CPU)
- Groq API (`openai/gpt-oss-120b`)

## Run in Google Colab

1. Open the notebook: [`youtube_chatbot_RAG.ipynb`](https://colab.research.google.com/drive/1RCmJ6K4wJhO7veM-4fQqZPmLXxWPdZJK?usp=sharing).
2. Run the installation and import cells.
3. Set `video_id` to the ID of a YouTube video with an available English transcript.
4. When prompted, enter your Groq API key. Keep the key private; do not commit it to GitHub.
5. Run the remaining cells to build the FAISS index and ask questions about the video.

## Example

```python
result = main_chain.invoke("How does ChatGPT work?")
print(result)
```

## Current Scope

- Works with a single video transcript at a time.
- Requires an available English transcript.
- Uses in-memory FAISS indexing; the index is rebuilt when the notebook runs.
- Answers depend on the transcript content retrieved for each question.

## Dependencies

The notebook installs its dependencies with:

```bash
pip install youtube-transcript-api langchain-community langchain-groq langchain-text-splitters faiss-cpu tiktoken python-dotenv langchain langchain_huggingface
```

## Notes

This is a learning project demonstrating a basic Retrieval-Augmented Generation (RAG) workflow: transcript ingestion, text chunking, embeddings, vector retrieval, and LLM-based question answering.
