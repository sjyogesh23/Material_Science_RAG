# NewRAG

An offline chatbot that answers questions about a materials-science textbook. Every answer cites the
book and can highlight the exact passage on the PDF page. Questions outside the book get `#OutofContext`.

Runs fully on your machine: local models via [Ollama](https://ollama.com), an embedded Postgres
database, and a FastAPI web app. It also includes token-usage and architecture pages, and an
evaluation dashboard.

## Prerequisites

Install these first:

| What | Why | Download |
|---|---|---|
| Python 3.12 | runs the app | [python.org/downloads](https://www.python.org/downloads/) (on Windows, keep the "py launcher" option checked) |
| Ollama | serves the chat, embedding and reranker models | [ollama.com/download](https://ollama.com/download) |
| Git | clone the repo | [git-scm.com/downloads](https://git-scm.com/downloads) |
| Microsoft Visual C++ Redistributable (Windows only) | needed by PyTorch | [learn.microsoft.com - latest VC++ redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) |

Optional:

| What | Why | Download |
|---|---|---|
| NVIDIA GPU driver | faster answers and ingest; everything also runs on CPU | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) |
| PyTorch with CUDA | GPU build of torch matching your driver | [pytorch.org/get-started](https://pytorch.org/get-started/locally/) |
| Hugging Face token | higher download limits for the Hugging Face models (put it in `HF_TOKEN` in `.env`) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

Nothing else needs to be installed by hand:

- **Postgres** is embedded (`pgserver`, installed by pip), with its data in `pgdata/`.
- **Models** are downloaded by `run.py` on the first start: `qwen3:4b-instruct-2507-q4_K_M` (chat, 2.5 GB) and
  `qwen3-embedding:0.6b` (0.6 GB). `run.py` also builds the reranker `qwen3-reranker:0.6b-q8_0` (0.6 GB) from
  [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
  The optional "deep" mode uses `qwen3.5:9b` (6.6 GB): `ollama pull qwen3.5:9b`.
- **The book** is in the repo (`data/textbooks/`, already extracted and chunked). It is
  [Steimel, *Materials Science and Engineering*](https://scholarlycommons.pacific.edu/open-textbooks/8),
  licensed CC BY-NC-ND 4.0.

Plan for about 10 GB of free disk space (Python packages + models) and 8 GB of RAM.

## Run it

Windows:

```powershell
py -3.12 -m venv .venv
# NVIDIA GPU only: install the CUDA build of torch first (pick the CUDA version for your driver on pytorch.org)
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python run.py
```

Linux / macOS:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python run.py
```

`run.py` starts the database and Ollama, downloads missing models, loads the book, then prints the links:

- Chat: http://127.0.0.1:8000/
- Usage, feedback & architecture: http://127.0.0.1:8000/audit
- Evaluation: http://127.0.0.1:8000/eval

Press Ctrl+C to stop. Options: `--port 8000`, `--no-browser`, `--ingest` (re-load the book into the database).

## Tests

```powershell
.venv\Scripts\python -m pytest
```
