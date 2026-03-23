# Minerva

Minerva is an experimental, privacy-first, Python-based AI assistant. It leverages local Large Language Models (LLMs) to converse, learn, and dynamically adapt to the user over time—all while keeping your data strictly on your own device. It is currently in heavy development and not ready for production use. Contributions welcome!

## Architecture & Features

Minerva operates on a **Tool-Based Memory Architecture** rather than passively summarizing chat logs. 
When conversing, the LLM is equipped with advanced `<think>` capabilities and two core tools: `retrieve` and `manage_memory`.

- **Active Retrieval (`retrieve`)**: Minerva natively indexes all your facts into a local SQLite database and vector index, bridged via an Entity Knowledge Graph. When you ask a question about your past or personal life, Minerva pauses text generation, calls the retrieval tool via XML tags, fetches semantic matches and topological `FactEdge` neighbors, and seamlessly injects the context back into the conversation thread to synthesize a perfect response.
- **Asynchronous Storage (`manage_memory`)**: Whenever Minerva learns new facts about you (or old facts change) during a chat, she proactively issues a multi-element JSON array payload to store, update, or delete them. Crucially, the active conversational LLM natively formats the topological `FactEdge` triplets directly inside the tool call. At the end of the generator stream, a background `MemoryOrchestrator` thread securely catches these pre-mapped relation chains, automatically handling fuzzy string resolutions and computing vector embeddings instantly, bypassing the need for heavy offline extraction pipelines.

## Core Modules
- **`src.config`**: Centralized configuration and prompt schema loading.
- **`src.utils`**: General utility functions like semantic similarities and standardized short-ID generation.
- **`src.memory.db`**: Unified SQLAlchemy ORM layer handling both active DB connections and schema models (`EntityNode`, `GraphEdge`, `EmbeddingIndex`).
- **`src.memory.orchestrator`**: Background queue manager dealing with the dense extraction/graph expansion work off the main thread.
- **`src.models.rag_chat`**: The intelligent conversation interceptor acting as the application entry point.

## Installation

Ensure you have a modern GPU and Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

To enable GPU acceleration via PyTorch with CUDA:

```bash
pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
### For llama-cpp-python:
With CUDA:
Make sure you have the CUDA toolkit installed and configured for your system. Multiple versions may lead to silent failures. Use `nvcc --version` to check your current version.

For Windows:
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --upgrade llama-cpp-python
```
For Linux:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade llama-cpp-python
```
With Vulkan:
For Windows:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on"; pip install --upgrade llama-cpp-python
```
For Linux:
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install --upgrade llama-cpp-python
```

*Note: Installation of `llama-cpp-python` with CUDA bindings can take up to 30 minutes to compile, depending on your system configuration. Please be patient and trust the process!*

## Usage

Start your personal assistant by running:
```bash
python -m src.chat
```

### Configuration
All model hyperparameters (context window, batch sizes, mmaps) and the model repo/filename are managed centrally in `config.toml`. You can modify this file to easily switch from the default `Qwen3-8B-GGUF` model to something else without touching any Python code.

### Programmatic Usage
You can easily wrap Minerva in your own applications (like a Discord bot or FastAPI server) using the `Chat` class:
```python
from src.chat import Chat

assistant = Chat()
# Automatically hooks into RAG capabilities and returns a generator
for token in assistant.send_message("Hello Minerva!", stream=True):
    print(token, end="", flush=True)
```

## Testing

The Minerva test suite strictly isolates your production `.db` files from its operations by booting isolated `sqlite:///:memory:` instances for the assertions.

To run the entire test suite (including graph building, semantic deletion, and RAG chat mechanics), use `pytest`:
```bash
pytest tests/
```