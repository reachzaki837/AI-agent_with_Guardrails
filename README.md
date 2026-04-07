# AI Agent with Guardrails

A side-by-side comparison of a **bare AI agent** and a **guardrailed AI agent**, both enhanced with Retrieval-Augmented Generation (RAG) over company policy documents. The guardrailed agent uses [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) to enforce safety rules, redact PII, and validate responses against uploaded policies—all running **locally** via [Ollama](https://ollama.com/).

---

## Features

| Feature | Normal Agent | Guardrail Agent |
|---|---|---|
| RAG over policy PDFs | ✅ | ✅ |
| PII scrubbing (phone, email, SSN) | ❌ | ✅ |
| Input self-check (prompt injection, jailbreaks) | ❌ | ✅ |
| Output self-check (harmful content, credential leaks) | ❌ | ✅ |
| Policy-violation blocking | ❌ | ✅ |
| Runs fully locally | ✅ | ✅ |

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   NeMo Guardrails                   │
│                                                     │
│  INPUT RAILS                                        │
│  1. scrub_pii      ← Presidio (phone/email/SSN)     │
│  2. self_check_input ← llama3.2:3b (injection check)│
│                                                     │
│  RAG RETRIEVAL                                      │
│  3. policy_parser  ← ChromaDB + nomic-embed-text    │
│     (find top-k relevant policy chunks)             │
│                                                     │
│  LLM GENERATION                                     │
│  4. mistral:7b     ← Main model via Ollama          │
│                                                     │
│  OUTPUT RAILS                                       │
│  5. self_check_output ← llama3.2:3b (harm check)   │
│  6. check_policy_violation ← policy chunk vs reply  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Safe, Policy-Compliant Response
```

### Components

| File | Description |
|---|---|
| `guardrail_agent.py` | Guardrailed agent — NeMo pipeline with PII scrubbing, self-checks, and policy enforcement |
| `normal_agent.py` | Bare agent — RAG-enabled but no safety middleware |
| `policy_parser.py` | RAG engine: reads PDFs, chunks text, builds a ChromaDB vector store, and retrieves relevant policy excerpts |
| `config/config.yml` | NeMo model config (main: mistral:7b, self-check: llama3.2:3b) and rail definitions |
| `config/rails.co` | Colang flows: `scrub pii` and `check policy violation` |
| `config/actions.py` | Custom NeMo action: `scrub_pii` using Microsoft Presidio |
| `config/prompts.yml` | Prompt templates for `self_check_input` and `self_check_output` |

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### Required Ollama Models

Pull the three models used by the agents before running:

```bash
ollama pull mistral:7b          # Main conversational model
ollama pull llama3.2:3b         # Self-check safety model
ollama pull nomic-embed-text    # Embedding model for RAG
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/reachzaki837/AI-agent_with_Guardrails.git
cd AI-agent_with_Guardrails

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Install the spaCy English model (required by Presidio)
python -m spacy download en_core_web_lg
```

---

## Usage

### Start Ollama

Make sure Ollama is running before launching either agent:

```bash
ollama serve
```

### Run the Guardrailed Agent

```python
# guardrail_agent.py — edit the policy_paths list at the bottom of the file
agent = GuardrailAgent(policy_paths=[
    "/path/to/your/company_policy.pdf",
])
asyncio.run(agent.run())
```

```bash
python guardrail_agent.py
```

### Run the Normal (Bare) Agent

```python
# normal_agent.py — edit the policy_paths list at the bottom of the file
agent = NormalAgent(policy_paths=[
    "/path/to/your/company_policy.pdf",
])
agent.run()
```

```bash
python normal_agent.py
```

Type `quit` to exit either agent.

---

## How It Works

### RAG Pipeline (`policy_parser.py`)

1. **PDF Extraction** — `pypdf` reads all pages from one or more PDF files.
2. **Text Chunking** — `RecursiveCharacterTextSplitter` splits the content into 1,000-character chunks with a 200-character overlap.
3. **Embedding** — `nomic-embed-text` (via Ollama) converts each chunk into a vector.
4. **Vector Store** — All vectors are stored in an in-memory [ChromaDB](https://www.trychroma.com/) instance.
5. **Retrieval** — At query time, the top-*k* most relevant chunks are fetched and injected into the system prompt.

### Guardrail Pipeline (`guardrail_agent.py`)

1. **PII Scrubbing (Input Rail)** — Microsoft Presidio detects and replaces phone numbers, email addresses, and US SSNs with `[REDACTED]` before the query reaches the LLM.
2. **Self-Check Input (Input Rail)** — `llama3.2:3b` checks whether the user message attempts prompt injection, jailbreaking, or requests illegal actions.
3. **LLM Generation** — `mistral:7b` generates a response, optionally grounded by retrieved policy chunks.
4. **Self-Check Output (Output Rail)** — `llama3.2:3b` checks the bot's reply for harmful content or credential leaks.
5. **Policy Violation Check (Output Rail)** — The retrieved policy excerpt is compared against the bot's reply; if a violation is detected, the response is replaced with a refusal message.

---

## Configuration

### `config/config.yml`

Defines the models and active rails:

```yaml
models:
  - type: main          # Primary LLM
    engine: openai
    model: mistral:7b
    parameters:
      base_url: "http://127.0.0.1:11434/v1"
      api_key: "ollama"

  - type: self_check    # Safety checker LLM
    engine: openai
    model: llama3.2:3b
    parameters:
      base_url: "http://127.0.0.1:11434/v1"
      api_key: "ollama"

rails:
  input:
    flows: [scrub pii, self check input]
  output:
    flows: [self check output, check policy violation]
```

### `config/prompts.yml`

Customizable safety prompts for `self_check_input` and `self_check_output`. Edit these to tighten or relax the safety policies.

---

## Python packages and runtime components

Install the Python dependencies listed in `requirement.txt` for the default setup. The table below is a reference for the main libraries and local runtime components used by this project; some entries are optional or may need to be installed/configured separately from the Python requirements.

| Package / Component | Purpose |
|---|---|
| `nemoguardrails` | Guardrails framework and Colang runtime |
| `langchain-community` | LLM wrappers, ChromaDB integration |
| `langchain-ollama` | Ollama embeddings for LangChain |
| `langchain-openai` | OpenAI-compatible wrapper used by the normal agent configuration |
| `chromadb` | Local vector database |
| `ollama` | Local Ollama runtime and/or Python client, depending on how you run the project |
| `pypdf` | PDF text extraction |
| `presidio-analyzer` | PII detection |
| `presidio-anonymizer` | PII redaction |
| `spacy` | NLP backend for Presidio |
| `rich` | Terminal formatting |
| `pydantic` | Data validation |
| `python-docx` | (Optional) Word document support |

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
