# AI Engineering

This repository is a hands-on set of Python demos for understanding how modern language models work from the outside in: tokenization, embeddings, GPT-2 internals, sampling, streaming, log probabilities, structured outputs, and the broad training stages behind LLMs.

Most files are meant to be run directly as small teaching scripts. Some examples run fully locally, while others call the OpenAI API or download open-source models from Hugging Face.

## What You Will Learn

- How text becomes tokens and token IDs.
- How token embeddings and positional embeddings are used by GPT-style models.
- How sentence embeddings can represent semantic similarity.
- How decoding settings such as temperature, `top_k`, `top_p`, and `min_p` affect generation.
- How streaming works with normal HTTP, chunked responses, Server-Sent Events, and OpenAI streaming.
- How log probabilities can be used for confidence scoring and classification.
- How structured outputs make model responses easier to consume in code.
- How base models, supervised fine-tuning, and preference optimization differ conceptually.

## Project Structure

```text
.
|-- tokenizer.py
|-- vocabulary_printing.py
|-- basic_embedding.py
|-- token_embeddings_api_call.py
|-- GPT2_workflow.py
|-- sampling_deep_dive/
|-- streaming_demo/
|-- logprobs/
|-- structured_outputs/
`-- pretraining_sft_rlhf/
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install openai python-dotenv tiktoken torch transformers sentence-transformers scikit-learn numpy matplotlib fastapi uvicorn requests sseclient-py
```

For scripts that call the OpenAI API, create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_api_key_here
```

## Suggested Learning Path

Run the examples in this order if you are new to the topics.

### 1. Tokenization

```bash
python tokenizer.py
python vocabulary_printing.py
```

These scripts show how text is split into tokens and converted into token IDs using `tiktoken`.

### 2. Embeddings

```bash
python basic_embedding.py
python token_embeddings_api_call.py
```

`basic_embedding.py` uses a local sentence-transformer model to compare words in embedding space and plot them with PCA.

`token_embeddings_api_call.py` calls the OpenAI embeddings API for individual tokens and short phrases.

### 3. GPT-2 Workflow

```bash
python GPT2_workflow.py
```

This script walks through a GPT-2 forward pass:

- token IDs
- token embeddings
- positional embeddings
- token plus position inputs
- hidden states across layers
- next-token probabilities
- greedy generation

It also writes `token_embeddings.tsv`, which contains the first 100 GPT-2 token embeddings.

### 4. Sampling Deep Dive

```bash
python sampling_deep_dive/01_temperature_visualization.py
python sampling_deep_dive/02_topk_topp_minp.py
python sampling_deep_dive/03_deterministic_vs_creative.py
```

These scripts explain generation controls:

- `temperature`: controls how sharp or flat the probability distribution becomes.
- `top_k`: keeps only the highest `k` tokens.
- `top_p`: keeps the smallest set of tokens whose cumulative probability reaches `p`.
- `min_p`: keeps tokens whose probability is large enough relative to the top token.

The folder also contains generated visualizations:

- `temperature_effect.png`
- `sampling_filters.png`


### 5. Streaming Demo

Start each server in one terminal, then run the matching client in another terminal.

Normal request/response:

```bash
python streaming_demo/server_normal.py
python streaming_demo/client_normal.py
```

Chunked streaming:

```bash
python streaming_demo/server_chunked.py
python streaming_demo/client_chunked.py
```

Server-Sent Events:

```bash
python streaming_demo/server_sse.py
python streaming_demo/client_sse.py
```

OpenAI streaming:

```bash
python streaming_demo/client_openai_normal.py
python streaming_demo/client_openai_stream.py
```

The `streaming_demo/yields/` folder explains the Python `yield` keyword, which is the core idea behind streaming generators.

### 6. Log Probabilities

```bash
python logprobs/01_raw_logprobs.py
python logprobs/02_confidence_scoring.py
python logprobs/03_classification_free.py
```

These scripts show how token-level log probabilities can be used to inspect model certainty, compute confidence scores, and build lightweight classifiers without training a separate model.

### 7. Structured Outputs

```bash
python structured_outputs/01_naive_approach.py
python structured_outputs/02_json_mode.py
python structured_outputs/03_json_schema.py
```

These examples compare three approaches to extracting structured data:

- asking the model naturally
- asking for JSON mode
- enforcing a strict JSON schema

The JSON schema version is the most reliable because the model response must match the exact shape expected by your program.

### 8. Pretraining, SFT, and RLHF

```bash
python pretraining_sft_rlhf/01_base_model.py
python pretraining_sft_rlhf/02_sft_effect.py
python pretraining_sft_rlhf/03_rlhf_preference.py
```

These scripts demonstrate the conceptual differences between:

- a base model
- a supervised fine-tuned model
- a preference-aligned model

## Local vs API Examples

Run locally:

- `tokenizer.py`
- `vocabulary_printing.py`
- `basic_embedding.py`
- `GPT2_workflow.py`
- `sampling_deep_dive/*`
- `streaming_demo/server_*`
- `streaming_demo/client_normal.py`
- `streaming_demo/client_chunked.py`
- `streaming_demo/client_sse.py`
- `streaming_demo/yields/*`

Require `OPENAI_API_KEY`:

- `token_embeddings_api_call.py`
- `streaming_demo/client_openai_normal.py`
- `streaming_demo/client_openai_stream.py`
- `logprobs/*`
- `structured_outputs/*`
- some scripts in `pretraining_sft_rlhf/`

## Common Issues

If `transformers` or `sentence-transformers` downloads fail, check your internet connection. The first run may take a while because models are downloaded and cached locally.

If matplotlib opens a chart window and the script appears paused, close the chart window to let the script finish.

If an OpenAI script fails with an authentication error, confirm that `.env` exists and contains `OPENAI_API_KEY`.

If a FastAPI server says the port is already in use, stop the previous server process or change the port in the script.

## Notes

This repository is designed for learning, so many scripts print intermediate values rather than hiding details behind functions. Read the output slowly: the printed token IDs, probabilities, embeddings, and chunks are the main lesson.
