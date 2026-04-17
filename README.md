# KVComm

Official implementation of the paper [KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](https://openreview.net/forum?id=F7rUng23nw) (ICLR 2026).

A framework for communicating between Large Language Models (LLMs), focusing on how models can effectively share information to improve collaborative reasoning and question-answering performance.

## Installation

```bash
pip install -r requirements.txt
```

Note: Requires `transformers==4.53.3` specifically.

## Datasets

| Dataset           | Task Type             | Description               | Data Path                         |
|-------------------|-----------------------|---------------------------|-----------------------------------|
| `hotpotqa`        | Multi-hop QA          | Wikipedia-based reasoning | HuggingFace                       |
| `qasper`          | Scientific QA         | Paper-based questions     | HuggingFace                       |
| `musique`         | Multi-hop QA          | Compositional reasoning   | HuggingFace                       |
| `multifieldqa_en` | Multi-domain QA       | Cross-field knowledge     | HuggingFace                       |
| `twowikimqa`      | Multi-hop QA          | Wikipedia bridge entities | HuggingFace                       |
| `tipsheets`       | Custom QA             | Synthetic reasoning tasks | `dataloader/data/tipsheets.jsonl` |
| `countries`       | Geographic QA         | Country-based questions   | `dataloader/data/countries.jsonl` |
| `tmath`           | Mathematical          | Math problem solving      | `dataloader/data/TMATH`           |

## Quick Start

### Baseline Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_baseline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### Skyline Test
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_skyline \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct
```

### KVComm Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --top_layers 0.3
```

### Activation Communication
```bash
python com.py \
    --test_task tipsheets \
    --do_test_ac \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --layer_k 26 \
    --layer_j 26 \
    --f replace
```

### Natural Language Debate
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_nld \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

### CIPHER Communication
```bash
python com.py \
    --test_task hotpotqa \
    --do_test_cipher \
    --model_A meta-llama/Llama-3.1-8B-Instruct \
    --model_B meta-llama/Llama-3.1-8B-Instruct \
    --nld_max_tokens_model_A_and_B_phase1 256 \
    --sender_aware
```

## Communication Methods

### 1. KVComm (Cross-View Communication)
- **Mechanism**: Shares key-value cache from model A's specified layers to model B
- **Parameters**: `--layers_list`, `--layer_from`, `--layer_to`, `--top_layers`
- **Use Case**: Efficient information transfer with minimal computational overhead

### 2. Activation Communication (AC)
- **Mechanism**: Injects hidden activations from model A into model B at specific layers
- **Parameters**: `--layer_k` (source), `--layer_j` (target), `--f` (fusion method)
- **Fusion Methods**: `replace`, `sum`, `mean`

### 3. Natural Language Debate (NLD)
- **Mechanism**: Models exchange natural language responses and refine answers
- **Parameters**: `--nld_max_tokens_model_A_and_B_phase1`, `--sender_aware`
- **Process**: Initial responses → Exchange → Refinement

### 4. CIPHER Communication
- **Mechanism**: Models communicate through learned embedding representations
- **Features**: Temperature-controlled generation, nearest neighbor decoding

## Configuration Options

### Model Configuration
- `--model_A`, `--model_B`: Hugging Face model identifiers
- `--device`: CUDA device (default: `cuda:0`)
- `--max_input_length`: Maximum input token length (default: 64000)

### Communication Parameters
- `--layers_list`: Specific layers for KVComm communication
- `--top_layers`: Percentage of top-importance layers to use
- `--layer_k`, `--layer_j`: Source and target layers for AC
- `--f`: Fusion function for AC (`replace`, `sum`, `mean`)

### Evaluation Settings
- `--test_task`: Dataset to evaluate on
- `--limit`: Limit number of evaluation examples
- `--calib_size`: Calibration set size for layer importance

### Experiment Tracking
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity
- `--run_name`: Custom experiment name

## Layer Importance Analysis

The framework includes automatic layer importance detection:

```bash
python com.py \
    --test_task hotpotqa \
    --do_test \
    --top_layers 0.3
```

This automatically identifies which layers are most important for communication and selects them for the main evaluation.

### Reproducibility note

The repository does not currently include the code used for FLOPs and memory measurement. Clarification on the exact measurement setup (e.g., execution path and whether lm_head is included) would be helpful for reproducing the reported results.
