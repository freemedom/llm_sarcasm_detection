# `code/llama_models` Overview

This directory contains LLaMA-based sarcasm detection implementations for multiple reasoning strategies, plus a dedicated ToC (Tree-of-Cues) training pipeline.

## File-by-File Summary

### `llama_io-cot-coc.py`
- **Purpose:** Baseline prompting with three modes: `io`, `cot`, `coc`.
- **Core behavior:** Builds prompts from `Text`, runs Hugging Face `pipeline("text-generation")`, maps outputs to binary labels (`Not Sarcastic` -> `0`, else `1`), and evaluates metrics.
- **Input:** `datasets/test_{task_name}.csv`
- **Output:** `llama_output/{strategy}/output_{strategy}_{task_name}.csv` and metric JSON under the same strategy folder.

### `llama_tot_api.py`
- **Purpose:** Tree-of-Thought style inference for sarcasm classification.
- **Core behavior:** Uses `SmartLLMChain` (LangChain Experimental) with multiple idea generation (`n_ideas=3`), parses final text for label cues, and evaluates.
- **Input:** `datasets/test_{task_name}.csv`
- **Output:** `llama_output/tot/output_tot_{task_name}.csv` + `metric_tot_{task_name}.json`
- **Extra:** Supports chunked/resumable processing with temporary per-chunk CSV files.

### `llama_goc_api.py`
- **Purpose:** Graph-of-Cues (GoC) inference.
- **Core behavior:** For each sample, extracts cue texts, builds a cue graph (`networkx`), iteratively checks cue sufficiency, selects next cue, then predicts sarcasm.
- **Input:** `datasets/test_{task_name}.csv`
- **Output:** `llama_output/goc/output_goc_{task_name}{ablation_type}.csv` + metric JSON.
- **Ablation support:** `_wo_lin`, `_wo_con`, `_wo_emo` control available cue categories.

### `llama_boc_api.py`
- **Purpose:** Bag-of-Cues (BoC) inference with random cue subsets and majority voting.
- **Core behavior:** Samples cue subsets from a cue pool, creates multiple prompts per sample (`num_set=3`), runs LLM generation for each set, and applies majority vote over predicted labels.
- **Input:** `datasets/test_{task_name}.csv`
- **Output:** `llama_output/boc/output_boc_{task_name}{ablation_type}_again.csv` + metric JSON.
- **Dataset-specific path:** Uses a context-aware prompt for `mustard` (`Text` + `Context`).

### `toc_llama_hf.py`
- **Purpose:** Defines the ToC model architecture used for training.
- **Core behavior:** Wraps a frozen LLaMA causal LM, projects multiple cue embeddings via small trainable linear layers, combines cue tensor with prompt embeddings, and predicts at sequence end.
- **Main class:** `ToC_llama(nn.Module)`
- **Used by:** `train_llama_toc_hf_ddp.py`

### `train_llama_toc_hf_ddp.py`
- **Purpose:** Multi-GPU distributed training/evaluation for the ToC LLaMA model.
- **Core behavior:** Uses PyTorch DDP (`torchrun`), `DistributedSampler`, custom `TextDataset`, epoch-wise test export, and metric logging.
- **Input:** `datasets_llama_toc/train_{task_name}_with_toc_cues.csv` and `.../test_{task_name}_with_toc_cues.csv`
- **Output:** `llama_output/toc/output_toc_{task_name}_wo_lin.csv` and per-epoch files/metrics.
- **Notes:** Freezes base LLaMA parameters and trains ToC-specific parts.

## Strategy Mapping

- **Direct prompting/inference scripts:** `llama_io-cot-coc.py`, `llama_tot_api.py`, `llama_goc_api.py`, `llama_boc_api.py`
- **Model definition + training pipeline:** `toc_llama_hf.py` + `train_llama_toc_hf_ddp.py`

## Typical Commands

```bash
# IO / CoT / CoC
python code/llama_models/llama_io-cot-coc.py --task_name iacv2 --strategy coc

# ToT
python code/llama_models/llama_tot_api.py --task_name iacv2 --strategy tot

# GoC
python code/llama_models/llama_goc_api.py --task_name iacv2 --strategy goc

# BoC
python code/llama_models/llama_boc_api.py --task_name iacv2 --strategy boc

# ToC training (multi-GPU)
torchrun --nproc_per_node 6 code/llama_models/train_llama_toc_hf_ddp.py --task_name iacv2
```

## Practical Notes

- Most inference scripts use Hugging Face/LangChain generation and then parse free-form text into binary labels with regex.
- `goc`/`boc` support cue ablation via `--ablation_type`:
  - `_wo_lin` (without linguistic cues)
  - `_wo_con` (without contextual cues)
  - `_wo_emo` (without emotional cues)
- ToC is separated from other strategies because it is a trainable architecture (not only prompt engineering), requires cue-augmented datasets, and is launched with distributed training.
