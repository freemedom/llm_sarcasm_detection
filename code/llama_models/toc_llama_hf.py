"""
prefix prepending

Hugging Face LLaMA module for Tree-of-Cues (ToC) training.

Architecture sketch:
- Load a causal LLaMA (`AutoModelForCausalLM`) and freeze the transformer backbone.
- Three cue streams (see below) are tokenized separately.
- Each stream is embedded with the *same* `embed_tokens`, passed through its own small trainable linear (`dim_reduction_nets`) and padded to a fixed small width.
- The three reduced cue tensors are fused with a broadcast `einsum` into one cue prefix tensor per batch item, then concatenated *before* the main prompt embeddings.
- The model runs forward on `inputs_embeds`; logits at the last real token position are used for next-token style yes/no (or similar) supervision in the trainer.

`ZeroOutputLinear` is an experimental helper (currently commented out) that would zero
the linear output—useful for ablations or debugging cue fusion without cue contribution.

Three cue streams (what this means):
- The model does not merge all evidence into one flat string first. It uses three *parallel*
  token sequences—one per cue family—aligned with `cue_types` (e.g. linguistic, contextual,
  emotional) and the dataset columns that hold each cue text.
- Linguistic: wording, rhetoric, punctuation, style, etc. Contextual: topic, situation,
  background. Emotional: sentiment, contrast, affective cues, etc.
- "Stream" = a full sequence of tokens for that cue type (own `input_ids` / mask), through
  the shared `embed_tokens`, then its own `dim_reduction_nets` branch, until the three
  branches are fused (via `einsum`) into one cue prefix prepended to the main prompt.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict


import random
import numpy as np

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    # LlamaTokenizerFast,
    # LlamaForCausalLM,
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    pipeline
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class ZeroOutputLinear(nn.Linear):
    """Linear layer that discards activations (returns zeros); kept for optional ablation paths."""
    def forward(self, input):
        output = super().forward(input)
        return torch.zeros_like(output)
    
class ToC_llama(nn.Module):    
    """Frozen LLaMA backbone + trainable cue projection/fusion in front of the main prompt."""
    def __init__(self,
                 model_id: str = 'meta-llama/Meta-Llama-3-8B',
                 cache_dir: str = 'llama3-8b-hf/original',
                 cue_types: list = ["linguistic", "contextual", "emotional"],
                 max_cue_len: int = 32,
                ):
        super().__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = cache_dir,token = None)
        self.llama = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = cache_dir, token = None)

        # Train only cue-side adapters; keep pretrained LLaMA weights fixed.
        # Parameter-efficient training: disable gradients on the pretrained transformer trunk.
        # `self.llama.model` is HuggingFace's inner module (decoder blocks + the embedding layer
        # the LM uses internally). Setting `requires_grad = False` keeps these weights constant
        # during backprop—only modules that remain True (e.g. `dim_reduction_nets` below) are
        # updated. This reduces memory, avoids catastrophic forgetting on the base LLM, and
        # matches the ToC design (learn cue fusion, not full finetune). Output head (`lm_head`)
        # is not touched here; the training script may freeze it separately.
        for param in self.llama.model.parameters():
            param.requires_grad = False

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cue_types = cue_types

        # Optional: replace one linear with ZeroOutputLinear to silence a cue branch.
        #zero_output_linear = ZeroOutputLinear(self.llama.config.hidden_size, 2**4-1)
        #for param in zero_output_linear.parameters():
        #    param.requires_grad = False

        # One small projection per cue type: hidden_size -> (2^4 - 1) features (+1 dim after F.pad below).
        # One small projection per cue type. Output dim is 2**4 - 1 (= 15) by design so that after
        # `F.pad(..., (0, 1), value=1)` each branch has width 16 = 2**4. That pairs with the einsum
        # fusion below; the exact 15/16 choice is a structural hyperparameter (not fixed by the task).
        self.dim_reduction_nets = nn.ModuleList([
                                        #zero_output_linear,
                                        nn.Linear(self.llama.config.hidden_size, 2**4-1, bias=True),
                                        nn.Linear(self.llama.config.hidden_size, 2**4-1, bias=True),
                                        nn.Linear(self.llama.config.hidden_size, 2**4-1, bias=True)
                                        ])
        
    def forward(self, 
                cue_ids: torch.Tensor, 
                cue_masks: torch.Tensor, 
                prompt_ids: torch.Tensor, 
                prompt_masks: torch.Tensor,
                # max_cue_len: int = 64,
                ):
                #print('get the 3 cue indices')
        # Fuse cue embeddings with prompt embeddings, then run the frozen LM on the joint sequence.
        concatenated_input, input_mask = self.tensor_of_cues(prompt_ids,prompt_masks, cue_ids, cue_masks)

    

        # `input_ids` are tokenizer-produced integer token indices (shape like (B, T)),
        # e.g. text tokens ["I"," love"," this"] -> ids [40, 3021, 445] (illustrative).
        # Here we bypass `input_ids` and pass `inputs_embeds` directly because we prepend
        # a learned cue prefix before prompt embeddings. The Transformer/LM head path stays the same.
        result = self.llama.forward(inputs_embeds=concatenated_input,
                                    attention_mask=input_mask,
                                    return_dict = True)


        logits = result['logits']

        # Take logits at the last non-padding position (standard LM head index for "next token" targets).
        indices = torch.arange(input_mask.size(1), device=input_mask.device)

        masked_indices = input_mask * indices
        max_indices = masked_indices.argmax(dim=1)
        logits = logits[torch.arange(max_indices.size(0)), max_indices]


        # generated_probs = F.softmax(logits, dim=-1)

        pred = torch.argmax(logits, dim=-1)
        
        return logits, pred
    

    
    def tensor_of_cues(self, prompt_ids, prompt_masks, output_cue_ids, cue_masks):
        """Build [cue_prefix | prompt] embedding sequence and matching attention mask."""

        # Prefix attention mask derived from cue padding masks (layout follows the training batch).
        binary_mask = cue_masks.sum(dim=1) > 0

        # output_cue_ids: (batch, num_cue_types, seq); embed each cue with shared token embeddings. (B, 3, L, H)
        output_cue_embeddings = self.llama.model.embed_tokens(output_cue_ids)

        # (num_cue_types, batch, seq, hidden) so we can zip with `dim_reduction_nets`. (3, B, L, H)
        output_cue_embeddings = output_cue_embeddings.transpose(0,1)
        
        
        
        
        # Project each cue sequence to a low-rank tensor; pad an extra dimension (value 1) for einsum fusion.
        # Project each cue sequence, then append a *fixed* trailing dimension of 1 (not learned).
        # Rationale: the three branches are combined with a multiplicative einsum (outer-product-like).
        # Without a constant 1 channel, products only encode "full" three-way interactions among learned features; the padded 1 acts like a homogeneous-coordinate trick so products can also represent lower-order / separable terms when some factors take the constant slot.
        # Using pad(value=1) differs from `Linear(..., 16)`: the 16th entry is always 1, not a free weight row. Removing this pad requires revisiting the einsum layout.
        # for Traverse the first dimension; The kth way uses the kth dim_reduction_nets[k]
        
        
        # Per cue stream: `net(cue_embed)` maps (B, L, H) -> (B, L, 15). Then F.pad(..., (0, 1), value=1)
        # pads only the *last* dimension: 0 on the left, 1 new slot on the right, filled with constant 1.
        # Toy example on the feature dimension (same idea per row/token):
        #   [[a1,a2,a3], [b1,b2,b3]]  ->  [[a1,a2,a3,1], [b1,b2,b3,1]]
        # So each token's feature vector grows from 15 to 16; the new coordinate is always 1 (not learned).
        # Why: the three streams are fused with a multiplicative einsum; a fixed 1 channel behaves like a
        # homogeneous-coordinate trick so products can include lower-order / separable terms when one
        # factor uses the constant slot. Using `Linear(..., 16)` instead would make the 16th dimension
        # learned weights, not a fixed bias channel—different semantics. Changing this requires revisiting einsum.
        # zip pairs stream k with `dim_reduction_nets[k]` after transpose(0,1) made the leading dim cue-type.
        cue_tensors = [F.pad(net(cue_embed),(0,1), value=1) for net, cue_embed in zip(self.dim_reduction_nets, output_cue_embeddings)]

        # Dynamic einsum: outer-style product across the three cue tensors, broadcast over batch and length.
        # Naming: b=batch, i/j/k/... index cue feature axes after projection.
        start_char = 'i'
        input_elements = []
        output_expr = 'bi'
        current_char = start_char
        for _ in range(len(cue_tensors)):
            next_char = chr(ord(current_char) + 1)
            input_elements.append(f'b{start_char}{next_char}')
            output_expr = output_expr + next_char
            current_char = next_char

        input_expr = ','.join(input_elements)
        einsum_expr = f'{input_expr}->{output_expr}'

        cue_tensor = torch.einsum(einsum_expr, *cue_tensors).flatten(start_dim=-len(cue_tensors), end_dim=-1) # "bij,bjk,bkl->bijkl"
        #cue_tensor = self.scale(cue_tensor)

        prompt_embeddings = self.llama.model.embed_tokens(prompt_ids) # L: The number of tokens after this cue is truncated/completed by the tokenizer (max_length=max_cue_len), that is, a text cue corresponds to an id sequence of length L.
        # B=batch，L=max_cue_len，H=hidden_size

        # Cue prefix length equals flattened fused feature width; then append prompt tokens.
        
        # Prefix-prepending intentionally changes the model's output distribution:
        # the cue tensor acts as a learned conditioning prefix placed before prompt embeddings.
        # This is expected behavior (prefix-tuning style conditioning), not model corruption.
        # When trained well, the prefix improves task guidance; if too strong, it can dominate
        # the original prompt semantics and hurt stability.
        # Shape after concatenation: (B, L + S, H), where L is cue-prefix length and S is prompt length.
        concatenated_input = torch.cat([cue_tensor, prompt_embeddings],dim=1) # concatenated_input: (B, L + S, H)
        input_mask = torch.cat([binary_mask,prompt_masks], dim = 1)
        return concatenated_input, input_mask
    
    
