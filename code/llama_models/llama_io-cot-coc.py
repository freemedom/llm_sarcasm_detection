"""
LLaMA sarcasm inference script for three prompting strategies: IO, CoT, and CoC.

Role in the project
-------------------
This file covers the "simple prompting" baselines used in the SarcasmCue paper /
repo for Meta-Llama-3-8B-Instruct. Unlike GoC/BoC (cue pools + multi-pass voting)
or ToC (trainable cue fusion), this script is single-pass prompt → generate →
parse label. It is the entry point for comparing:
  - IO  (Input-Output / standard prompting)
  - CoT (Chain-of-Thought)
  - CoC (Chain of Contradiction; paper method that contrasts surface sentiment
        vs. true intention)

High-level pipeline
-------------------
1) Load a test split from `{dataset_path}/test_{task_name}.csv`
   (default: `datasets/test_{task_name}.csv`). Expected columns include at least
   `Text` (input utterance) and `Label` (0/1 ground truth).
2) For each row, build a chat-style user message whose content is a strategy-
   specific instruction string (`generate_IO_prompt` / `generate_CoT_prompt` /
   `generate_CoC_prompt`). Prompts are stored as
   `[{'role': 'user', 'content': ...}]` so the HF chat template can be applied.
3) Load `meta-llama/Meta-Llama-3-8B-Instruct` locally via
   `AutoModelForCausalLM` + `pipeline("text-generation")` (`configure_pipeline`),
   with `device_map="auto"` and float16.
4) Run batched generation over the Dataset column `"prompt"`
   (`batch_size=64`, sampling: `temperature=0.6`, `top_p=0.9`,
   `max_new_tokens=256`). Stop on EOS / `<|eot_id|>`.
5) Parse the last assistant turn's free-form text into a binary prediction:
   if the output matches the word boundary pattern `not sarcastic` (case-
   insensitive) → label 0; otherwise → label 1.
6) Write per-sample outputs and evaluate against `Label`
   (Precision / Recall / Accuracy / F1 variants / ROC-AUC / confusion matrix).

Prompt strategy details
-----------------------
- IO:  Direct classification; ask only for the label
       ['Not Sarcastic', 'Sarcastic'] with no intermediate reasoning.
- CoT: Ask the model to "think step by step" before producing a label
       (open-ended reasoning; no fixed sarcasm-specific steps).
- CoC: Paper-style Chain of Contradiction. The model may answer directly if
       confident; otherwise it follows three fixed steps:
         (1) surface sentiment cues,
         (2) true intention (rhetoric / style / etc.),
         (3) compare (1) vs (2) to decide sarcasm.
       Note: this differs from GoC/BoC, which enumerate a larger cue pool.

CLI arguments
-------------
  --task_name     Dataset key used in filenames (default: iacv2).
                  Typical values: iacv1, iacv2, semeval, mustard.
  --dataset_path  Directory containing `test_{task_name}.csv` (default: datasets).
  --output_path   Root for prediction CSVs (default: llama_output).
  --metric_path   Root for metric JSONs (default: llama_output).
  --strategy      One of: io | cot | coc (default: coc).

Outputs
-------
  {output_path}/{strategy}/output_{strategy}_{task_name}.csv
      Adds columns `prompt`, `llm_output` (raw generation), `pred` (0/1).
  {metric_path}/{strategy}/metric_{strategy}_{task_name}.json
      Aggregate classification metrics from `eval_performance`.

Example
-------
  python code/llama_models/llama_io-cot-coc.py --task_name iacv2 --strategy coc

Caveats
-------
- Label parsing is regex-based and biased toward "sarcastic" (1) whenever the
  exact phrase "not sarcastic" is absent; noisy generations can flip labels.
- Sampling is on (`do_sample=True`), so runs are not fully deterministic.
- `get_random_cues` is unused here (leftover from cue-based siblings).
- Model weights are expected under `cache_dir='/root/autodl-tmp/llama/original'`
  (data disk; system disk is too small for 8B weights). Hugging Face access for
  Llama-3 Instruct may be required.



old comments:

Expected label mapping:
- "not sarcastic" -> 0
- otherwise       -> 1


High-level pipeline:
1) Load a test split from `datasets/test_{task_name}.csv`.
2) Build a strategy-specific prompt per sample (`io`, `cot`, or `coc`).
3) Run batched generation with Hugging Face text-generation pipeline.
4) Convert free-form outputs into binary labels via regex matching.
5) Save predictions and compute classification metrics.

"""

import pandas as pd
import json
import re
import os
import random
from sklearn import metrics
from collections import Counter
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    # AutoTokenizer, 
    pipeline,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def configure_pipeline():
    # Initialize local LLaMA-Instruct model and tokenizer for generation.
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp/llama/original')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp/llama/original')

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id

    return pipe

def generate_IO_prompt(data_point):
        # create prompts from the loaded dataset and tokenize them
    # IO: direct classification prompt (no explicit intermediate reasoning).
    if data_point:
        return f"""
        ### Instruction:
        '''You are a sarcasm classification classifier. Assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic']. Only return the label without any other texts.'''

        ### Input:
        {data_point}

        ### Response:

        """


def generate_CoT_prompt(data_point):
        # create prompts from the loaded dataset and tokenize them
    # CoT: asks the model to reason step-by-step before labeling.
    if data_point:
        return f"""
        ### Instruction:
        '''You are a sarcasm classification classifier. Use Chain of Thought approach to assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].'''

        ### Input:
        {data_point}
        Let's think step by step.

        ### Response:

        ### Label: 

        """

def generate_CoC_prompt(data_point):
        # create prompts from the loaded dataset and tokenize them
    # CoC: confidence-aware prompting; direct answer if confident, otherwise reason.
    # Chain of Contradiction
    if data_point:
        return f"""
        ### Instruction:
        '''You are a sarcasm classification classifier. Assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].'''

        ### Input:
        {data_point}
        You can choose to output the result directly if you believe your judgment is reliable,
        or
        You think step by step if your confidence in your judgment is less than 90%:
        Step 1: What is the SURFACE sentiment, as indicated by clues such as keywords, sentimental phrases, emojis?
        Step 2: Deduce what the sentence really means, namely the TRUE intention, by carefully checking any rhetorical devices, language style, etc.
        Step 3: Compare and analysis Step 1 and Step 2, infer the final sarcasm label.

        ### Response:

        ### Label: 

        """
    
def get_random_cues(cue_pool, n):
    return random.sample(list(cue_pool), n)


def eval_performance(y_true, y_pred, metric_path=None):
    # Compute and print standard binary classification metrics, then optionally dump JSON.

    # Precision
    metric_dict = {}
    precision = metrics.precision_score(y_true, y_pred)
    print("Precision:\n\t", precision)
    metric_dict['Precision'] = precision

    # Recall
    recall = metrics.recall_score(y_true, y_pred)
    print("Recall:\n\t",  recall)
    metric_dict['Recall'] = recall

    # Accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy:\n\t", accuracy)
    metric_dict['Accuracy'] = accuracy

    print("-------------------F1, Micro-F1, Macro-F1, Weighted-F1..-------------------------")
    print("-------------------**********************************-------------------------")

    # F1 Score
    f1 = metrics.f1_score(y_true, y_pred)
    print("F1 Score:\n\t", f1)
    metric_dict['F1'] = f1


    # Micro-F1 Score
    micro_f1 =  metrics.f1_score(y_true, y_pred, average='micro')
    print("Micro-F1 Score:\n\t",micro_f1)
    metric_dict['Micro-F1'] = micro_f1


    # Macro-F1 Score
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("Macro-F1 Score:\n\t", macro_f1)
    metric_dict['Macro-F1'] = macro_f1

    # Weighted-F1 Score
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    print("Weighted-F1 Score:\n\t", weighted_f1)
    metric_dict['Weighted-F1'] = weighted_f1


    print("------------------**********************************-------------------------")
    print("-------------------**********************************-------------------------")


    # ROC AUC Score
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        print("ROC AUC:\n\t", roc_auc) 
    except:
        print('Only one class present in y_true. ROC AUC score is not defined in that case.')
        metric_dict['ROC-AUC'] = 0

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))  

    if metric_path is not None:
       json.dump(metric_dict,open(metric_path,'w'),indent=4)

if __name__ == '__main__':
    # CLI controls dataset naming and strategy selection.
    parser = argparse.ArgumentParser(description='Running io,cot or coc based on llama for sarcasm detection.')
    # parser.add_argument('--dataset_name', metavar='D', type=str, help='dataset name', default='iacv2')
    parser.add_argument('--task_name', metavar='T', type=str, help='task name', default='iacv2')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='llama_output')
    parser.add_argument('--metric_path', metavar='M', type=str, help='metrics path', default='llama_output')
    parser.add_argument('--strategy', metavar='S', type=str, help='prompting strategy', default='coc')

    args = parser.parse_args()
    task_name = args.task_name
    strategy = args.strategy
    # Build canonical input/output paths from the task/strategy convention used in this repo.
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/{strategy}/output_{strategy}_{task_name}.csv' #f'output_toc_new/output_toc_'+ task_name +'.csv'# +'_wo_emo2.csv'
    metric_path = f'{args.metric_path}/{strategy}/metric_{strategy}_{task_name}.json' #f'output_toc_new/metric_toc_'+ task_name +'.json'# +'_wo_emo2.json'
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metric_path) or ".", exist_ok=True)
    # Legacy variable: retained from chunk-based scripts, but not used in this file.
    # chunks = args.chunks
    pipe = configure_pipeline()

    # Load test data and materialize prompts according to the selected strategy.
    df = pd.read_csv(dataset_path)
    

    if strategy == 'io':
        df['prompt'] = df.apply(lambda row: [{'role': 'user','content':generate_IO_prompt(row['Text'])}], axis=1)
    elif strategy == 'cot':
        df['prompt'] = df.apply(lambda row: [{'role': 'user','content':generate_CoT_prompt(row['Text'])}], axis=1)
    elif strategy == 'coc':
        df['prompt'] = df.apply(lambda row: [{'role': 'user','content':generate_CoC_prompt(row['Text'])}], axis=1)

    else:
        print('Wrong strategy.')
        exit(1)

    dataset = Dataset.from_pandas(df)

    # Stop generation on EOS or chat end-of-turn token.
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # for i in range(num_set):
    output_texts = []
    labels = []
    
    # Batch inference over prompts; parse textual outputs into binary predictions.
    for out in tqdm(pipe(KeyDataset(dataset, "prompt"),  
                         batch_size=64, 
                         do_sample=True,
                         temperature=0.6,
                         top_p=0.9,
                         max_new_tokens=256,
                         eos_token_id=terminators,
                         pad_token_id=pipe.tokenizer.eos_token_id),total=len(dataset)): 

        result = out[0]['generated_text'][-1]['content']
        result = result.lower().strip()
        if re.search(r"\bnot sarcastic\b", result, re.IGNORECASE):
            labels.append(0)
        else:
            labels.append(1)
        output_texts.append(result)
        
    df['llm_output'] = output_texts
    df['pred'] = labels
    # Persist per-sample outputs and run final evaluation against ground truth labels.
    df.to_csv(output_path, index=0)
    print("Evaluation....")
    eval_performance(df['Label'], df['pred'], metric_path)
