"""
LLaMA Tree-of-Thought (ToT) style sarcasm classification via LangChain.

Pipeline:
1) Load `test_{task_name}.csv` from `--dataset_path`.
2) For each row, run `SmartLLMChain` with `n_ideas=3` (multiple internal reasoning paths).
3) Post-process model text: optionally strip a `**Conclusion**:` block, then map output to
   binary labels (0 if "not sarcastic", else 1).
4) Process in chunks with on-disk checkpoints for resume; merge, delete chunk files, evaluate.

Note: `HuggingFaceEndpoint` targets Meta-Llama-3-8B-Instruct; the CLI description string
below still mentions GPT-4o historically.
"""

import argparse
from langchain_experimental.smart_llm import SmartLLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import json
import re
import csv
import os
from sklearn import metrics
import logging
import numpy as np


from langchain_huggingface import HuggingFaceEndpoint



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_pipeline():
    # Remote LLaMA endpoint for text generation (LangChain wrapper).
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    # max_new_tokens=100,
    do_sample=False,
    )
    return llm

def generate_ToT_prompt(llm, data_point):
  # SmartLLMChain expands the task into multiple candidate ideas (n_ideas=3) before synthesis.
  # The input text is embedded in the template; chain.run({}) executes the full ToT-style chain.
  hard_question =f'''I am a sarcasm classification classifier. The task is to assign a correct label from ['Not Sarcastic', 'Sarcastic'] for the input text: {data_point}.'''
  prompt = PromptTemplate.from_template(hard_question)

  chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=False)

  return (chain.run({}))


def eval_performance(y_true, y_pred, metric_path=None):
    # Standard binary classification metrics; optionally persist summary JSON.
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
    # Entry point: chunked inference + evaluation. `--token` is parsed but not passed to the endpoint here.
    parser = argparse.ArgumentParser(description='Running Tree-of-Thoughts based on GPT-4o for sarcasm detection.')
    # parser.add_argument('--dataset_name', metavar='D', type=str, help='dataset name', default='iacv2')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--task_name', metavar='T', type=str, help='task name', default='iacv2')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='llama_output')
    parser.add_argument('--metric_path', metavar='P', type=str, help='metrics path', default='llama_output')
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--token', metavar='K', type=str, help='token', default=None)
    parser.add_argument('--strategy', metavar='S', type=str, help='prompting strategy', default='tot')

    args = parser.parse_args()
    task_name = args.task_name
    strategy = args.strategy
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/{strategy}/output_{strategy}_{task_name}.csv' #f'output_toc_new/output_toc_'+ task_name +'.csv'# +'_wo_emo2.csv'
    metric_path = f'{args.metric_path}/{strategy}/metric_{strategy}_{task_name}.json' #f'output_toc_new/metric_toc_'+ task_name +'.json'# +'_wo_emo2.json'
    token = args.token
    chunks = args.chunks

    llm = configure_pipeline()

    df = pd.read_csv(dataset_path)

    # Log message legacy naming: this step runs ToT generation per row, not cue extraction.
    logger.info('generating cues...')


    # Split dataframe into chunks; each chunk can be resumed from `chunk_file_path` if present.
    chunk_size = int(np.ceil(len(df) / args.chunks))
    df_chunks = []
    for chunk_num in range(args.chunks):
        logger.info('processing chunk {}...'.format(chunk_num))
        chunk_file_path = output_path.replace('.csv',f'_{chunk_num}.csv')
        if os.path.exists(chunk_file_path):
            df_chunk = pd.read_csv(chunk_file_path)
            df_chunks.append(df_chunk)
            continue

        df_chunk = df[chunk_num*chunk_size:min(len(df), (chunk_num+1)*chunk_size)]
        output_texts = []
        labels = []
        for i, row in df_chunk.iterrows():
            result = generate_ToT_prompt(llm, row['Text'])
            result = result.lower().strip()
            print("***********************************************")
            # print("result:",result)
            # If the chain returns a structured answer, keep only the conclusion paragraph.
            match = re.search(r"(?i)\*\*conclusion\*\*:\n(.*)", result, re.DOTALL)
            if match:
                result = match.group(1).strip()
                # print(result)
            else:
                result = result     
            print("***********************************************")
            output_texts.append(result)

            # Heuristic binary mapping aligned with other scripts in this repo.
            if re.search(r"\bnot sarcastic\b", result, re.IGNORECASE):
                labels.append(0)
            else:
                labels.append(1)
            print("----------------")
        df_chunk['llm_output'] = output_texts
        df_chunk['pred']= labels
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)

    logger.info("Evaluation....")
    # Reassemble full predictions, remove intermediate chunk CSVs, then score against `Label`.
    df = pd.concat(df_chunks)
    df.to_csv(output_path, index=0)
    for i in range(args.chunks):
        chunk_file_path = output_path.replace('.csv',f'_{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
    eval_performance(df['Label'], df['pred'], metric_path)
   