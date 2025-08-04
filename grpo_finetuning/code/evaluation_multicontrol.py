import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import datasets
from termcolor import colored
import json
import os
# from accelerate import dispatch_model, infer_auto_device_map
# disable annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings('ignore', message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token.As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, matthews_corrcoef, mean_absolute_error, mean_squared_error

from utils import *
from logging_utils import *
from grpo_exps import prompt_for_strategy, build_prompt_dataset_for_strategy
from rewards import get_trues_and_preds_strategy, polarity_profile_label_mapping


def evaluate(args, prompt_dataset, model):
    cefr_asked, cefr_got = [], []
    load_asked, load_got = [], []
    polarity_asked, polarity_got = [], []
    tokenizer = args['tokenizer']

    for item in tqdm(prompt_dataset):
        # generate from the prompt
        prompt, asked_labels = item['prompt'], item['asked']
        input_tokens = tokenizer([prompt], return_tensors="pt", padding=True).to(args['device'])
        output_tokens = model.generate(**input_tokens, max_new_tokens=args['max_new_tokens'], pad_token_id=args['tokenizer'].eos_token_id)
        completions = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        # remove the prompt from the generated content
        completions[0] = completions[0][len(prompt):].strip()
        print("COMPLETIONS:", completions)
        asked, proficiency_levels, polarity_profiles = get_trues_and_preds_strategy(completions, [asked_labels], args)
        load_got_iter, polarity_got_iter = polarity_profile_label_mapping[polarity_profiles[0]]

        # deduce trues and preds for all 3 criteria: cefr, load, polarity
        # we consider got labels only when there is a required (asked) label to compare to. Otherwise, we don't store it to compute performance metrics
        if "ProficiencyLevel" in asked_labels.keys():
            cefr_asked.append(asked_labels['ProficiencyLevel'])
            cefr_got.append(proficiency_levels[0])

        if "Load" in asked_labels.keys():
            load_asked.append(asked_labels['Load'])
            load_got.append(load_got_iter)

        if "Polarity" in asked_labels.keys():
            polarity_asked.append(asked_labels['Polarity'])
            polarity_got.append(polarity_got_iter)

    # compute classification metrics out of the trues/preds lists
    metrics = {
        'cefr': {
            'acc': accuracy_score(cefr_asked, cefr_got),
            'prec': precision_score(cefr_asked, cefr_got, average='weighted'),
            'rec': recall_score(cefr_asked, cefr_got, average='weighted'),
            'f1':f1_score(cefr_asked, cefr_got, average='weighted'),
            'mcc':matthews_corrcoef(cefr_asked, cefr_got),
            'mae':mean_absolute_error(cefr_asked, cefr_got),
            'mse':mean_squared_error(cefr_asked, cefr_got)
        },
        'load': {
            'acc': accuracy_score(load_asked, load_got),
            'prec': precision_score(load_asked, load_got, average='weighted'),
            'rec': recall_score(load_asked, load_got, average='weighted'),
            'f1':f1_score(load_asked, load_got, average='weighted'),
            'mcc':matthews_corrcoef(load_asked, load_got),
            'mae':mean_absolute_error(load_asked, load_got),
            'mse':mean_squared_error(load_asked, load_got)
        },
        'polarity': {
            'acc': accuracy_score(polarity_asked, polarity_got),
            'prec': precision_score(polarity_asked, polarity_got, average='weighted'),
            'rec': recall_score(polarity_asked, polarity_got, average='weighted'),
            'f1':f1_score(polarity_asked, polarity_got, average='weighted'),
            'mcc':matthews_corrcoef(polarity_asked, polarity_got),
            'mae':mean_absolute_error(polarity_asked, polarity_got),
            'mse':mean_squared_error(polarity_asked, polarity_got)
        }
    }

    # dump into json
    with open(f"../metrics/{args['experiment']}.json", 'w') as f:
        json.dump(metrics, f)

    # return displayable versions of the metrics
    print("CEFR LEVEL EVALUATION")
    print(classification_report(cefr_asked, cefr_got), confusion_matrix(cefr_asked, cefr_got))

    print("\nLOAD EVALUATION")
    print(classification_report(load_asked, load_got), confusion_matrix(load_asked, load_got))

    print("\nPOLARITY EVALUATION")
    print(classification_report(polarity_asked, polarity_got), confusion_matrix(polarity_asked, polarity_got))

if __name__ == "__main__":

    args = {
    'device': activate_gpu(),
    'eval_iters': 100,
    'n_batches': 32,
    'max_new_tokens':50,
    'experiment': 'qwen_emo_grpo_eval',
    }

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter_model_name = "./checkpoints/gpro_multicontrol_qwen/checkpoint-200" # put the right checkpoint here

    dataset = build_prompt_dataset_for_strategy(args, b_size=args['eval_iters'])

    clm_model = AutoModelForCausalLM.from_pretrained(  
        model_name,
        low_cpu_mem_usage=True,         # recommanded param
        trust_remote_code=True,
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map='auto',      # send to the right device
    )

    # PLUG THE ADAPTERS
    clm_model = PeftModel.from_pretrained(clm_model, adapter_model_name)

    args.update({
        'model': clm_model,
        'tokenizer': AutoTokenizer.from_pretrained(model_name, padding_side='left'),
    })

    args['tokenizer'].pad_token_id = args['tokenizer'].eos_token_id
    evaluate(args, dataset, clm_model)




