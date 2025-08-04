# This code is inspired from: https://github.com/jasonvanf/llama-trl/blob/main/tuning_lm_with_rl.py

import os
# disable annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import torch
torch.manual_seed(42) # set seed on all devices for reproducibility

from dataclasses import dataclass, field
from typing import Optional
import json
from torch.utils.data import DataLoader
from peft import LoraConfig, PrefixTuningConfig, PeftModel
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report, matthews_corrcoef
from trl import GRPOTrainer, GRPOConfig

from utils import *
from logging_utils import save_batch_generations
from rewards import make_reward_function_emo, make_reward_function_cefr, make_reward_function_strategy

@dataclass
class ScriptArguments:
    """
    All the arguments required to launch Proximal Policy Optimization (PPO) training.

    Attributes:
        model_name (Optional[str]):             The name of the model. Default is an empty string.
        reward_model_name (Optional[str]):      The name of the reward model. Default is an empty string.
        dataset_name (Optional[str]):           The name of the dataset to train on (now local files, then batches). Default is an empty string.
        learning_rate (Optional[float]):        The learning rate for training. Default is 1.41e-5.
        max_length (Optional[int]):             The maximum length for input sequences. Default is 512.
        output_max_length (Optional[int]):      The maximum length for generated sequences. Default is 20.
        mini_batch_size (Optional[int]):        The minibatch size for PPO training. Default is 1.
        batch_size (Optional[int]):             The batch size for training. Default is 32.
        ppo_epochs (Optional[int]):             The number of PPO epochs. Default is 4.
        early_stopping (Optional[bool]):        Whether to enable early stopping. Default is False.
        target_kl (Optional[float]):            The KL divergence target for early stopping. Default is 0.1.
        reward_baseline (Optional[float]):      A baseline value subtracted from the reward. Default is 0.0. # NOTE to be deleted.
        output_dir (Optional[str]):             The directory where checkpoints are saved. Default is "./checkpoints/tuning_llama_rl/".
        experiment_name (Optional[str]):        The name to give to the PPO training experiment. 
    """
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the training dataset name"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=50, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    output_dir: Optional[str] = field(default="./checkpoints/ppo_for_llama/", metadata={"help": "n steps to save the model"})
    experiment_name: Optional[str] = field(default="llama_3.1_8b_instruct", metadata={"help": "the name of the experiment"})


class PromptDataset(torch.utils.data.Dataset):
    '''
    Custom dataset for handling prompt data for training.

    Attributes:
        data (DatasetDict): The dataset containing prompt ids and labels.
        length (int): The length of the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns a dictionary with prompt ids and asked labels for a given index.
    '''
    def __init__(self, prompts, labels):
        self.prompts = prompts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        # the column must be named prompt to apply GRPO
        return {'prompt': prompt, 'asked': label}


def build_prompts_dataset(b_size):
    '''
    Build a dataset in a Hugging Face format that contains all the prompts to be used during training.

    Parameters:
        b_size (int): Batch size.

    Returns:
        DataLoader: DataLoader containing the prompt dataset.
    '''
    # concept = 'NextIntention'
    # classes = ['LoadedPositive', 'LoadedNegative', 'LoadedNeutral', 'NonLoadedPositive', 'NonLoadedNegative', 'NonLoadedNeutral']
    concept = 'ProficiencyLevel'
    classes = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    all_asked_labels = []
    all_prompt_texts = [] 

    # get all data
    for _ in range(args['n_batches']*b_size):
        prompt_text, asked_label = random_prompt(concept, classes)
        all_asked_labels.append(asked_label)
        all_prompt_texts.append(prompt_text)
    
    # use dataset class PromptDataset to make a dataset of prompt in their textual format (tokenization is performed afterwards in GRPOTrainer)
    prompt_data = PromptDataset(all_prompt_texts, all_asked_labels)

    return prompt_data

def build_prompt_dataset_for_strategy(args, b_size):
    concepts = {
        "ProficiencyLevel": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "Load": ["Loaded", "NonLoaded"],
        "Polarity": ["Positive", "Negative", "Neutral"]
    }

    concept_value_mapping = {
        'ProficiencyLevel': {
            'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5,
        },
        'Load': {
            'NonLoaded': 0, 'Loaded': 1,
        },
        'Polarity': {
            'Negative': 0, 'Neutral': 1, 'Positive': 2,
        }
    }

    all_asked_labels, all_prompt_texts = [], []

    for _ in range(args['n_batches']*b_size):
        while True:
            selected = {}
            for concept, values in concepts.items():
                if random.random() < 2/3:
                    selected[concept] = random.choice(values)

            if selected:
                break

        prefix = "[" + ", ".join(f"{k}:{v}" for k, v in selected.items()) + "]"
        prompt_text = prompt_for_strategy(prefix)

        labels_dict = {k: concept_value_mapping[k][selected[k]] for k in selected.keys()}

        all_prompt_texts.append(prompt_text)
        all_asked_labels.append(labels_dict)

    prompt_data = PromptDataset(all_prompt_texts, all_asked_labels)

    return prompt_data


if __name__=='__main__':

    # args handling + create required folders if not exist
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    # overall GRPO config
    training_args = GRPOConfig(
        output_dir=script_args.output_dir,
        run_name=script_args.experiment_name,
        learning_rate=5e-4, # 5e-4 seems to be good for CEFR, but too high for emotion
        # adam_beta1 = 0.9,
        # adam_beta2 = 0.99,
        # weight_decay = 0.1,
        # warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        # target_kl=0.02,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=100,
        num_train_epochs=2,
        save_steps=5,
        # max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
    )

    # store some additional args here
    args = {
    'device':       activate_gpu(),
    'gen_kwargs':{
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 1,
        "do_sample": True,
        'repetition_penalty':1.2, # it is for repetition between sentences in the case of batched inference?
        "max_new_tokens":script_args.output_max_length,
        # "num_return_sequences":1
    },
    'experiment': script_args.experiment_name,
    'n_batches':100,
    'n_eps':5,
    }

    # get prompt dataset
    dataset = build_prompt_dataset_for_strategy(args, script_args.batch_size)

    # adapters config
    lora_config = LoraConfig(
        r=16,                                  
        lora_alpha=8,                           
        target_modules=["q_proj", "v_proj"],   
        bias="lora_only",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,  
    )

    # LLM config (CLM architecture for GRPO training)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    # get the appropriate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # set the reward function
    pond_opposite_distance_reward = make_reward_function_strategy(args)

    # train the model using GRPO
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[pond_opposite_distance_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    grpo_trainer.train()