# This script gathers all auxiliary functions related to saving stuff and logging info from training/finetuning

import os
from owlready2 import *
from termcolor import colored
from pprint import pprint
from datetime import datetime
import torch
import numpy as np
import logging
from datasets import Dataset, DatasetDict
from collections import Counter
import multiprocess as mp
import threading
import json
import re
import csv
from numpy import random as rd
import random
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# set default tensor type
torch.set_default_dtype(torch.float16)

from utils import get_datetime


# -----------------------------------------------------------------------------------------
# Data management
# -----------------------------------------------------------------------------------------

def load_dataset(dataset_name):
    '''
    Load a dataset save in the huggingface format, located in the data folder.

    Parameters:
        dataset_name (str): the name of the dataset, i.e. the associated folder inside the data folder. This string should correspond to an existing dataset in the folder, otherwise an exception will be raised.

    Returns:
        dataset (DatasetDict): the dataset as a DatasetDict object.
    '''
    path = f"./data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"../data/{dataset_name}")
    else: 
        raise Exception("No folder with the such name found in the data folder.")

    return dataset


def save_dataset(dataset, dataset_name, output_format='huggingface'):
    '''
    Save the dataset into a HuggingFace format using the save_to_disk method.

    Parameters:
        dataset (DatasetDict):      the dataset to save in a DatasetDict format
        dataset_name (str):         the name to give to the file
        output_format (str):        either 'huggingface' or 'json'. Default is 'huggingface'.
    '''
    if output_format == "huggingface":
        dataset.save_to_disk(f'../data/{dataset_name}')
    elif output_format == "json":
        with open(f"data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")

# -----------------------------------------------------------------------------------------
# Ontology integration management
# -----------------------------------------------------------------------------------------

def remove_quotes(s):
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return repr(s)

def save_batch_generations(batch_labels, batch_generations, batch_index, experiment):
    '''
    This function saves the batch generations in a TSV file for further processing. Each row in `batch_generations` is written to the file,
    with newline characters replaced by '\\n'.

    Parameters:
        batch_generations (list of str): The list of generated data rows to be saved.
        batch_index (int): The index of the current batch.
        experiment (str): The name of the experiment.
    '''
    # label_mapping = {0:'LoadedPositive', 1:'LoadedNegative', 2:'LoadedNeutral', 3:'NonLoadedPositive', 4:'NonLoadedNegative', 5:'NonLoadedNeutral'}
    label_mapping = {0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'C1', 5:'C2'}
    os.makedirs('../objects', exist_ok=True)
    file_path = f'../objects/batch_generations_{batch_index}_{experiment}.tsv'
    with open(file_path, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        for i in range(len(batch_generations)):
            row, label = batch_generations[i], batch_labels[i]
            row = row.replace('\n', '\\n')
            output = f'Asked: {label_mapping[label]} | {row}'
            tsv_writer.writerow([output])

    return file_path[3:] # remove ../ to make it relative


def save_batch_generations_strategy(batch_labels, batch_generations, batch_index, experiment):
    '''
        Here, batch_labels contains, for each generated sample, a list of labels such as [A1, Loaded] or [C2, NonLoaded, Positive]
    '''
    print("BATCH LABELS:", batch_labels)
    os.makedirs('../objects', exist_ok=True)
    file_path = f'../objects/batch_generations_{batch_index}_{experiment}.tsv'
    with open(file_path, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        for i in range(len(batch_generations)):
            row, label_list = batch_generations[i], batch_labels[i]
            row = row.replace('\n', '\\n')
            output = f"Asked: {label_list} | {row}"
            tsv_writer.writerow([output])

    return file_path[3:]

# -----------------------------------------------------------------------------------------
# Fine-tuning results logging utils
# -----------------------------------------------------------------------------------------

def store_split_generations(split, file_paths, trues, experiment):
    '''
    Concatenates generated sentences from all batches of the split 'split' into one TSV file.

    Parameters:
        split (str): The name of the split, either 'train', 'validation', or 'test'.
        file_paths (list): The list of paths to the generated sentences in each batch.
        trues (list): All the gold readability level labels.
        experiment (str): The name of the experiment (folder where all logs are saved).
    '''
    # readability_levels_mapping = {0:'EasilyReadableText', 1:'StandardReadableText', 2:'HardlyReadableText'}
    sentence_length_mapping = {0:'ShortFullText', 1:'LongFullText'}
    with open(f'../results/{experiment}/generations_{split}.tsv', 'a') as all_gens:
        tsv_writer = csv.writer(all_gens, delimiter='\t')
        # iterate through batch files
        for i, path in enumerate(file_paths):
            # store gold labels and generations for this batch
            abs_path = f'../{path}'
            with open(abs_path, 'r') as batch_gens:
                for j, row in enumerate(batch_gens):
                    tsv_writer.writerow([sentence_length_mapping[trues[i+j]], row])

            # remove batch-wise generations file path
            os.remove(abs_path)


def save_epoch_data(target, trues, preds, epoch_or_iter, experiment):
    '''
    This function saves the trues and preds at each epoch for train and validation and for each run on test set (several runs are performed to ensure stability)

    Parameters:
        target (str): either 'train', 'validation' or 'test'
        trues (tensor): the gold labels for sentence readability levels
        preds (tensor): the readability levels of the generated sentences
        epoch_or_iter (int):       the # of the epoch (train and val sets) or the iter (test set)
    '''
    with open(f'../results/{experiment}/predictions_{target}.csv', 'a+', newline='') as f:
        write = csv.writer(f)

        # Write data in columns
        for i in range(len(trues)):
            write.writerow([ trues[i], preds[i], epoch_or_iter ])
