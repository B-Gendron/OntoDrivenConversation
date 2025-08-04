import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import nltk
import pandas as pd
import os
import json
from datasets import DatasetDict
from termcolor import colored

def features_boxplots(X_train, y_train, num_cols=4):
    '''
    Generate boxplots for each feature in the dataset, grouped by proficiency level.

    Args:
        X_train (pd.DataFrame): The input features.
        y_train (pd.Series): The target labels corresponding to the features.
        num_cols (int): Number of columns to arrange the subplots.

    Saves:
        A figure named 'boxplots.png' showing the boxplots.
    '''
    num_features = X_train.shape[1]
    num_rows = math.ceil(num_features / num_cols)

    plt.figure(figsize=(num_cols * 5, num_rows * 4))

    for i, column in enumerate(X_train.columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.boxplot(x=y_train, y=X_train[column])
        plt.title(f'Boxplot of {column} by Level')

    plt.tight_layout()
    plt.savefig('boxplots.png')


def features_violinplots(X_train, y_train, num_cols=4):
    '''
    Generate violin plots for each feature in the dataset, grouped by proficiency level.

    Args:
        X_train (pd.DataFrame): The input features.
        y_train (pd.Series): The target labels corresponding to the features.
        num_cols (int): Number of columns to arrange the subplots.

    Saves:
        A figure named 'violin_plots.png' showing the violin plots.
    '''
    num_features = X_train.shape[1]
    num_rows = math.ceil(num_features / num_cols)

    plt.figure(figsize=(num_cols * 5, num_rows * 4))

    for i, column in enumerate(X_train.columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.violinplot(x=y_train, y=X_train[column])
        plt.title(f'Boxplot of {column} by Level')

    plt.tight_layout()
    plt.savefig('violin_plots.png')


def split_sentences(df):
    '''
    Split long texts into individual sentences, preserving their original IDs.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'text' columns.

    Returns:
        pd.DataFrame: A new DataFrame where each row corresponds to a single sentence.
    '''
    new_rows = []
    for _, row in df.iterrows():
        sentences = nltk.sent_tokenize(row['text'])  # Split text into sentences
        for sentence in sentences:
            new_rows.append({'id': row['id'], 'sentence': sentence})  # Track original sample 'id' or index
    
    return pd.DataFrame(new_rows)


def activate_gpu(force_cpu=False):
    '''
    Return the appropriate device based on GPU availability.

    Args:
        force_cpu (bool): A flag to force using the CPU, regardless of GPU availability.

    Returns:
        str: The name of the selected device ('cuda', 'mps', or 'cpu').
    '''
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            print('DEVICE = ', colored('CPU', "blue"))
    return device

# -----------------------------------------------------------------------------------------
# Data management
# -----------------------------------------------------------------------------------------

def load_dataset(dataset_name):
    '''
    Load a dataset saved in the HuggingFace format, located in the data folder.

    Args:
        dataset_name (str): The name of the dataset, corresponding to an existing folder inside the data folder.

    Returns:
        DatasetDict: The dataset as a DatasetDict object.

    Raises:
        Exception: If no folder with the specified name is found in the data folder.
    '''
    path = f"../data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"../data/{dataset_name}")
    else: 
        raise Exception("No folder with the such name found in the data folder.")

    return dataset


def save_dataset(dataset, dataset_name, output_format='huggingface'):
    '''
    Save the dataset into a HuggingFace format using the `save_to_disk` method.

    Args:
        dataset (DatasetDict): The dataset to save in a DatasetDict format.
        dataset_name (str): The name to give to the file.
        output_format (str): Either 'huggingface' or 'json'. Default is 'huggingface'.

    Prints:
        A message indicating the accepted formats if an unrecognized format is given.
    '''
    if output_format == "huggingface":
        dataset.save_to_disk(f'../data/{dataset_name}')
    elif output_format == "json":
        with open(f"../data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")


def save_cefr_s_into_csv():
    '''
    Save the dataset splits into .csv files for further processing.
    '''
    cefr = load_dataset('edesaras/CEFR-Sentence-Level-Annotations')
    train_df = cefr['train'].to_pandas()
    val_df = cefr['validation'].to_pandas()
    test_df = cefr['test'].to_pandas()

    train_df.to_csv('../data/cefr/train.csv', index=False)
    val_df.to_csv('../data/cefr/validation.csv', index=False)
    test_df.to_csv('../data/cefr/test.csv', index=False)