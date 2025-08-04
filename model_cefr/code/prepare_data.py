import pandas as pd
import re
import argparse
import os

def gold_label_strategy(data, strategy):
    '''
    Assign gold labels based on a specified strategy.

    Args:
        data (pd.DataFrame): Input data with 'Annotator I' and 'Annotator II' columns.
        strategy (int): Strategy to apply (0: max, 1: 'Annotator I', 2: 'Annotator II').

    Returns:
        pd.DataFrame: Data with assigned 'level' and no 'Annotator I' or 'Annotator II'.
    '''
    if strategy == 1:
        data['level'] = data['Annotator I']

    elif strategy == 2:
        data['level'] = data['Annotator II']

    elif strategy == 0:
        data['level'] = data[['Annotator I', 'Annotator II']].max(axis=1)

    else:
        print('Wrong strategy value. The accepted values are 0, 1 and 2.')

    data = data.drop(['Annotator I', 'Annotator II'], axis=1)

    return data 


def remove_extra_spaces(sentence):
    '''
    Clean extra spaces around punctuation and quotations in a sentence.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Cleaned sentence.
    '''

    # remove spaces before punctuation ("Hello !"" --> "Hello!"")
    sentence = re.sub(r'\s+([?.!,’])', r'\1', sentence)

    # replace spaces in contractions ("don ' t" --> "don't")
    sentence = re.sub(r'’\s', r'’', sentence)
    sentence = re.sub(r'\s’', r'’', sentence)

    # fix spacing around quotes and other punctuation marks ("' Quotation '" --> "'Quotation'")
    sentence = re.sub(r'\s+([’‘“”"])', r'\1', sentence)
    sentence = re.sub(r'([’‘“”"])\s+', r'\1', sentence)

    return sentence

def clean_text(data):
    '''
    Apply extra space removal to text data.

    Args:
        data (pd.DataFrame): DataFrame containing a 'text' column.

    Returns:
        pd.DataFrame: DataFrame with cleaned 'text'.
    '''
    data['text'] = data['text'].map(lambda s: remove_extra_spaces(s))
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str)
    args = parser.parse_args()

    DATA_FOLDER = args.data_folder
    os.makedirs(os.path.dirname(f'../data/cefr_processed/'), exist_ok=True)

    for split in ['train', 'validation', 'test']:
        data = pd.read_csv(f'{DATA_FOLDER}/{split}.csv')
        labelled_cefr = gold_label_strategy(data, strategy=0)
        cefr = clean_text(labelled_cefr)
        cefr.to_csv(f'../data/cefr_processed/{split}.csv')
