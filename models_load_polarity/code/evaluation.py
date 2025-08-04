import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_from_disk('../data/idx_cefr_ed_dd_go') # aggregates the validation splits of CEFR-S, EmpatheticDialogues, GoEmotions and helf of DailyDialog (see paper for dataset descriptions)
eval_dataset = concatenate_datasets([dataset['validation'], dataset['test']])

# reset index column to further map the right prediction to the right utterance
if 'index' in eval_dataset.column_names:
    eval_dataset = eval_dataset.remove_columns('index')
eval_dataset = eval_dataset.map(lambda _, idx: {"index": idx}, with_indices=True)

load_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
polarity_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

def load_wise_tokenization(example):
    return load_tokenizer(example["utterance"], padding="max_length", truncation=True, max_length=128)

def polarity_wise_tokenization(example):
    return polarity_tokenizer(example["utterance"], padding="max_length", truncation=True, max_length=128)

class RobertaWithAdditionalLayers(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(RobertaWithAdditionalLayers, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # freeze pre-trained roberta
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False

        # Additional linear layers with dropout
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, 1)
        
        # Dropout and activation layers
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # CLS token output
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        logits = self.classifier(x)
        output = self.sigmoid(logits)
        
        return output
    

# load the best classifiers for each task
load_predictor = RobertaWithAdditionalLayers()
load_predictor.load_state_dict(torch.load('../models/best_load_predictor.pt', weights_only=True))
load_predictor.to(device)

polarity_predictor = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
polarity_state_dict = torch.load('../models/best_polarity_predictor.pt')
polarity_predictor.load_state_dict(polarity_state_dict)
polarity_predictor.to(device)

# tokenize dataset with both tokenizers
load_tokenized_dataset = eval_dataset.map(load_wise_tokenization, batched=True)
load_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'index'])
polarity_tokenized_dataset = eval_dataset.map(polarity_wise_tokenization, batched=True)
polarity_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'index'])

# get the dataloader for both tasks. No splits here since we evaluate on the whole dataset
load_loader = DataLoader(load_tokenized_dataset, batch_size=32, shuffle=True)
polarity_loader = DataLoader(polarity_tokenized_dataset, batch_size=32, shuffle=True)

load_predictions, polarity_predictions = [], []
with torch.no_grad():

    # perform eval for the load
    for batch in tqdm(load_loader):
        # get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        indexes = batch['index'].cpu().numpy()

        # apply model
        output = load_predictor(input_ids=input_ids, attention_mask=attention_mask)
        # store predicted class along with indexes
        probs = output.detach().cpu().numpy()
        preds = (probs >= 0.5).astype(float)
        for idx, pred in zip(indexes, preds):
            load_predictions.append({'idx': idx, 'load': pred})


    # perform eval for the polarity
    for batch in tqdm(polarity_loader):
        # get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        indexes = batch['index'].cpu().numpy()

        # apply model
        output = polarity_predictor(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits

        # store predicted class along with indexes
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        for idx, pred in zip(indexes, preds):
            polarity_predictions.append({'idx': idx, 'polarity': pred})


# store all the predictions in a dataframe
load_df = pd.DataFrame(load_predictions)
polarity_df = pd.DataFrame(polarity_predictions)

merged_df = pd.merge(load_df, polarity_df, on='idx')
merged_df.to_csv('../data/test/fixed_cefr_ed_load_polarity_predictions.csv', index=False)