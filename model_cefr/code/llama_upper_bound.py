import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

df = pd.read_csv('../data/cefr_leveled_texts.csv')
label_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
df['label'] = df['label'].map(label_mapping)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

clf_model = AutoModelForSequenceClassification.from_pretrained(  
    "meta-llama/Meta-Llama-3-8B-Instruct",
    pad_token_id=tokenizer.pad_token_id,
    num_labels=len(label_mapping),
    low_cpu_mem_usage=True,
    return_dict=True,
    device_map='cuda:0'  # assume GPU is available
)

config = LoraConfig(
        r=16,                                         # rank of lora module
        lora_alpha=4,                                 # resclaling weights parameters
        target_modules=["q_proj", "v_proj", "score"], # act on relevant layers, in decoders and at the top for prediction
        bias="lora_only",                             # should be better than default setting in our case
        lora_dropout=0.1,                             # conventional setting
    )

model = get_peft_model(clf_model, config)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
data_collator = DataCollatorWithPadding(tokenizer)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# apply tokenization
max_length = 200
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, max_length=max_length, return_tensors="pt", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# create DataLoaders for the training and validation datasets
train_loader = DataLoader(train_dataset.remove_columns(['text', '__index_level_0__']),
                          batch_size=8, shuffle=True, collate_fn=data_collator)

val_loader = DataLoader(val_dataset.remove_columns(['text', '__index_level_0__']),
                        batch_size=8, collate_fn=data_collator)

train_labels = train_dataset['labels']
val_labels = val_dataset['labels']

# put model to the right device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 50
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()

        # batch to the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) 
        
        # train
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} finished with average loss: {epoch_loss / len(train_loader)}")

# validation loop
model.eval()
all_logits = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        all_logits.append(logits.cpu())

# compute predictions
all_logits = torch.cat(all_logits)
pred_labels = torch.argmax(all_logits, dim=1).numpy()

# back to the original label format
inv_label_mapping = {v: k for k, v in label_mapping.items()}
true_labels = [inv_label_mapping[label] for label in val_labels]
pred_labels = [inv_label_mapping[label] for label in pred_labels]

# classification metrics
print(classification_report(true_labels, pred_labels, target_names=label_mapping.keys()))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels, labels=list(label_mapping.keys())))
