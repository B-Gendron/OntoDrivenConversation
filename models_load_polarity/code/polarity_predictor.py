import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# Load the dataset and tokenizer
dataset = load_from_disk('../data/sst3')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

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
        self.classifier = nn.Linear(256, 3)
        
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
        
        return logits

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# create DataLoaders for train and validation sets
train_loader = DataLoader(tokenized_datasets['train'], batch_size=32, shuffle=True)
val_loader = DataLoader(tokenized_datasets['validation'], batch_size=32)

# load the model and freeze all layers except the last three
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# put model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# instantiate optimizer and loss function (binary clf seen as multiclass with 2 classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()

# training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # training
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # compute loss and predictions
        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # compite accuracy and F1 score for training set
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_train_loss = train_loss / len(train_loader)
    
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.10f}, Accuracy: {train_accuracy:.10f}, F1 Score: {train_f1:.10f}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    best_val_accuracy = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            # compute loss and predictions
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels)
    
    # compute accuracy and F1 score on validation set
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    # save the model if it has the highest validation accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model_save_path = '../models/best_polarity_predictor.pt'
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved with accuracy {best_val_accuracy:.4f} at {model_save_path}")
