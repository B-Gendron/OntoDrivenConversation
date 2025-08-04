import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

train_df = pd.read_csv('../data/cefr/train.csv')
val_df = pd.read_csv('../data/cefr/validation.csv')
test_df = pd.read_csv('../data/cefr/test.csv')

df = pd.concat([train_df, val_df, test_df], ignore_index=True)
annotator_I = df['Annotator I']
annotator_II = df['Annotator II']

report = classification_report(annotator_I, annotator_II)
conf_matrix = confusion_matrix(annotator_I, annotator_II)
mae = mean_absolute_error(annotator_I, annotator_II)

print("Classification Report:")
print(report)
print("MAE: ", mae)
print("\nConfusion Matrix:")
print(conf_matrix)