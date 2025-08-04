import os
from datasets import load_dataset

# Load the dataset
ds = load_dataset("SetFit/sst5")

# reduce to 3 labels in the following way
#   Originally: 0=very neg, 1=neg, 2=neutral, 3=pos, 4=very pos
#   Now: 0=neg, 1=neutral, 2=pos
def transform_label(value):
    if value in [0, 1]:
        return 0
    elif value == 2:
        return 1
    elif value in [3, 4]:
        return 2

# apply label transformation
for split in ds.keys():
    ds[split] = ds[split].remove_columns("label_text")
    ds[split] = ds[split].map(lambda x: {"label": transform_label(x["label"])})

output_dir = '../data/sst5'
os.makedirs(output_dir, exist_ok=True)
ds.save_to_disk(output_dir)