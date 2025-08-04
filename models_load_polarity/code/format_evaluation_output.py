import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict

# get the predictions and the utterances and store them in the same dataframe
evaluation_df = pd.read_csv('../data/test/fixed_cefr_ed_load_polarity_predictions.csv')
evaluation_df['load'] = evaluation_df['load'].str.strip('[]').astype(float).astype(int)
evaluation_df = evaluation_df.sort_values(by='idx').reset_index(drop=True)

test_data = load_from_disk('../data/test/utt_cefr_ed_dd_go_test')
test_df = pd.DataFrame.from_dict(test_data)

evaluation_df = evaluation_df.rename(columns={'idx': 'index'})
merged_df = pd.merge(evaluation_df, test_df, on='index')
merged_df = merged_df[['index', 'utterance', 'load', 'polarity']]

output_path = "../data/test/res_annotated_utterances.csv"
merged_df.to_csv(output_path, index=False)

# create a dataset of label-wrapped sentences from it
def determine_label(row):
    if row['polarity'] == 0 and row['load'] == 1:
        return "LoadedNegative"
    elif row['polarity'] == 1 and row['load'] == 1:
        return "LoadedNeutral"
    elif row['polarity'] == 2 and row['load'] == 1:
        return "LoadedPositive"
    elif row['polarity'] == 0 and row['load'] == 0:
        return "NonLoadedNegative"
    elif row['polarity'] == 1 and row['load'] == 0:
        return "NonLoadedNeutral"
    elif row['polarity'] == 2 and row['load'] == 0:
        return "NonLoadedPositive"
    
merged_df['label'] = merged_df.apply(determine_label, axis=1)

merged_df['utterance'] = merged_df.apply(
    lambda row: f"[NextIntention: {row['label']}] {row['utterance']} [NextIntention: {row['label']}]",
    axis=1
)

label_wrapped_merged_df = merged_df[['utterance']]
label_wrapped_merged_df.to_csv('../data/test/label_wrapped_utterances.csv')

# save this dataset into a DatasetDict format to use it for Llama3 CLM fine-tuning
train_df = label_wrapped_merged_df.sample(frac=0.8, random_state=42)
remaining_df = label_wrapped_merged_df.drop(train_df.index)
validation_df = remaining_df.sample(frac=0.5, random_state=42)
test_df = remaining_df.drop(validation_df.index)

# convert each split to a Dataset and create + save DatasetDict
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

dataset_dict.save_to_disk("../data/label_wrapped/cefr_ed_dd_go_label_wrapped")