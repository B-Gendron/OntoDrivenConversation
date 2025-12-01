# Ontology-Driven Control of Conversational Agents with a Use-Case for Job Interview Practice

This repository is dedicated to ESWC 2026 reviewing commity for the paper "Ontology-Driven Control of Conversational Agents with a Use-Case for Job Interview Practice". It gathers the scripts and objects that has been developed and used by the author as part of this work. As mentioned in the paper, we provide the ontology as well as the datasets used from the different training phases (if they are not directly provided, the link to the HuggingFace repository is provided in the source code). We also provide the adapter models that results of our fine-tuning procedure. However, we could not provide the predictors for the emotional load and the emotion polarity as they exceed GitHub's file size limit of 100.00 MB. We apologize for the inconvenience and tried to put as much of the useful material as possible in the repository.

## Table of Contents

- [Content overview](#contents)
- [Setup guide](#quickstart)

<a name="contents"></a>
## Content overview
The repository is organized as follows, containing five sub-folders:

```
├─── conv_agent
│   ├─── adapters
│   ├─── code
│   ├─── eval_corpus
│   └─── ontologies
├─── grpo_finetuning
│   ├─── code
│   ├─── metrics
│   ├─── call_ontology_cefr.sh
│   └─── call_ontology_emo.sh
├─── model_cefr
│   ├─── code
│   ├─── data
|   ├─── models
│   ├─── plots
│   └─── resources
├─── models_load_polarity
│   ├─── code
│   └─── logs
├─── models_load_polarity
│   ├─── code
│   ├─── rdf
│   └─── resources
├─── README.md
└─── requirements.txt
```

### `conv_agent` 

This subfolder contains all the scripts to interact with our fine-tuned conversational agent. 

It also contains, in the `eval_corpus` folder, the 15 conversations evaluated by the human annotators (see paper). Each conversation is stored in a dedicated folder, where each conversation turn is stored in a JSON file with fields like:

- `prompt` and `generated_content`: the user input and agent response 
- `current_phase`: detected current phase of the conversation
- `next_characteristics`: expected descriptor values
- `user_utt_features`**: linguistic features of the user utterance (useful for CEFR level prediction)
- `id`: unique identifier of the conversation  
- `runtimes`: reasoning time of pellet reasoner and inference time of the language model

### `grpo_finetuning`

This subfolders contains the code to run GRPO fine-tuning on LLMs and evaluate the generation control according to the descriptor values. See the section below "Run GRPO fine-tuning" for the instructions to run the code.

### `onto_processing`

This subfolders contains the ontologies and (normally, when enough space) the auxiliary models to evaluate the descriptor values on the textual content and incorporate this information in the ontology. It is used only for the GRPO fine-tuning part. For the conversational agent the ontology in processed in the same folder. Each utterance is seen as an ontology individual. For that, we use the `owlready2` python library that allows to create an ontology and call reasoners to perform inference on it. This part of the code is crucial as it allows to retrieve the inferred concepts that will then be used to annotate fine-tuning data, to evaluate generated content and to guide the implementation of conversational strategies in the language model.

### `model_cefr`

This subfolder contains all the utils to train the decision tree classifier that classifies texts in their right Proficiency CEFR levels (A1, A2, B1, B2, C1, C2). The fitted decision tree that we used for further evaluation is stored in `models/best_clf.pkl`. The decision tree training script is `decision_tree.py`. In order to compare this classifier to some upper bounds, we tried to fit a Llama model (with a classification head) on the same task, in `llama_upper_bound.py`. Additionally, as we have a dataset provided with two expert annotations for each data sample, we compute the inter-annotator agreeement score in `annotators_upper_bound.py`.

### `models_load_polarity` 

This subfolder contains the code to fine-tune RoBERTa-based models on load and polarity classification. Load classification is a binary classification (the text is either loaded or non loaded), while polarity classification is a three-class classification (negative, neutral and positive polarities). These classifiers are fine-tuned respectively in `code/load_predictor.py` and `code/polarity_predictor.py`. Then, the script `code/evaluation.py` is used to annotate the data dedicated to LLM fine-tuning with the classifiers predictions. Moreover, in the `logs` folder, we provide the logs of the best fine-tuning runs of the two classifiers.

<a name="quickstart"></a>
## Setup guide

Start be setting up all the required models and python modules to run the code:
```bash
cd OntoDrivenConversation
python3 -m venv interview_strat
source interview_strat/bin/activate
# install the required libraries and their dependencies
pip3 install -r requirements.txt
# install SpaCy model for linguistic feature retrieval
python3 -m spacy download en_core_web_sm
```

The following subsections give examples of how to run the code for fine-tuning and evaluation, providing we have all the necessary data and auxiliary models aleardy fitted.

### Engage a conversation with the agent

To start a conversation, simply go to `conv_agent/code/` folder and run:
```bash
python3 generation.py
```

It will display an input to insert user content. 
```bash
User > 
```

### Run GRPO fine-tuning

To run GRPO fine-tuning, use the script `grpo_finetuning/code/run_grpo.sh` along with an experiment name. Set the model to fine-tune using a valid HuggingFace path here:
```bash
--model_name "Qwen/Qwen2.5-7B-Instruct" \"
```
In the paper, the HuggingFace paths we used are the following:
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
Please note that in order to access to the models from HuggingFace, you might need to ask for access credentials from a HuggingFace account and wait for the approval of the provider (Meta for Llama, Alibaba Cloud for Qwen). 

After GRPO training, you can evaluate the quality of the generation control by running the evaluation script `grpo_finetuning/code/evaluation_multicontrol.py`. At the end of the evaluation steps, it will display classification reports and confusion matrices for each descriptor in the log file and store the classification metrics in a json file in the `grpo_fintuning/metrics/` folder.

In the main part of the script, put a consistent combination of pre-trained model and trained adapters obtained through GRPO fine-tuning. For instance:
```python
model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_model_name = "./checkpoints/grpo_multicontrol_qwen2.5/checkpoint-200" # put the right checkpoint id here
```
