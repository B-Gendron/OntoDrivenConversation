import os
import argparse
from rdflib import *
from owlready2 import *
from transformers import RobertaModel, AutoModelForSequenceClassification, RobertaTokenizer, AutoTokenizer
import torch.nn as nn

# from other scripts
from utils import *
from language_processing import *
from preprocessing import *

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
    

def remove_previous_individuals(ontology):
    for indiv in list(ontology.individuals()):
        destroy_entity(indiv)


def individual_generation(onto, load_predictor, polarity_predictor, load_tokenizer, polarity_tokenizer, batch_index, file_path, experiment, repo):
    '''
    Generates ontology individuals for utterances including linguistic features and predictions.

    Args:
        onto (Ontology): The ontology to add individuals to.
        load_predictor: Pre-trained model for load prediction.
        polarity_predictor: Pre-trained model for polarity prediction.
        load_tokenizer: Tokenizer for loading prediction.
        polarity_tokenizer: Tokenizer for polarity prediction.
        batch_index (int): Index of the current batch.
        file_path (str): Path to the file containing utterances.
        experiment (str): Experiment identifier.
        repo (str): Repository name.
    '''

    remove_previous_individuals(onto) # empty the ontology individuals
    os.makedirs(f'../{repo}/rdf', exist_ok=True) # create the appropriate directory for further saving (only if it does not exist yet)

    # retrieve linguistic features for each utterance
    batch_dict = dict()
    with open(os.path.join(f"../{repo}/", file_path), 'r') as f:
        for i, utterance in enumerate(f):
            utterance = utterance.strip()
            if "|" in utterance and len(utterance.split(" | ")) > 1:
                utterance = utterance.split(" | ")[1] # remove the "Asked: A1 |" stuff
            if utterance != '':
                batch_dict = get_utt_linguistic_features_emo(utterance, load_predictor, polarity_predictor, load_tokenizer, polarity_tokenizer, i, class_dict=batch_dict)

    if batch_dict != {}:
        # assign linguistic features to ontology classes and property outputs
        all_utterance_IDs = batch_dict['FullTextID']
        all_texts = batch_dict['FullText']

        for i in range(len(all_utterance_IDs)):
            utt_ID = all_utterance_IDs[i]
            utt_text = all_texts[i]
            utt = onto.Utterance(utt_ID, 
                                fullText=[utt_text],
                                hasLoad=[batch_dict['load'][i]],
                                hasPolarity=[batch_dict['polarity'][i]]
                            )
            
            intention = onto.NextIntention(f'{utt_ID}_int')
            intention.isInferredFromUtterance = onto.Utterance(utt_ID)
            
    # apply reasoner on the ontology
    with onto:
        sync_reasoner_pellet(debug=0) # debug 0=no verbose, 1=verbose command, 2=verbose command+output

    # save the RDF triplets along with the utterance ontology
    onto.save(file=f"../{repo}/rdf/individual_batch_{batch_index}_{experiment}.rdf", format="rdfxml")

if __name__ == "__main__":

    NLP = spacy.load("en_core_web_sm")
    onto = load_ontology("./rdf/emotion_strategy.rdf")

    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--batch_index", help="Batch index from fine-tuning procedure.", type=int, required=True)
    parser.add_argument("-p", "--file_path", help="Path to generated sentences for the current batch.", type=str, required=True)
    parser.add_argument("-e", "--experiment", help="The experiment name to add in the individual file name.", type=str, required=True)
    parser.add_argument("-r", "--repository", help="The name of the repository where batch generations are stored.", type=str, required=True)

    # retrieve values from argparse
    arguments = parser.parse_args()
    batch_index = arguments.batch_index
    file_path = arguments.file_path
    experiment = arguments.experiment
    repo = arguments.repository


    # retrieve the fine-tuned load and polarity classifier so that we can perform inference on them in language_processing.py script
    load_predictor = RobertaWithAdditionalLayers()
    load_predictor.load_state_dict(torch.load('./models/best_load_predictor.pt', weights_only=True))

    polarity_predictor = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    polarity_state_dict = torch.load('./models/best_polarity_predictor.pt')
    polarity_predictor.load_state_dict(polarity_state_dict)

    load_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    polarity_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

    individual_generation(onto, 
                          load_predictor, 
                          polarity_predictor, 
                          load_tokenizer,
                          polarity_tokenizer,
                          batch_index, 
                          file_path, 
                          experiment, 
                          repo)