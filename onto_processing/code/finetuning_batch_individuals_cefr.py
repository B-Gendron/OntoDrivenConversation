import os
import argparse
from rdflib import *
from owlready2 import *
import spacy

# from other scripts
from utils import *
from language_processing import *

def remove_previous_individuals(ontology):
    for indiv in list(ontology.individuals()):
        destroy_entity(indiv)

def individual_generation(onto, easy_words, batch_index, file_path, experiment, repo):
    '''
    Generates ontology individuals for a batch of utterances.

    Args:
        onto (Ontology): The ontology to add individuals to.
        easy_words (list): List of easy words.
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
                batch_dict = get_utt_linguistic_features_cefr(nlp=NLP, utterance=str(utterance), easy_words=easy_words, dial_idx=batch_index, utt_idx=i, class_dict=batch_dict)

    if batch_dict != {}:
        # assign linguistic features to ontology classes and property outputs
        all_utterance_IDs = batch_dict['FullTextID']
        all_texts = batch_dict['FullText']

        current_token_index = 0
        for i in range(len(all_utterance_IDs)):
            n_tokens = batch_dict['n_tokens'][i]
            utt_ID = all_utterance_IDs[i]
            utt_text = all_texts[i]
            utt = onto.FullText(utt_ID, 
                                fullText=[utt_text],
                                hasColeReadabilityScore=[batch_dict['cole'][i]], 
                                hasFleschKincaidReadabilityScore=[batch_dict['flesch_kincaid'][i]], 
                                hasAverageWordLength=[batch_dict['average_word_length'][i]],
                                hasMTLDmeasure=[batch_dict['mtld'][i]],
                                hasSMOGReadabilityScore=[batch_dict['smog'][i]],
                                hasPronounDensity=[batch_dict['pronoun_density'][i]],
                            )
            # get all token info for this utterance
            all_token_ids = batch_dict['TokenID'][current_token_index:current_token_index+n_tokens]
            all_token_strings = batch_dict['Token'][current_token_index:current_token_index+n_tokens]

            # containsToken (FullText -> Token)
            utt.containsToken = [onto.Token(tok_id) for tok_id in all_token_ids]

            for i in range(len(all_token_ids)):
                # DATAPROP tokenString (Token -> value (str))
                tok_id = utt.containsToken[i]
                tok_id.tokenString = [all_token_strings[i]]

            current_token_index += n_tokens
            
    # apply reasoner on the ontology
    with onto:
        sync_reasoner_pellet(debug=0) # debug 0=no verbose, 1=verbose command, 2=verbose command+output

    # save the RDF triplets along with the utterance ontology
    onto.save(file=f"../{repo}/rdf/individual_batch_{batch_index}_{experiment}.rdf", format="rdfxml")

if __name__ == "__main__":

    NLP = spacy.load("en_core_web_sm")
    onto = load_ontology("./rdf/utterances_cefr.rdf")

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

    # load the list of easy words to compute dale chall score
    with open('./resources/easy_words.txt', 'r') as file:
        easy_words = set(word.strip().lower() for word in file.readlines())

    individual_generation(onto, easy_words, batch_index, file_path, experiment, repo)