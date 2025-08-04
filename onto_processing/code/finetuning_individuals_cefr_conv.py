import os
import argparse
from rdflib import *
from owlready2 import *

# from other scripts
from utils import *
from language_processing import *
from preprocessing import *

def remove_previous_individuals(ontology):
    individuals_to_keep = ontology.search(is_a = ontology.CEFRLevel)
    for indiv in list(ontology.individuals()):
        if str(indiv) not in str(individuals_to_keep):
            destroy_entity(indiv)

def individual_generation(onto, conv_index, file_path, experiment, repo):
    '''
    Generates RDF file representing one conversation with utterances and their linguistic features.

    Args:
        onto (Ontology): The ontology to add individuals to.
        conv_index (int): Index of the conversation batch.
        file_path (str): Path to the file containing utterances.
        experiment (str): Experiment identifier.
        repo (str): Repository name.
    '''

    remove_previous_individuals(onto) # empty the ontology individuals

    os.makedirs(f'../{repo}/rdf', exist_ok=True) # create the appropriate directory for further saving (only if it does not exist yet)

    # retrieve linguistic features for each utterance
    conv_dict = dict()
    with open(os.path.join(f"../{repo}/", file_path), 'r') as f:
        for i, utterance in enumerate(f):
            utterance = utterance.strip()
            if utterance != '':
                conv_dict = get_utt_linguistic_features_conv(nlp=NLP, utterance=str(utterance), batch_idx=conv_index, utt_idx=i, class_dict=conv_dict)

    if conv_dict != {}:
        # assign linguistic features to ontology classes and property outputs
        all_utterance_IDs = conv_dict['UtteranceID']
        all_texts = conv_dict['FullText']

        for i in range(len(all_utterance_IDs)):
            utt_ID = all_utterance_IDs[i]
            utt_text = all_texts[i]

            # define the utterance individual with all the required data property values
            utt = onto.Utterance(utt_ID, 
                                fullText=[utt_text],
                                hasColeReadabilityScore=[conv_dict['cole'][i]], 
                                hasFleschKincaidReadabilityScore=[conv_dict['flesch_kincaid'][i]], 
                                hasAverageWordLength=[conv_dict['average_word_length'][i]],
                                hasMTLDmeasure=[conv_dict['mtld'][i]],
                                hasSMOGReadabilityScore=[conv_dict['mtld'][i]],
                                hasPronounDensity=[conv_dict['pronoun_density'][i]],
                            )
            
    # define the conversation individuals 
    conv = onto.Conversation(f'{conv_index}_conv')
    conv.containsUtterance = [onto.Utterance(utt_id) for utt_id in all_utterance_IDs]

    # say that the conversation does not contain any more utterance than those specified in the ontology
    nb_utt_in_conv = len(all_utterance_IDs)
    conv.is_a.append(onto.containsUtterance.exactly(nb_utt_in_conv, onto.Utterance))
    
    # declare that all Utterance individuals are different
    AllDifferent([onto.Utterance(utt_id) for utt_id in all_utterance_IDs])
    
    # apply reasoner on the ontology
    with onto:
        sync_reasoner_pellet(debug=0) # debug 0=no verbose, 1=verbose command, 2=verbose command+output

    # save the RDF triplets along with the utterance ontology
    onto.save(file=f"../{repo}/rdf/individual_batch_{conv_index}_{experiment}.rdf", format="rdfxml")

if __name__ == "__main__":

    NLP = spacy.load("en_core_web_sm")
    onto = load_ontology("./rdf/controllable_conversation.rdf")

    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--batch_index", help="Batch index from fine-tuning procedure.", type=int, required=True)
    parser.add_argument("-p", "--file_path", help="Path to generated sentences for the current batch.", type=str, required=True)
    parser.add_argument("-e", "--experiment", help="The experiment name to add in the individual file name.", type=str, required=True)
    parser.add_argument("-r", "--repository", help="The name of the repository where batch generations are stored.", type=str, required=True)

    # retrieve values from argparse
    arguments = parser.parse_args()
    conv_index = arguments.batch_index
    file_path = arguments.file_path
    experiment = arguments.experiment
    repo = arguments.repository

    individual_generation(onto, conv_index, file_path, experiment, repo)