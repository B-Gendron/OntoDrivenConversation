import os
import argparse
from tqdm import tqdm
import time
from rdflib import *
from owlready2 import *
from urllib.parse import quote, unquote
# from other scripts
from utils import *
from language_processing import *

owlready2.reasoning.JAVA_MEMORY = 4000

def load_ontology(path):
    # check for the validity of the given path
    assert os.path.exists(path), (f"'{path}' is not a valid path.")

    # load the ontology
    onto = get_ontology(path).load()

    return onto

def remove_previous_individuals(ontology):
    print(list(ontology.classes()))
    individuals_to_delete = ontology.search(is_a = ontology.UserUtterance) + ontology.search(is_a = ontology.AgentUtterance)
    print(individuals_to_delete)
    for indiv in list(ontology.individuals()):
        if str(indiv) in str(individuals_to_delete):
            destroy_entity(indiv)


def count_ontology_elements(onto):
    '''
        A debug function that displays the number of elements of each type in a OWL ontology (using owlready2).
    '''
    counts = {
        "Classes": len(list(onto.classes())),
        "Individuals": len(list(onto.individuals())),
        "Object Properties": len(list(onto.object_properties())),
        "Data Properties": len(list(onto.data_properties())),
        "Annotation Properties": len(list(onto.annotation_properties()))
    }
    
    print("Ontology Element Counts:")
    for key, value in counts.items():
        print(f"{key}: {value}")
    
    return counts


def individual_generation(world, onto, nlp, conv_index, file_path, load_predictor, load_tokenizer, polarity_predictor, polarity_tokenizer, experiment):
    '''
        The generated rdf file represents one conversation.
    '''
    # print(count_ontology_elements(onto))
    # remove_previous_individuals(onto) # empty the ontology individuals
    # retrieve linguistic features for each utterance
    conv_dict = dict()
    with open(file_path, 'r') as f:
        for i, utterance in enumerate(f):
            utterance = utterance.strip()
            if utterance != '':
                conv_dict = get_utt_linguistic_features_conv(nlp, str(utterance), load_predictor, load_tokenizer, polarity_predictor, polarity_tokenizer, conv_index, i, conv_dict)

    if conv_dict != {}:
        # assign linguistic features to ontology classes and property outputs
        all_utterance_IDs = conv_dict['UtteranceID']
        all_texts = conv_dict['FullText']

        for i in range(len(all_utterance_IDs)): 
            utt_ID = all_utterance_IDs[i]
            utt_text = all_texts[i]

            # define the utterance individual with all the required data property values
            utt = onto.UserUtterance(utt_ID)

            # Data properties
            utt.fullText = [utt_text]
            utt.hasColeReadabilityScore = [conv_dict['cole'][i]]
            utt.hasFleschKincaidReadabilityScore = [conv_dict['flesch_kincaid'][i]]
            utt.hasAverageWordLength = [conv_dict['average_word_length'][i]]
            utt.hasMTLDMeasure = [conv_dict['mtld'][i]]
            utt.hasSMOGReadabilityScore = [conv_dict['smog'][i]]
            utt.hasPronounDensity = [conv_dict['pronoun_density'][i]]

            # Object properties (subproperties of hasCharacteristic)
            utt.hasEmotionalLoad = onto.search_one(iri="*#loaded") if conv_dict['load'][i] == 1 else onto.non_loaded
            utt.hasPolarity = (
                onto.negative if conv_dict['polarity'][i] == 0
                else onto.neutral if conv_dict['polarity'][i] == 1
                else onto.positive
            )

    
    # define the conversation individuals 
    conv = onto.Conversation(f'{conv_index}_conv')
    conv.includes = [onto.Utterance(utt_id) for utt_id in all_utterance_IDs]

    # say that the conversation does not contain any more user utterance than those specified in the ontology
    nb_utt_in_conv = len(all_utterance_IDs)
    conv.is_a.append(onto.includes.exactly(nb_utt_in_conv, onto.UserUtterance))
    
    # declare that all Utterance individuals are different
    AllDifferent([onto.UserUtterance(utt_id) for utt_id in all_utterance_IDs])
    
    # apply reasoner on the world where the ontology has been loaded
    tic = time.time()
    with onto:
        sync_reasoner_pellet(world, debug=0, infer_property_values=True) # debug 0=no verbose, 1=verbose command, 2=verbose command+output
    toc = time.time()

    # save the RDF triplets along with the utterance ontology
    onto.save(file=f"{rdf_dir()}/{experiment}.rdf", format="rdfxml")

    return toc-tic

# if __name__ == "__main__":

#     NLP = spacy.load("en_core_web_sm")
#     onto = load_ontology("../ontologies/controllable_conversation.rdf")

#     # setup parser
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--batch_index", help="Batch index from fine-tuning procedure.", type=int, required=True)
#     parser.add_argument("-p", "--file_path", help="Path to generated sentences for the current batch.", type=str, required=True)
#     parser.add_argument("-e", "--experiment", help="The experiment name to add in the individual file name.", type=str, required=True)
#     parser.add_argument("-r", "--repository", help="The name of the repository where batch generations are stored.", type=str, required=True)

#     # retrieve values from argparse
#     arguments = parser.parse_args()
#     conv_index = arguments.batch_index
#     file_path = arguments.file_path
#     experiment = arguments.experiment
#     repo = arguments.repository

#     individual_generation(onto, conv_index, file_path, experiment, repo)