import spacy
import pandas as pd
import os
import datasets
from pprint import pprint
from termcolor import colored
import time
from urllib.parse import quote, unquote

# readability metrics
import lftk

# other scripts
from utils import *


def describe_utterances(nlp, l_utt):
    '''
        @param nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features
        @param l_utt (list): a list of strings representing the utterances to describe

        @returns df_utts (DataFrame): a Pandas dataframe containing the feature values for all the utterances
    
    '''
    data = {'utt_idx':[], 'tok_idx':[], 'tok_id':[], 'tok_text':[], 'lem_id':[], 'lem_text':[], 'pos':[], 'tag':[]}

    # for each utterance in the list
    current_index = 1
    for utt in l_utt:
        # format utterance and instantiate token counter
        doc = nlp(utt)
        count_tokens = 0

        # for each token, compute and store feature values
        for token in doc:
            data['tok_idx'].append(count_tokens)
            data['tok_id'].append(token.norm_)
            data['tok_text'].append(token.text)
            data['lem_id'].append(token.lemma)
            data['lem_text'].append(token.lemma_)
            data['pos'].append(token.pos_)
            data['tag'].append(token.tag_)
            count_tokens += 1

        # use token counter to put the utterance index as many times as tokens it contains
        data['utt_idx'].extend(count_tokens*[current_index-1])

        # increment current utterance index
        current_index += 1
        
    df_utt = pd.DataFrame(data)
    df_utt.to_csv(f"{descriptions_dir()}/utt_description_{get_datetime()}.csv")
    return df_utt


def flesch_kincaid(doc):
    sentences = list(doc.sents)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    syllables = sum([len([syll for syll in token.text.lower() if syll in 'aeiouy']) for token in words])

    fk_score = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return fk_score

def gunning_fog_index(doc, nlp):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    complex_words = [word for word in words if len(word) > 2 and word.lower() in nlp.vocab.strings]
    fog_index = 0.4 * ((len(words) / len(sentences)) + 100 * (len(complex_words) / len(words)))
    return fog_index

def coleman_liau_index(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_chars = sum(len(token.text) for token in doc if not token.is_punct and not token.is_space)
    sentences = list(doc.sents)
    L = (num_chars / len(words)) * 100
    S = (len(sentences) / len(words)) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8
    return cli

def smog_index(doc, nlp):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    complex_words = [word for word in words if len(word) > 2 and word.lower() in nlp.vocab.strings]
    smog_index = 1.0430 * (30 * (len(complex_words) / len(sentences)))**0.5 + 3.1291
    return smog_index

def pronoun_density(doc):
    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    return len(pronouns) / len(doc) if len(doc) > 0 else 0


def dale_chall_score(doc, easy_words):
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    difficult_words = sum(1 for word in words if word not in easy_words)
    p = (difficult_words / len(words)) * 100
    dc_score = 0.1579 * p + 0.0496 * (len(words) / len(sentences))
    return dc_score

def has_conjunctions(doc):
    conjunctions = {'however', 'therefore', 'furthermore', 'although', 'because'}
    return any(token.text.lower() in conjunctions for token in doc)

def named_entity_count(doc):
    return len(doc.ents)

def adjective_ratio(doc):
    return len(doc) / (0.001+sum(1 for token in doc if token.pos_ == 'ADJ'))

def mtld(doc):
    '''
    MTLD (Measure of Textual Lexical Diversity): A more sophisticated measure that accounts for text length when evaluating lexical diversity.
    '''
    words = [token.text.lower() for token in doc if token.is_alpha]
    if not words:
        return 0
    factor_count = 0
    token_count = 0
    ttr = 1.0
    for i in range(len(words)):
        token_count += 1
        unique_tokens = len(set(words[:i + 1]))
        ttr = unique_tokens / token_count
        if ttr < 0.72:
            factor_count += 1
            token_count = 0
    return len(words) / (factor_count if factor_count > 0 else 1)

def average_word_length(doc):
    return sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0

def morphological_complexity(doc):
    inflectional_count = 0
    derivational_count = 0
    lemma_forms = defaultdict(set)
    
    # Example suffixes for derivational morphology (this is a simplification)
    derivational_suffixes = ['ness', 'ment', 'tion', 'er', 'ing', 'ly', 'able', 'ible', 'al']
    
    for token in doc:
        if token.is_alpha and not token.is_stop:
            lemma_forms[token.lemma_].add(token.text)
            # Inflectional: if lemma is different from the token form
            if token.lemma_ != token.text.lower():
                inflectional_count += 1
            # Derivational: check for common derivational suffixes
            for suffix in derivational_suffixes:
                if token.text.endswith(suffix):
                    derivational_count += 1
                    break
    
    total_tokens = len([token for token in doc if token.is_alpha and not token.is_stop])
    
    if total_tokens == 0:
        return 0, 0

    inflectional_ratio = inflectional_count / total_tokens
    morphological_diversity = len(lemma_forms) / total_tokens
    
    return inflectional_ratio, morphological_diversity


# Classes: ['FullText', 'TokenString', 'ARIReadabilityScore', 'ReadabilityScore', 'CLReadabilityScore', 'FKGLReadabilityScore', 'FOGIReadabilityScore', 'TokenPOStag', 'SMOGReadabilityScore']
def get_utt_linguistic_features(nlp, utterance, dial_idx=0, utt_idx=0, class_dict=dict()):
    '''
        This function stores the linguistic features computed from the given utterance in the dictionnary dict provided in input. 

        @param nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features from the utterance
        @param utt_idx (int): the utterance index in the description dictionary. Default is zero as default dictionary is empty
        @param class_dict (dict): the description dictionary to put utterances, IDs and linguistic features in. Default is an empty dictionary
        @param utterance (str): the utterance to study. It is bound to be an individual of type 'FullText' in the ontology. The syntax used in the values of the output dictionnary make it usable iteratively, thus considering a set of utterances. 

        @return class_values (dict): a dictionnary whose keys are the ontology classes. 
    '''
    # retrieve linguistic feature on the utterance
    doc = nlp(utterance)
    LFTK = lftk.Extractor(docs = doc)
    readability_metrics = LFTK.extract(features = ["fkgl", "fogi", "smog", "cole", "auto"])

    full_text = utterance.replace(' ', '_')
    token_norms = [token.norm for token in doc]

    # if the dictionnary doesn't contain any other utterance description, then it is empty so we need to instantiate the keys
    if len(class_dict) == 0:
        keys = ['FullText', 'FullTextID', 'Token', 'Lemma', 'TokenID', 'LemmaID', 'TokenPOS', 'TokenTag', 'fkgl', 'fogi', 'smog', 'cole', 'auto', 'n_tokens']
        class_dict = {k:[] for k in keys}

    # store the number of tokens in the utterance
    class_dict['n_tokens'].append(len(token_norms))

    # store full text text and ID
    class_dict['FullText'].append(full_text) # replace space by _ to keep consistent naming conventions with the ontology
    # Nomenclature pour utterances = indice utterance _ utt _ concat des premiers chiffres de chaque norme token
    class_dict['FullTextID'].append(f'{dial_idx}_{utt_idx}_utt_{concatenate([str(norm)[0] for norm in token_norms[:8]])}') # for the last part of the ID, keep only the 8 first characters

    # store token and lemma texts and IDs
    class_dict['Token'].extend([token.text for token in doc])
    class_dict['Lemma'].extend([token.lemma_ for token in doc])
    # Nomenclature pour tokens/lemmes = indice utterance _ indice token/lemme _ tok/lem _ norme token/lemme
    class_dict['TokenID'].extend([f'{dial_idx}_{utt_idx}_{i}_tok_{token.norm}' for i, token in enumerate(doc)])
    class_dict['LemmaID'].extend([f'{dial_idx}_{utt_idx}_{i}_lem_{token.lemma}' for i, token in enumerate(doc)])

    # POS and tags
    class_dict['TokenPOS'].extend([quote(token.pos_) for token in doc])
    # class_dict['TokenPOSID'] = [f'{utt_idx}_{i}_pos_{token.pos}' for i, token in enumerate(doc)]
    class_dict['TokenTag'].extend([quote(token.tag_) for token in doc])
    # class_dict['TokenTagID'] = [f'{utt_idx}_{i}_tag_{token.tag}' for i, token in enumerate(doc)]

    # readability metrics
    for metric in ['fkgl', 'fogi', 'smog', 'cole', 'auto']:
        class_dict[metric].append(readability_metrics[metric])

    return class_dict


# Classes: ['FullText', 'TokenString', 'ARIReadabilityScore', 'ReadabilityScore', 'CLReadabilityScore', 'FKGLReadabilityScore', 'FOGIReadabilityScore', 'TokenPOStag', 'SMOGReadabilityScore']
def get_utt_linguistic_features_cefr(nlp, utterance, easy_words, dial_idx=0, utt_idx=0, class_dict=dict()):
    '''
        This function stores the linguistic features computed from the given utterance in the dictionnary dict provided in input. 

        @param nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features from the utterance
        @param utt_idx (int): the utterance index in the description dictionary. Default is zero as default dictionary is empty
        @param class_dict (dict): the description dictionary to put utterances, IDs and linguistic features in. Default is an empty dictionary
        @param utterance (str): the utterance to study. It is bound to be an individual of type 'FullText' in the ontology. The syntax used in the values of the output dictionnary make it usable iteratively, thus considering a set of utterances. 

        @return class_values (dict): a dictionnary whose keys are the ontology classes. 
    '''
    # compute linguistic features useful for CEFR class attribution
    doc = nlp(utterance)
    ir, md = morphological_complexity(doc)
    readability_metrics = {
        'flesch_kincaid': flesch_kincaid(doc),
        # 'fogi': gunning_fog_index(doc, nlp),
        'cole': coleman_liau_index(doc),
        'mtld': mtld(doc),
        # 'inflectional_ratio': ir,
        # 'morph_diversity': md,
        'smog': smog_index(doc, nlp),
        'pronoun_density': pronoun_density(doc),
        'average_word_length': average_word_length(doc),
        # 'dc_score': dale_chall_score(doc, easy_words),
        # 'conjunctions': has_conjunctions(doc),
        # 'named_entity_count': named_entity_count(doc),
        # 'adjective_ratio': adjective_ratio(doc),
    }

    full_text = utterance.replace(' ', '_')
    token_norms = [token.norm for token in doc]

    # if the dictionnary doesn't contain any other utterance description, then it is empty so we need to instantiate the keys
    if len(class_dict) == 0:
        keys = ['FullText', 'FullTextID', 'Token', 'TokenID', 'n_tokens'] + list(readability_metrics.keys())
        class_dict = {k:[] for k in keys}

    # store the number of tokens in the utterance
    class_dict['n_tokens'].append(len(token_norms))

    # store full text text and ID
    class_dict['FullText'].append(full_text) # replace space by _ to keep consistent naming conventions with the ontology
    # Nomenclature pour utterances = indice utterance _ utt _ concat des premiers chiffres de chaque norme token
    class_dict['FullTextID'].append(f'{dial_idx}_{utt_idx}_utt_{concatenate([str(norm)[0] for norm in token_norms[:8]])}') # for the last part of the ID, keep only the 8 first characters

    # store token and lemma texts and IDs
    class_dict['Token'].extend([token.text for token in doc])
    # Nomenclature pour tokens/lemmes = indice utterance _ indice token/lemme _ tok/lem _ norme token/lemme
    class_dict['TokenID'].extend([f'{dial_idx}_{utt_idx}_{i}_tok_{token.norm}' for i, token in enumerate(doc)])

    # readability metrics
    for metric in readability_metrics.keys():
        class_dict[metric].append(readability_metrics[metric])

    return class_dict


def get_load_prediction(load_predictor, load_tokenizer, utterance):
    # tokenize utterance and send model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_predictor.to(device)
    tokenized_utterance = load_tokenizer(utterance, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # apply model
    input_ids, attention_mask = tokenized_utterance['input_ids'].to(device), tokenized_utterance['attention_mask'].to(device)
    output = load_predictor(input_ids, attention_mask)
    probs = output.detach().cpu().numpy()
    pred = (probs >= 0.6).astype(float)

    return int(pred[0, 0])


def get_polarity_prediction(polarity_predictor, polarity_tokenizer, utterance):
    # tokenize utterance and send model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    polarity_predictor.to(device)
    tokenized_utterance = polarity_tokenizer(utterance, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # apply model
    input_ids, attention_mask = tokenized_utterance['input_ids'].to(device), tokenized_utterance['attention_mask'].to(device)
    output = polarity_predictor(input_ids, attention_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1).cpu().numpy()

    return pred.item()


def get_utt_linguistic_features_conv(nlp, utterance, load_predictor, load_tokenizer, polarity_predictor, polarity_tokenizer, batch_idx=0, utt_idx=0, class_dict=dict()):
    '''
        This function stores the linguistic features computed from the given utterance in the dictionnary dict provided in input. 

        @param nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features from the utterance
        @param utt_idx (int): the utterance index in the description dictionary. Default is zero as default dictionary is empty
        @param class_dict (dict): the description dictionary to put utterances, IDs and linguistic features in. Default is an empty dictionary
        @param utterance (str): the utterance to study. It is bound to be an individual of type 'FullText' in the ontology. The syntax used in the values of the output dictionnary make it usable iteratively, thus considering a set of utterances. 

        @return class_values (dict): a dictionnary whose keys are the ontology classes. 
    '''
    # compute linguistic features useful for CEFR class attribution
    doc = nlp(utterance)
    readability_metrics = {
        'flesch_kincaid': flesch_kincaid(doc),
        'cole': coleman_liau_index(doc),
        'mtld': mtld(doc),
        'smog': smog_index(doc, nlp),
        'pronoun_density': pronoun_density(doc),
        'average_word_length': average_word_length(doc),
       'load': get_load_prediction(load_predictor, load_tokenizer, utterance),
        'polarity': get_polarity_prediction(polarity_predictor, polarity_tokenizer, utterance),
    }

    # if the dictionnary doesn't contain any other utterance description, then it is empty so we need to instantiate the keys
    if len(class_dict) == 0:
        keys = ['FullText', 'UtteranceID'] + list(readability_metrics.keys())
        class_dict = {k:[] for k in keys}

    # store full text text and ID
    full_text = utterance.replace(' ', '_')
    class_dict['FullText'].append(full_text)
    class_dict['UtteranceID'].append(f'{batch_idx}_{utt_idx}_utt')

    # readability metrics
    for metric in readability_metrics.keys():
        class_dict[metric].append(readability_metrics[metric])

    return class_dict