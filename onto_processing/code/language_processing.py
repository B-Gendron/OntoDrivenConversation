from urllib.parse import quote
import lftk
import torch

# from other scripts
from utils import *

def concatenate(iterable, sep=""):
    sentence = iterable[0]
    for word in iterable[1:]:
        sentence += (sep + word)
    return sentence


def flesch_kincaid(doc):
    sentences = list(doc.sents)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    syllables = sum([len([syll for syll in token.text.lower() if syll in 'aeiouy']) for token in words])

    if len(words) != 0 and len(sentences) != 0:
        fk_score = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        return fk_score
    else:
        return 0.0

def gunning_fog_index(doc, nlp):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    complex_words = [word for word in words if len(word) > 2 and word.lower() in nlp.vocab.strings]
    if len(words) != 0 and len(sentences) != 0:
        fog_index = 0.4 * ((len(words) / len(sentences)) + 100 * (len(complex_words) / len(words)))
        return fog_index
    else:
        return 0.0

def coleman_liau_index(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_chars = sum(len(token.text) for token in doc if not token.is_punct and not token.is_space)
    sentences = list(doc.sents)
    if len(words) != 0:
        L = (num_chars / len(words)) * 100
        S = (len(sentences) / len(words)) * 100
        cli = 0.0588 * L - 0.296 * S - 15.8
        return cli
    else:
        return 0.0

def smog_index(doc, nlp):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    complex_words = [word for word in words if len(word) > 2 and word.lower() in nlp.vocab.strings]
    if len(sentences) != 0:
        smog_index = 1.0430 * (30 * (len(complex_words) / len(sentences)))**0.5 + 3.1291
        return smog_index
    else:
        return 0.0

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
    lemma_forms = dict(set)
    
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


def get_utt_linguistic_features(nlp, utterance, dial_idx=0, utt_idx=0, class_dict=dict()):
    '''
        This function stores the linguistic features computed from the given utterance in the dictionnary dict provided in input. 

        Args:
            nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features from the utterance
            utt_idx (int): the utterance index in the description dictionary. Default is zero as default dictionary is empty
            class_dict (dict): the description dictionary to put utterances, IDs and linguistic features in. Default is an empty dictionary
            utterance (str): the utterance to study. It is bound to be an individual of type 'FullText' in the ontology. The syntax used in the values of the output dictionnary make it usable iteratively, thus considering a set of utterances. 

        Returns:
            class_values (dict): a dictionnary whose keys are the ontology classes. 
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
    class_dict['FullTextID'].append(f'{dial_idx}_{utt_idx}_utt_{concatenate([str(norm)[0] for norm in token_norms[:8]])}') # for the last part of the ID, keep only the 8 first characters

    # store token and lemma texts and IDs
    class_dict['Token'].extend([token.text for token in doc])
    class_dict['Lemma'].extend([token.lemma_ for token in doc])
    class_dict['TokenID'].extend([f'{dial_idx}_{utt_idx}_{i}_tok_{token.norm}' for i, token in enumerate(doc)])
    class_dict['LemmaID'].extend([f'{dial_idx}_{utt_idx}_{i}_lem_{token.lemma}' for i, token in enumerate(doc)])

    # POS and tags
    class_dict['TokenPOS'].extend([quote(token.pos_) for token in doc])
    class_dict['TokenTag'].extend([quote(token.tag_) for token in doc])

    # readability metrics
    for metric in ['fkgl', 'fogi', 'smog', 'cole', 'auto']:
        class_dict[metric].append(readability_metrics[metric])

    return class_dict



def get_utt_linguistic_features_cefr(nlp, utterance, easy_words, dial_idx=0, utt_idx=0, class_dict=dict()):
    '''
        This function stores the linguistic features computed from the given utterance in the dictionnary dict provided in input. 

        Args:
            nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features from the utterance
            utt_idx (int): the utterance index in the description dictionary. Default is zero as default dictionary is empty
            class_dict (dict): the description dictionary to put utterances, IDs and linguistic features in. Default is an empty dictionary
            utterance (str): the utterance to study. It is bound to be an individual of type 'FullText' in the ontology. The syntax used in the values of the output dictionnary make it usable iteratively, thus considering a set of utterances. 

        Returns:
            class_values (dict): a dictionnary whose keys are the ontology classes. 
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
    class_dict['FullTextID'].append(f'{dial_idx}_{utt_idx}_utt_{concatenate([str(norm)[0] for norm in token_norms[:8]])}') # for the last part of the ID, keep only the 8 first characters

    # store token and lemma texts and IDs
    class_dict['Token'].extend([token.text for token in doc])
    class_dict['TokenID'].extend([f'{dial_idx}_{utt_idx}_{i}_tok_{token.norm}' for i, token in enumerate(doc)])

    # readability metrics
    for metric in readability_metrics.keys():
        class_dict[metric].append(readability_metrics[metric])

    return class_dict


def get_utt_linguistic_features_conv(nlp, utterance, batch_idx=0, utt_idx=0, class_dict=dict()):
    '''
        This function stores the linguistic features computed from the given utterance in the dictionnary dict provided in input. 

        Args:
            nlp (spacy.lang.en.English): the spacy model used to retrieve linguistic features from the utterance
            utt_idx (int): the utterance index in the description dictionary. Default is zero as default dictionary is empty
            class_dict (dict): the description dictionary to put utterances, IDs and linguistic features in. Default is an empty dictionary
            utterance (str): the utterance to study. It is bound to be an individual of type 'FullText' in the ontology. The syntax used in the values of the output dictionnary make it usable iteratively, thus considering a set of utterances. 

        Returns:
            class_values (dict): a dictionnary whose keys are the ontology classes. 
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

def get_load_prediction(load_predictor, load_tokenizer, utterance):
    '''
    Predicts the load (mental load) of an utterance using the provided model and tokenizer.
    
    Args:
        load_predictor: Model to predict the load.
        load_tokenizer: Tokenizer for preprocessing the utterance.
        utterance: Input text utterance.
    
    Returns:
        int: Predicted load value (0 or 1).
    '''
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
    '''
    Predicts the polarity of an utterance using the provided model and tokenizer.
    
    Args:
        polarity_predictor: Model to predict the polarity.
        polarity_tokenizer: Tokenizer for preprocessing the utterance.
        utterance: Input text utterance.
    
    Returns:
        int: Predicted polarity class (0=negative, 1=neutral, or 2=positive).
    '''
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


def get_utt_linguistic_features_emo(utterance, load_predictor, polarity_predictor, load_tokenizer, polarity_tokenizer, utt_idx, class_dict=dict()):
    '''
    Extracts linguistic features and emotion predictions for an utterance and organizes them in a dictionary.
    
    Args:
        utterance: Input text utterance.
        load_predictor: Load classifier.
        polarity_predictor: Polarity classifier.
        load_tokenizer: Tokenizer for the load predictor.
        polarity_tokenizer: Tokenizer for the polarity predictor.
        utt_idx: Index of the utterance.
        class_dict: Optional dictionary to store the features and predictions.
    
    Returns:
        dict: A dictionary with linguistic features and emotion predictions.
    '''
    # compute linguistic features useful for CEFR class attribution
    emotion_class = {
        'load': get_load_prediction(load_predictor, load_tokenizer, utterance),
        'polarity': get_polarity_prediction(polarity_predictor, polarity_tokenizer, utterance),
    }

    # replace space by _ to have a correct namespace for the ontology
    full_text = utterance.replace(' ', '_')

    # if the dictionnary doesn't contain any other utterance description, then it is empty so we need to instantiate the keys
    if len(class_dict) == 0:
        keys = ['FullText', 'FullTextID'] + list(emotion_class.keys())
        class_dict = {k:[] for k in keys}

    # store full text text and ID
    class_dict['FullText'].append(full_text)
    class_dict['FullTextID'].append(f'{utt_idx}_utt')

    # emotion class info
    for metric in emotion_class.keys():
        class_dict[metric].append(emotion_class[metric])

    return class_dict