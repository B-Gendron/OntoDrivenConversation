import spacy
import pandas as pd
import argparse
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from nltk.corpus import wordnet as wn
import datasets
import nltk
import numpy as np

tqdm.pandas()
nltk.download('wordnet')

def calculate_tree_depth(doc):
    def depth(token):
        if not list(token.children):
            return 1
        return 1 + max(depth(child) for child in token.children)
    
    root = [t for t in doc if t.dep_ == "ROOT"][0]
    return depth(root)


def average_dependency_distance(doc):
    distances = []
    for token in doc:
        distance = abs(token.i - token.head.i)
        distances.append(distance)
    return sum(distances) / len(distances)


def dependency_relations_count(doc):
    relations = [token.dep_ for token in doc]
    dep_counts = Counter(relations)
    return sum(dep_counts.values())


def type_token_ratio(doc):
    '''
    Type-Token Ratio (TTR): Measures the ratio of unique words (types) to the total number of words (tokens). Higher lexical diversity often correlates with more advanced language proficiency.
    '''
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0


def unique_word_count(doc):
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    unique_tokens = set(tokens)
    return len(unique_tokens)

def flesch_kincaid(doc):
    sentences = list(doc.sents)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    syllables = sum([len([syll for syll in token.text.lower() if syll in 'aeiouy']) for token in words])

    fk_score = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return fk_score

def gunning_fog_index(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    complex_words = [word for word in words if len(word) > 2 and word.lower() in nlp.vocab.strings]
    fog_index = 0.4 * ((len(words) / len(sentences)) + 100 * (len(complex_words) / len(words)))
    return fog_index

def smog_index(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    complex_words = [word for word in words if len(word) > 2 and word.lower() in nlp.vocab.strings]
    smog_index = 1.0430 * (30 * (len(complex_words) / len(sentences)))**0.5 + 3.1291
    return smog_index

def coleman_liau_index(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_chars = sum(len(token.text) for token in doc if not token.is_punct and not token.is_space)
    sentences = list(doc.sents)
    L = (num_chars / len(words)) * 100
    S = (len(sentences) / len(words)) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8
    return cli

def automated_readability_index(doc):
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    num_chars = sum(len(token.text) for token in doc if not token.is_punct and not token.is_space)
    sentences = list(doc.sents)
    ari = (4.71 * (num_chars / len(words))) + (0.5 * (len(words) / len(sentences))) - 21.43
    return ari

def dale_chall_score(doc, easy_words):
    words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    difficult_words = sum(1 for word in words if word not in easy_words)
    p = (difficult_words / len(words)) * 100
    dc_score = 0.1579 * p + 0.0496 * (len(words) / len(sentences))
    return dc_score

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

def mlc(doc):
    clauses = [list(chunk) for chunk in doc.sents]
    clause_lengths = [len(clause) for clause in clauses]
    return sum(clause_lengths) / len(clause_lengths) if clause_lengths else 0

def sub_index(doc):
    clauses = [list(chunk) for chunk in doc.sents]
    subordinate_clauses = sum(1 for token in doc if token.dep_ in ['advcl', 'relcl', 'csubj', 'ccomp'])
    return subordinate_clauses / len(clauses) if clauses else 0

def pronoun_density(doc):
    pronouns = [token.text.lower() for token in doc if token.pos_ == "PRON"]
    return len(pronouns) / len(doc) if len(doc) > 0 else 0

def connectives_density(doc):
    connectives = ['however', 'therefore', 'moreover', 'consequently', 'thus', 'furthermore', 'nevertheless']
    connectives_count = sum(1 for token in doc if token.text.lower() in connectives)
    return connectives_count / len(doc) if len(doc) > 0 else 0

def morphological_complexity(doc):
    inflectional_count = 0
    derivational_count = 0
    lemma_forms = defaultdict(set)
    
    # example suffixes for derivational morphology (this is a simplification)
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
        return {"inflectional_ratio": 0, "derivational_ratio": 0, "morphological_diversity": 0}

    inflectional_ratio = inflectional_count / total_tokens
    derivational_ratio = derivational_count / total_tokens
    morphological_diversity = len(lemma_forms) / total_tokens
    
    return {
        "inflectional_ratio": inflectional_ratio,
        "derivational_ratio": derivational_ratio,
        "morphological_diversity": morphological_diversity
    }


def has_subordinate_clause(doc):
    return any(token.dep_ == 'mark' for token in doc)


def has_passive_voice(doc):
    return any(token.dep_ == 'auxpass' or token.dep_ == 'nsubjpass' for token in doc)


def has_long_noun_phrases(doc, threshold=3):
    return any(len(np) > threshold for np in doc.noun_chunks)


def has_advanced_vocabulary(doc, easy_words):
    return any(token.text.lower() not in easy_words for token in doc if token.is_alpha)


def has_coordinated_clauses(doc):
    return any(token.dep_ == 'cc' or token.dep_ == 'conj' for token in doc)


def has_nominalizations(doc):
    return any(token.pos_ == 'NOUN' and token.lemma_ != token.text for token in doc)


def has_embedded_clauses(doc):
    return any(token.dep_ == 'relcl' for token in doc)


def has_conjunctions(doc):
    conjunctions = {'however', 'therefore', 'furthermore', 'although', 'because'}
    return any(token.text.lower() in conjunctions for token in doc)


def has_negation(doc):
    return any(token.dep_ == 'neg' for token in doc)


def has_ellipsis(doc):
    # Heuristic for ellipsis detection
    return any(token.is_punct and token.text == '...' for token in doc)


def has_figurative_language(doc):
    simile_markers = {'like', 'as if'}
    return any(token.text.lower() in simile_markers for token in doc)


def has_polysyndeton(doc):
    conjunction_count = sum(1 for token in doc if token.dep_ == 'cc')
    return conjunction_count > 1


def has_long_sentence(doc, threshold=15):
    return len(doc) > threshold


def has_variety_of_tenses(doc):
    tenses = set()
    for token in doc:
        if token.tag_ in {'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:  # various verb forms
            tenses.add(token.tag_)
    return len(tenses) > 1


def has_indirect_speech(doc):
    reporting_verbs = {'said', 'told', 'asked', 'explained'}
    for token in doc:
        if token.lemma_ in reporting_verbs and any(child.dep_ == 'mark' for child in token.children):
            return True
    return False

def get_abstractness_score(token):
    synsets = wn.synsets(token.text)
    if not synsets:
        return 0  # treat as concrete if no synsets are found
    synset = synsets[0]
    if synset.hypernyms():
        hypernym = synset.hypernyms()[0]
        if hypernym.name().startswith('entity.n.01') or hypernym.name().startswith('physical_entity.n.01'):
            return 0  # this is a concrete word
    return 1  # therefore this is an abstract word

def abstractness_concreteness(doc):
    scores = [get_abstractness_score(token) for token in doc if token.is_alpha]
    return np.mean(scores)


def gather_features(text):
    doc = nlp(text)
    
    features = {
        'text': text,
        'token_count': len(doc),
        'sentence_count': len(list(doc.sents)),
        'noun_count': sum(1 for token in doc if token.pos_ == 'NOUN'),
        'verb_count': sum(1 for token in doc if token.pos_ == 'VERB'),
        'adjective_count': sum(1 for token in doc if token.pos_ == 'ADJ'),
        'adverb_count': sum(1 for token in doc if token.pos_ == 'ADV'),
        'named_entities': ', '.join(ent.text for ent in doc.ents),
        'named_entity_count': len(doc.ents),
        'average_word_length': sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0,
        'average_sentence_length': len(doc.text) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
        'tree_depth': calculate_tree_depth(doc),
        'average_dependency_distance': average_dependency_distance(doc),
        'dependency_relations_count': dependency_relations_count(doc),
        'type_token_ratio': type_token_ratio(doc),
        'unique_word_count': unique_word_count(doc),
        'flesch_kincaid': flesch_kincaid(doc),
        'gunning_fox_index': gunning_fog_index(doc),
        'smog_index': smog_index(doc),
        'coleman_liau_index': coleman_liau_index(doc),
        'automated_readability_index': automated_readability_index(doc),
        'dale_chall_score': dale_chall_score(doc, easy_words),
        'mtld': mtld(doc),
        'mlc': mlc(doc),
        'sub_index': sub_index(doc),
        'pronoun_density': pronoun_density(doc),
        'connectives_density': connectives_density(doc),
        'morph_complexity': morphological_complexity(doc),
        'has_subordinate_clause': has_subordinate_clause(doc),
        'has_passive_voice': has_passive_voice(doc),
        'has_long_noun_phrases': has_long_noun_phrases(doc),
        'has_advanced_vocabulary': has_advanced_vocabulary(doc, easy_words),
        'has_coordinated_clauses': has_coordinated_clauses(doc),
        'has_nominalizations': has_nominalizations(doc),
        'has_embedded_clauses': has_embedded_clauses(doc),
        'has_conjunctions': has_conjunctions(doc),
        'has_negations': has_negation(doc),
        'has_ellipsis': has_ellipsis(doc),
        'has_figurative_language': has_figurative_language(doc),
        'has_variety_of_tenses': has_variety_of_tenses(doc),
        'has_polysyndeton': has_polysyndeton(doc),
        'has_long_sentence': has_long_sentence(doc),
        'has_indirect_speech': has_indirect_speech(doc),
        'abstractness_concreteness': abstractness_concreteness(doc)
    }
    
    return features

def put_features_in_dataframe(df):

    features_list = df['text'].progress_apply(gather_features)
    features_df = pd.json_normalize(features_list)
    df = pd.concat([df, features_df], axis=1)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str)
    args = parser.parse_args()
    
    nlp = spacy.load('en_core_web_sm')
    DATA_FOLDER = args.data_folder

    with open('../resources/easy_words.txt', 'r') as file:
        easy_words = set(word.strip().lower() for word in file.readlines())

    test_dataset = datasets.load_from_disk('../../fine-tuning/data/onto_prefix_abc_cefr')
    test_dataset = test_dataset['test']
    test_df = pd.DataFrame(test_dataset)
    test_df.rename(columns={'utterance': 'text'}, inplace=True)
    test_df.to_csv('../data/cefr_test_ontogpt/test.csv')

    data = pd.read_csv('../data/cefr_test_ontogpt/test.csv')
    data_with_linguistic_features = put_features_in_dataframe(data)
    data_with_linguistic_features.to_csv('../data/cefr_test_ontogpt_ling_features/test.csv', index=False)