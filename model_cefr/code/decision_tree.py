import pandas as pd
import numpy as np
import copy
import numpy as np
from collections import Counter
import nltk
nltk.download('punkt')

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split

from utils import *

train_df = pd.read_csv('../data/cefr_texts_ling_features/train.csv')
val_df = pd.read_csv('../data/cefr_texts_ling_features/validation.csv')
test_df = pd.read_csv('../data/cefr_sentences_ling_features/test.csv')

# remove the text since we just want to play with the features here
train_df = train_df.drop(['text', 'text.1', 'named_entities'], axis=1)
val_df = val_df.drop(['text', 'text.1', 'named_entities'], axis=1)
df = pd.concat([train_df, val_df], keys=['train', 'val'])

df["adjective_ratio"] = df["token_count"] / (0.001+df["adjective_count"])
test_df["adjective_ratio"] = test_df["token_count"] / (0.001+test_df["adjective_count"])


X = df[[
        # 'unique_word_count',
        'named_entity_count',
        # 'noun_ratio',
        # 'verb_ratio',
        # 'adverb_ratio',
        'adjective_ratio',
        # 'dep_ratio',
        'average_word_length',
        # 'average_dependency_distance',
        # 'type_token_ratio',
        # 'average_sentence_length',
        'gunning_fox_index',
        'coleman_liau_index',
        'dale_chall_score',
        'tree_depth', 
        'flesch_kincaid',
        'smog_index',
        'mtld',
        # 'mlc',
        'pronoun_density',
        'connectives_density',
        'has_long_noun_phrases',
        'has_coordinated_clauses',
        'has_nominalizations', 
        'has_variety_of_tenses',
        'has_long_sentence',
        'has_subordinate_clause',
        'has_passive_voice',
        'has_long_noun_phrases',
        'has_embedded_clauses',
        'has_conjunctions',
        'has_negations',
        'has_passive_voice',
        'has_long_noun_phrases',
        'has_advanced_vocabulary',
        'has_coordinated_clauses',
        'has_variety_of_tenses',
        'has_ellipsis',
        'has_figurative_language',
        'has_polysyndeton',
        'has_indirect_speech',
        'morph_complexity.inflectional_ratio',
        # 'morph_complexity.derivational_ratio',
        'morph_complexity.morphological_diversity',
]]

test_set = test_df[[
        # 'unique_word_count',
        'named_entity_count',
        'adjective_ratio',
        # 'dep_ratio',
        'average_word_length',
        # 'average_dependency_distance',
        # 'type_token_ratio',
        # 'average_sentence_length',
        'gunning_fox_index',
        'coleman_liau_index',
        'dale_chall_score',
        'tree_depth', 
        'flesch_kincaid',
        'smog_index',
        'mtld',
        # 'mlc',
        'pronoun_density',
        'connectives_density',
        'has_long_noun_phrases',
        'has_coordinated_clauses',
        'has_nominalizations', 
        'has_variety_of_tenses',
        'has_long_sentence',
        'has_subordinate_clause',
        'has_passive_voice',
        'has_long_noun_phrases',
        'has_embedded_clauses',
        'has_conjunctions',
        'has_negations',
        'has_passive_voice',
        'has_long_noun_phrases',
        'has_advanced_vocabulary',
        'has_coordinated_clauses',
        'has_variety_of_tenses',
        'has_ellipsis',
        'has_figurative_language',
        'has_polysyndeton',
        'has_indirect_speech',
        'morph_complexity.inflectional_ratio',
        # 'morph_complexity.derivational_ratio',
        'morph_complexity.morphological_diversity',
]]

# split data into train and val
y = df['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# setup the model
clf = DecisionTreeClassifier(min_samples_leaf=50, max_depth=5, random_state=65)

# train
clf.fit(X_train, y_train)

from pickle import dump
with open("../models/best_clf.pkl", "wb") as f:
    dump(clf, f, protocol=5)

print("**** TRAIN ***")
y_pred = clf.predict(X_train)
print(Counter(y_pred))
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

print("**** VAL ***")
y_pred = clf.predict(X_val)
print(Counter(y_pred))
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

label_to_num = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
y_val_num = [label_to_num[label] for label in y_val]
y_pred_num = [label_to_num[label] for label in y_pred]
print(mean_absolute_error(y_val_num, y_pred_num))

feature_importances = pd.DataFrame(clf.feature_importances_,
                                index = X_train.columns,
                                columns=['importance']).sort_values('importance', 
                                                                    ascending=False)


print(feature_importances[feature_importances.importance > 0])

print("**** TEST ***")
y_test = test_df['level']#.map({1:'A1', 2:'A2', 3:'B1', 4:'B2', 5:'C1', 6:'C2'}).tolist()
mapping = {'A1':1, 'A2':2, 'B1':3, 'B2':4, 'C1':5, 'C2':6}
y_pred_test = np.array([mapping[label] for label in clf.predict(test_set)]).tolist()
new_df = pd.DataFrame({
    'Text': test_df['text'],
    'Prediction': y_pred_test
})
new_df.to_csv('predictions_cefr_test.csv', index=False)

print(new_df)
print(Counter(y_pred_test))
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))
print(mean_absolute_error(y_test, y_pred_test))

## extract the ontological rules from the decision tree rules

tree_rules = tree.export_text(clf, feature_names=list(X_train))

def explore_tree(tree, root_id, labels, cllab, cbranch=[], result=[]):
    if tree.children_left[root_id] == tree.children_right[root_id]:
        cl = cllab[tree.value[root_id].argmax()]
        result.append((cl, cbranch))
        return result
    else:
        feature = labels[tree.feature[root_id]]
        threshold = tree.threshold[root_id]
        cbranch1 = copy.copy(cbranch)
        cbranch2 = copy.copy(cbranch)
        cbranch1.append({"A": feature, "D": 0, "T": threshold})
        cbranch2.append({"A": feature, "D": 1, "T": threshold})
        explore_tree(tree, tree.children_left[root_id], labels, cllab, cbranch=cbranch1)
        explore_tree(tree, tree.children_right[root_id], labels, cllab, cbranch=cbranch2)
        return result

def findCorrespondingCond(cond, clause):
    for cond2 in clause:
        if cond["A"] == cond2["A"] and cond["T"] == cond2["T"]: return cond2
    return None
   
def simplify(clauses):
    # phase 1, remove oposing elements accross otherwse identical ORs
    something_changed = True
    while something_changed:
        something_changed = False
        for i in range(len(clauses)-1):
            cl1 = clauses[i]
            for j in range(i+1, len(clauses)):
                cl2 = clauses[j]
                identical = []
                opposing = []
                # if they don't have the same size, there is no point
                if len(cl1) != len(cl2): break
                notgood = False
                for cond1 in cl1:
                    cond2 = findCorrespondingCond(cond1, cl2)
                    if cond2 == None: # not poing if we can find a correspinding one
                        notgood = True
                        break
                    if cond1["D"] == cond2["D"]: identical.append(cond1)
                    else: opposing.append((cond1, cond2))
                if notgood: break
                if len(identical) == len(cl1)-1 and len(opposing) == 1:
                    cl1.remove(opposing[0][0])
                    cl2.remove(opposing[0][1])
                    something_changed = True
    # phase 2, remove identical classes or more specific clauses
    something_changed = True
    while something_changed:
        something_changed = False
        toremove=[]
        for i in range(len(clauses)-1):
            cl1 = clauses[i]
            for j in range(i+1, len(clauses)):
                cl2 = clauses[j]
                identical = []
                for cond1 in cl1:
                    cond2 = findCorrespondingCond(cond1, cl2)
                    if cond2 is not None and cond1["D"] == cond2["D"]: identical.append(cond1)
                if len(identical) == len(cl1):
                    if cl2 not in toremove: toremove.append(cl2)
                    something_changed = True
                elif len(identical) == len(cl2):
                    if cl1 not in toremove: toremove.append(cl1)
                    something_changed = True                    
        for tr in toremove:
            clauses.remove(tr)
    # phase 3: removing redundant conditions in clases
    for cl in clauses:
        toremove=[]
        for i in range(len(cl)-1):
            cond1 = cl[i]
            for j in range(i+1, len(cl)):
                cond2 = cl[j]
                if cond1["A"] == cond2["A"]:
                    if cond1["D"] == 0 and cond2["D"] == 0:
                        if cond1["T"] < cond2["T"]: toremove.append(cond2)
                        else: toremove.append(cond1)
                    if cond1["D"] == 1 and cond2["D"] == 1:
                        if cond1["T"] < cond2["T"]: toremove.append(cond1)
                        else: toremove.append(cond2)
        for c in toremove:
            if c in cl: cl.remove(c)
    return clauses
  
## display them in OWL manchester syntax

def showAsOWL(res, data_prop_mapping, manchester_syntax, display=True):
    # collect list of clauses for each class
    cls = {}
    for r in res:
        if r[0] not in cls: cls[r[0]] = []
        cls[r[0]].append(r[1])

    cls_exprs = {k:"" for k in cls.keys()}
    for cl in cls:
        if display:
            print(cl)
        expr = simplify(cls[cl])
        cl_expr = f'{cl} equivalentTo: ' if not manchester_syntax else ''
        for i, clause in enumerate(expr):
            if i != 0: 
                cl_expr += " or "
            cl_expr += "("
            for j, cond in enumerate(clause):
                if j != 0: 
                    cl_expr += " and "
                    if manchester_syntax:
                        cl_expr += "("
                cl_expr += f"{data_prop_mapping[cond['A']]} some xsd:decimal["
                if cond["D"] == 0: 
                    cl_expr+= "<= "
                else: 
                    cl_expr += "> "
                cl_expr += f"{cond['T']} ]"
                if manchester_syntax and j+1 <= len(clause)-1: 
                    cl_expr += ")"
            cl_expr += ")" 
            # if not manchester_syntax else "))"

        cl_expr = f"FullText and \n({cl_expr})" if manchester_syntax else cl_expr

        if display: 
            print(cl_expr)
            print("\n*****************\n")
            
        cls_exprs[cl] = cl_expr

    return cls_exprs


cls = ["A1", "A2", "B1", "B2", "C1", "C2"]
data_prop_mapping = {
    'gunning_fox_index': 'hasFOGIReadabilityScore',
    'smog_index': 'hasSMOGReadabilityScore',
    'pronoun_density': 'hasPronounDensity',
    'dale_chall_score': 'hasDCReadabilityScore',
    'has_conjunctions': 'hasConjunctions',
    'named_entity_count': 'hasNamedEntityCount',
    'adjective_ratio': 'hasAdjectiveRatio',
    'flesch_kincaid': 'hasFleschKincaidReadabilityScore',
    'coleman_liau_index': 'hasColeReadabilityScore',
    'mtld': 'hasMTLDmeasure',
    'average_word_length': 'hasAverageWordLength',
    'morph_complexity.inflectional_ratio': 'hasMorphComplexityInflectionalRatio',
    'morph_complexity.morphological_diversity': 'hasMorphComplexityMorphDiversity',
}

res = explore_tree(clf.tree_, 0, X.columns, cls)
owl_expressions = showAsOWL(res, data_prop_mapping, manchester_syntax=True, display=True)