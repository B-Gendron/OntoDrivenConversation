from utils import create_batch_individual, get_proficiency_levels, get_emotion_labels
from logging_utils import save_batch_generations, save_batch_generations_strategy
import numpy as np

# aux function to retrieve trues and preds labels
def get_trues_and_preds_cefr(completions, asked, args, **kwargs):
    file_path = save_batch_generations(asked, completions, batch_index=0, experiment=args['experiment'])
    create_batch_individual(batch_index=0, file_path=file_path, experiment=args['experiment'], repository='rlppo-sandbox', task='cefr')
    # get the inferred generation readability (=proficiency) levels
    generations_rl = get_proficiency_levels(f"../rdf/individual_batch_0_{args['experiment']}.rdf")

    return asked, generations_rl

def get_trues_and_preds_emo(completions, asked, args, **kwargs):
    file_path = save_batch_generations(asked, completions, batch_index=0, experiment=args['experiment'])
    create_batch_individual(batch_index=0, file_path=file_path, experiment=args['experiment'], repository='rlppo-sandbox', task='emo')
    # get the inferred generation polarity classes
    generations_pc = get_emotion_labels(f"../rdf/individual_batch_0_{args['experiment']}.rdf")

    return asked, generations_pc

def get_trues_and_preds_strategy(completions, asked, args, **kwargs):
    file_path = save_batch_generations_strategy(asked, completions, batch_index=0, experiment=args['experiment'])
    # call the CEFR ontology to evaluate the proficiency level
    create_batch_individual(batch_index=0, file_path=file_path, experiment=args['experiment'], repository='rlppo-sandbox', task='cefr')
    proficiency_levels = get_proficiency_levels(f"../rdf/individual_batch_0_{args['experiment']}.rdf")

    # call the polarity profile ontology to evaluate the polarity profile
    create_batch_individual(batch_index=0, file_path=file_path, experiment=args['experiment'], repository='rlppo-sandbox', task='emo')
    polarity_profiles = get_emotion_labels(f"../rdf/individual_batch_0_{args['experiment']}.rdf")

    return asked, proficiency_levels, polarity_profiles


# for emotion labels, the intuition to compute the distance is to add one if we pick the wrong label among Loaded/NonLoaded, and to add 1 or 2 considering polarities are ordered like Neg/Neut/Pos (e.g.: dist+2 if Neg instead of Pos, dist+1 if Neut instead of Pos). We know the ordering of the labels from `get_emotion_labels`, so we encode this distance computation strategy in a numpy array. 
distances = np.array([
                    [0, 2, 1, 1, 3, 2],
                    [0, 0, 1, 3, 1, 2],
                    [0, 0, 0, 2, 2, 1],
                    [0, 0, 0, 0, 2, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0]
                    ])

polarity_profile_distances = distances + distances.T


## auxiliary function to compute the reward in the case of the interview conversation strategy

proficiency_profile_label_mapping = {
    'A1':0,
    'A2':1,
    'B1':2,
    'B2':3,
    'C1':4,
    'C2':5
}
# relies on the emo mapping {'LoadedPositive': 0, 'LoadedNegative': 1, 'LoadedNeutral': 2, 'NonLoadedPositive': 3, 'NonLoadedNegative': 4, 'NonLoadedNeutral': 5}
# first element of the list is 0=NonLoaded, 1=Loaded
# second element of the list is 0=Neg, 1=Neutral, 2=Pos
polarity_profile_label_mapping = {
    0:[1, 2],
    1:[1, 0],
    2:[1, 1],
    3:[0, 2],
    4:[0, 0],
    5:[0, 1]
}


# REWARD FUNCTIONS:

def one_hot_reward(completions, asked, epoch, args, **kwargs):
    trues, preds = get_trues_and_preds_cefr(completions, asked, args)
    return [float(t==p) for t, p in zip(trues, preds)]

def distance_reward(completions, asked, epoch, args, **kwargs):
    trues, preds = get_trues_and_preds_cefr(completions, asked, args)
    return [float(abs(t-p)) for t, p in zip(trues, preds)]

def inverse_distance_reward(completions, asked, epoch, args, eps=0.001, **kwargs):
    trues, preds = get_trues_and_preds_cefr(completions, asked, epoch, args)
    return [1/(float(abs(t-p))+eps) for t, p in zip(trues, preds)]

def make_squared_reward_function(args):
    def squared_distance_reward(completions, asked, max_dist=5, **kwargs):
        trues, preds = get_trues_and_preds_cefr(completions, asked, args)
        return [1 - (float(abs(t-p))**2)/(max_dist**2) for t, p in zip(trues, preds)]
    return squared_distance_reward

# REWARDS THAT ARE ACTUALLY USED IN GRPO TRAINING

# CEFR: make a wrapper function to comply with hf trl library constraints
def make_reward_function_cefr(args, alpha=0.5):
    def cefr_ponderated_opposite_distance_reward(completions, asked, **kwargs):
        trues, preds = get_trues_and_preds_cefr(completions, asked, args)
        return [(-alpha*float(abs(t-p))) for t, p in zip(trues, preds)]
    return cefr_ponderated_opposite_distance_reward

# EMO: make a wrapper function to comply with hf trl library constraints
def make_reward_function_emo(args, alpha=0.5):
    def polarity_ponderated_opposite_distance_reward(completions, asked, **kwargs):
        trues, preds = get_trues_and_preds_emo(completions, asked, args)
        # return the list of corresponding values in the distance matrix
        distances_list = [-alpha*polarity_profile_distances[t, p] for t, p in zip(trues, preds)]
        print("Rewards: ", distances_list)
        return distances_list
    return polarity_ponderated_opposite_distance_reward

# STRATEGY: experiments for ISWC paper
def make_reward_function_strategy(args, alpha=0.5):
    def all_labels_ponderated_opposite_distance_reward(completions, asked, **kwargs):
        trues, preds_cefr, preds_emo = get_trues_and_preds_strategy(completions, asked, args, **kwargs)

        distances_list = [0 for _ in range(len(trues))]
        preds_emo_mapped = [polarity_profile_label_mapping[pred] for pred in preds_emo]
        for i in range(len(trues)):
            asked_labels_dict = trues[i]
            # if we put control on proficiency, then we have to check how far we are from the required label
            if 'ProficiencyLevel' in asked_labels_dict.keys():
                distances_list[i] += abs(asked_labels_dict['ProficiencyLevel'] - preds_cefr[i])

            # if the load is controlled, we check the first element of preds_emo_mapped
            if 'Load' in asked_labels_dict.keys():
                distances_list[i] += 1 - (asked_labels_dict['Load'] == preds_emo_mapped[i][0])

            # if the polarity is controlled, we compute the distance between the required polarity and the second element of preds_emo_mapped
            if 'Polarity' in asked_labels_dict.keys():
                distances_list[i] += abs(asked_labels_dict['Polarity'] - preds_emo_mapped[i][1])
        
        return [-alpha*d for d in distances_list]
    return all_labels_ponderated_opposite_distance_reward