import os
from owlready2 import *
from termcolor import colored
import torch
import numpy as np
from numpy import random as rd
import random

import warnings
warnings.filterwarnings('ignore')

# set default tensor type
torch.set_default_dtype(torch.float16)

cefr_starters = [
    "Write a short story about a dragon and a knight who become friends.",
    "Explain how blockchain technology works.",
    "What were the main causes and effects of the French Revolution?",
    "Describe the process of photosynthesis.",
    "What is the meaning of life according to existential philosophy?",
    "Analyze the themes of '1984' by George Orwell.",
    "Write a poem about the ocean at sunrise.",
    "What are the advantages and disadvantages of nuclear energy?",
    "Summarize the plot of 'Pride and Prejudice'.",
    "Describe the major differences between classical and operant conditioning.",
    "What are the key features of quantum computing?",
    "Explain the significance of the Theory of Relativity.",
    "Write a dialogue between two characters discussing artificial intelligence.",
    "What are the economic impacts of climate change?",
    "Describe the main components of a healthy diet.",
    "Explain the concept of supply and demand in economics.",
    "What are the ethical implications of genetic engineering?",
    "Describe the cultural impact of the Beatles on modern music.",
    "Write a news article about a recent scientific discovery.",
    "What are the primary symptoms and treatments for depression?",
    "Explain the steps involved in the water cycle.",
    "What are the fundamental principles of democracy?",
    "Write a speech advocating for renewable energy sources.",
    "Describe the process of mitosis in cell division.",
    "What are the historical roots of the conflict in the Middle East?",
    "Explain how the internet has changed global communication.",
    "Write a review of a popular movie or TV show.",
    "What are the causes and effects of deforestation?",
    "Describe the role of women in ancient Greek society.",
    "What are the key factors in maintaining mental health?"
]

emo_starters = [
    "How do you feel about rainy days?",
    "Describe your favorite memory.",
    "What’s your opinion on early mornings?",
    "Tell me about a time you felt truly happy.",
    "What comes to mind when you think of failure?",
    "Do you enjoy meeting new people?",
    "How do you handle criticism?",
    "What do you think of surprises?",
    "Share your thoughts on family gatherings.",
    "What do you usually do when you’re alone?",
    "How do you feel about public speaking?",
    "What’s your opinion on taking risks?",
    "Tell me about a time you felt disappointed.",
    "What do you think about people who work too much?",
    "How do you feel when someone doesn’t keep a promise?",
    "What’s your reaction to unexpected compliments?",
    "How do you usually feel on your birthday?",
    "What do you think of long commutes?",
    "Tell me about something you regret.",
    "How do you feel when you accomplish a goal?",
    "What’s your opinion on social media?",
    "What do you think about trying new food?",
    "How do you feel when someone helps you unexpectedly?",
    "Tell me about a time you felt proud of yourself.",
    "What do you think about going to the movies alone?",
    "How do you feel when someone disagrees with you?",
    "What’s your opinion on forgiveness?",
    "How do you react when someone interrupts you?",
    "What comes to mind when you think of home?",
    "How do you feel about changes in your routine?"
]

# -----------------------------------------------------------------------------------------
# General purpose auxiliary functions
# -----------------------------------------------------------------------------------------

def get_datetime():
    '''
        This function gets the current date time and returns it as a string.

        @returns dt_string (str): current time formatted in a string
    '''
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string

def activate_gpu(force_cpu=False):
    '''
        A function to return the right device depending on GPU availability.

        Parameters:
            force_cpu (optional, str): If set to True, the device will be 'cpu' even if 'cuda' or 'mps' are available. Default is False.
    '''
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            print('DEVICE = ', colored('CPU', "blue"))
    return device


# -----------------------------------------------------------------------------------------
# Fine-tuning utils
# -----------------------------------------------------------------------------------------

def parse_indexes(levels_dict, ontology_concept):
    '''
        This auxiliary function parses and retrieves the utterance indexes from the readability level dictionnary when an utterance is reffered to using the ontology identifier. It return a dictionary with the same format than the input, containing only the indexes instead of the complete identifier.

        Parameters:
            levels_dict (dict):          the dict from the ontology individual search.
        
        Returns:
            levels_indexes_dict (dict): a dict that maps the readability levels to the associated utterance indexes in the current dialog.
    '''
    levels_indexes_dict = {}

    # iterate through the input dict
    for k in levels_dict.keys():
        for v in levels_dict[k]:
            # parse the utterance full name to retrieve its index
            splitted_v = re.split(r'[_\.]', v)[::-1]
            index = splitted_v[1] if ontology_concept == 'NextIntention' else splitted_v[2] # the 3rd element when individual_592.592_598_utt_21271711 is splitted by _ is 598, which is the utterance index
            # store the readability class for the current index
            levels_indexes_dict[int(index)] = k

    result_dict = dict(sorted(levels_indexes_dict.items()))

    return result_dict


def get_proficiency_levels(indiv_path):
    '''
    Retrieves and maps the proficiency levels for a given ontology individual.
    
    Parameters:
        indiv_path (str): Path to the ontology individual.
    
    Returns:
        labels (list): A list of CEFR proficiency labels (A1, A2, B1, B2, C1 or C2)
    '''
    # get labels mapping for different readability level
    proficiency_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()
    # crop first element as it simply corresponds to the class occurence
    utterance_levels = {
        'A1':       [str(i) for i in individual.search(is_a=individual.A1)[1:]],
        'A2':       [str(i) for i in individual.search(is_a=individual.A2)[1:]],
        'B1':       [str(i) for i in individual.search(is_a=individual.B1)[1:]],
        'B2':       [str(i) for i in individual.search(is_a=individual.B2)[1:]],
        'C1':       [str(i) for i in individual.search(is_a=individual.C1)[1:]],
        'C2':       [str(i) for i in individual.search(is_a=individual.C2)[1:]]
    }
    utterance_levels = parse_indexes(utterance_levels, 'ProficiencyLevel')
    labels = [proficiency_mapping[v] for v in utterance_levels.values()]

    individual.destroy()

    return labels


def get_emotion_labels(indiv_path):
    '''
    Retrieves and maps the proficiency levels for a given ontology individual.
    
    Parameters:
    indiv_path (str): Path to the ontology individual.
    
    Returns:
    labels (list): A list of CEFR proficiency labels (A1, A2, B1, B2, C1 or C2)
    rough_labels (list): A list of simplified proficiency labels (A, B or C)
    '''
    # get labels mapping for different readability level
    proficiency_mapping = {'LoadedPositive': 0, 'LoadedNegative': 1, 'LoadedNeutral': 2, 'NonLoadedPositive': 3, 'NonLoadedNegative': 4, 'NonLoadedNeutral': 5}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()

    utterance_levels = {
        'LoadedPositive':       [str(i) for i in individual.search(is_a=individual.LoadedPositiveUtterance)[1:]],
        'LoadedNegative':       [str(i) for i in individual.search(is_a=individual.LoadedNegativeUtterance)[1:]],
        'LoadedNeutral':        [str(i) for i in individual.search(is_a=individual.LoadedNeutralUtterance)[1:]],
        'NonLoadedPositive':    [str(i) for i in individual.search(is_a=individual.NonLoadedPositiveUtterance)[1:]],
        'NonLoadedNegative':    [str(i) for i in individual.search(is_a=individual.NonLoadedNegativeUtterance)[1:]],
        'NonLoadedNeutral':     [str(i) for i in individual.search(is_a=individual.NonLoadedNeutralUtterance)[1:]],
    }
    utterance_levels = parse_indexes(utterance_levels, 'NextIntention')
    labels = [proficiency_mapping[v] for v in utterance_levels.values()]

    individual.destroy()

    return labels


def random_prompt(concept, classes):
    '''
    This auxiliary function creates a prompt that asks for a sentence belonging to a specified class within a given concept. It is currently used for sentence lengths but is designed for more general purposes.

    Parameters:
        concept (str): The name of the ontology concept we want to learn, which is divided into the given classes.
        classes (list): A list of strings providing the names of all the classes.

    Returns:
        tuple:
            str: The randomly selected prompt to use for generation.
            int: The index of the corresponding class that is asked in the prompt.
    '''
    p = rd.uniform()
    n = len(classes)
    question = random.choice(cefr_starters)

    for k in range(1, n+1):
        if (k-1)/n < p < k/n:
            # You will be asked a question and given an instruction to format your answer between brackets. Answer directly after the brackets, and start your answer with a word, not a figure or any special character. You must provide elaborated sentences, not only one word. If you are asked something related to feelings, answer as a human would answer in such situation.

            ## This prompt is suitable for llama-like models, note for qwen models 
            # prompt = f'''
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            # You will be asked a question and given an instruction to format your answer between brackets. Answer directly after the brackets, and start your answer with a word, not a figure. You must provide elaborated sentences, not only one word. If you are asked something related to feelings, answer as a human would answer in such situation. <|eot_id|><|start_header_id|>user<|end_header_id|>

            # {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            # [{concept}: {classes[k-1]}]
            # ''' 

            # this prompt is tailored to qwen models
            # prompt = f'''
            # <|im_start|>system\n
            # You will be asked a question and given an instruction to format your answer between brackets. Answer directly after the brackets, and start your answer with a word, not a figure. You must provide elaborated sentences, not only one word. If you are asked something related to feelings, answer as a human would answer in such situation.<|im_end|>\n
            # <|im_start|>user\n
            # {question}.<|im_end|>
            # \n<|im_start|>assistant\n
            # [{concept}: {classes[k-1]}]
            # ''' 

            # this prompt is tailored for gemma3 model
            prompt = f'''
            <start_of_turn>user
            Answer to the question following the constraint given into brackets, directly after the brackets. Do not repeat the question.\n{question}.<end_of_turn>
            <start_of_turn>model
            [{concept}: {classes[k-1]}]'''

            return prompt, k-1
        

def prompt_for_strategy(prefix):
    question = random.choice(cefr_starters+emo_starters)
    # You will be asked a question and given an instruction to format your answer between brackets. Answer directly after the brackets, and start your answer with a word, not a figure or any special character. You must provide elaborated sentences, not only one word. If you are asked something related to feelings, answer as a human would answer in such situation.

    ## This prompt is suitable for llama-like models, not for qwen models 
    # prompt = f'''
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    # You will be asked a question and given an instruction to format your answer between brackets. Answer directly after the brackets, and start your answer with a word, not a figure. You must provide elaborated sentences, not only one word. If you are asked something related to feelings, answer as a human would answer in such situation. <|eot_id|><|start_header_id|>user<|end_header_id|>

    # {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # [{prefix}]
    # ''' 

    # this prompt is tailored to qwen models
    prompt = f'''
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You will be asked a question and given an instruction to format your answer between brackets. Generate text directly after the brackets.\n
    {question}\n
    [{prefix}]
    ''' 

    # this prompt is tailored for gemma3 model
    # prompt = f'''
    # <start_of_turn>user
    # Answer to the question following the constraint given into brackets, directly after the brackets. Do not repeat the question.
    # {question}.<end_of_turn>
    # <start_of_turn>model
    # [{prefix}]
    # '''

    return prompt


def pad_tensor(args, tensor, padding_length=20):
    '''
    Pads a 1-dimensional tensor with ones to a specified padding length.

    Parameters:
        tensor (torch.Tensor): The input 1-dimensional tensor to pad.
        padding_length (int): The length to pad the tensor to. Default is 20.

    Returns:
        torch.Tensor: The padded tensor.
    '''
    current_length = tensor.size(0)
    
    if current_length >= padding_length:
        return tensor[:padding_length]
    
    else:
        pad_length = padding_length - current_length
        padding_tensor = torch.ones(pad_length, dtype=tensor.dtype, device=args['device'])
        padded_tensor = torch.cat((tensor, padding_tensor), dim=0)

        return padded_tensor

def generate_from_random_prompts(args):
    '''
    This function generates text using a model and tokenizer provided in the arguments. It selects prompts randomly from specified classes of a concept, performs text generation, and returns the generated text along with corresponding labels and token ids.

    Parameters:
        args (dict): A dictionary of arguments containing, among others:
            - 'tokenizer' (Tokenizer): The tokenizer to use for encoding and decoding text.
            - 'gen_model' (Model): The model to use for text generation.
            - 'batch_size' (int): The number of text samples to generate.
            - 'device' (str): The device to run the model on (e.g., 'cpu' or 'cuda').
            - 'max_new_tokens' (int): The maximum number of new tokens to generate.

    Returns:
        tuple: A tuple containing:
            - list: A list of integers representing the labels of the generated text.
            - list: A list of strings representing the generated text.
            - list: A list of tensor ids representing the token ids of the generated text.
    '''
    # concept = 'ReadabilityLevel'
    # classes = ['EasyReadableText', 'StandardReadableText', 'HardlyReadableText']
    concept = 'FullText'
    classes = ['ShortFullText', 'LongFullText']

    batch_labels, batch_generations, batch_ids = [], [], []

    tokenizer = args['tokenizer']
    model = args['gen_model']
    for i in range(args['batch_size']):
        # get a randomly selected prompt (uniform law)
        prompt, label = random_prompt(concept, classes)
        batch_labels.append(label)

        # perform generation
        prompt = tokenizer(prompt, return_tensors="pt").to(args['device'])
        output = model.generate(**prompt, max_new_tokens=args['max_new_tokens'], repetition_penalty=1.5)[0] # prompt + gen

        # format the
        result = tokenizer.decode(output, skip_special_tokens=True)
        generation = result[result.find(':')+1:result.find('\n')]
        output_ids = get_and_pad_ids(tokenizer(generation, return_tensors="pt").to(args['device'])['input_ids'], args, padding_length=40)
        batch_ids.append(output_ids)
        # print(f'Sample {i}: \t | Class {label} | \t {generation}')

        # store result
        batch_generations.append(generation)

    return batch_labels, batch_generations, batch_ids

def get_and_pad_ids(output, args, padding_length=40):
    '''
    This function pads the token ids to the specified padding length with zeros if they are shorter, or truncates them if they are longer.

    Parameters:
        output (torch.Tensor): The tensor containing token ids to be padded or truncated.
        args (dict): A dictionary of arguments containing:
            - 'device' (str): The device to run the operation on (e.g., 'cpu' or 'cuda').
        padding_length (int, optional): The length to pad or truncate the token ids to. Default is 40.

    Returns:
        torch.Tensor: The tensor containing token ids padded or truncated to the specified length.
    '''
    current_length = output.shape[1]
    
    if current_length >= padding_length:
        return output[:, :padding_length]
    
    padding_size = padding_length - current_length
    padding = torch.zeros((1, padding_size), dtype=output.dtype, device=args['device'])
    
    padded_output = torch.cat((output, padding), dim=1)
    return padded_output


def create_batch_individual(batch_index, file_path, experiment, repository, task='cefr'):
    '''
    This function runs the external shell script `call_ontology.sh` to create an ontology individual for current batch.

    Parameters:
        batch_index (int): The index of the batch to be processed.
        file_path (str): The file path where the batch generated data has been previously stored.
        experiment (str): The name of the experiment.
    '''
    # define the command
    command = [f"../call_ontology_{task}.sh", str(batch_index), file_path, experiment, repository]

    # running the command using subprocess
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")