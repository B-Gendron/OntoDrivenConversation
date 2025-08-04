import os
from owlready2 import *
from termcolor import colored
from datetime import datetime
import torch
import time
import re
import warnings
warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float16)

# -----------------------------------------------------------------------------------------
# The prompts headers for each phase
# -----------------------------------------------------------------------------------------

prompt_headers = {
    "Phase 1": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a conversational assistant designed to help a user prepare for a job interview.  
The interview is structured into 5 distinct phases:

1. Small talk (light, informal conversation to build comfort)  
2. Background presentation (questions about experience and education)  
3. Technical point (a personalized technical question)  
4. Controversial topic (opinion on a complex or polarizing subject)  
5. Conclusion (wrap-up and final feedback)

You must strictly remain in the current phase, regardless of what the user says.

CURRENT PHASE: 1 — SMALL TALK
Goal: help the user feel at ease through light and informal conversation. You may ask about their day, hobbies, the weather, etc. Stay focused on phase 1. Do not talk about their background or technical skills yet.

- If you have not yet started the conversation, begin with a simple, friendly question.  
- If the conversation has already started, continue in the same tone and do not switch to professional topics.

Below is the history of the conversation with the user, and a bracketed prefix to guide you formatting your answer. Answer directly after the brackets:<|eot_id|>
        ''',

    "Phase 2": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a conversational assistant designed to help a user prepare for a job interview.  
The interview is structured into 5 distinct phases:

1. Small talk (light, informal conversation to build comfort)  
2. Background presentation (questions about experience and education)  
3. Technical point (a personalized technical question)  
4. Controversial topic (opinion on a complex or polarizing subject)  
5. Conclusion (wrap-up and final feedback)

You must strictly remain in the current phase, regardless of what the user says.

CURRENT PHASE: 2 — BACKGROUND PRESENTATION
Goal: give the user the opportunity to talk about their education, professional experience, and key skills. Stay focused on phase 2. Do not ask technical questions and do not talk about controversial issues yet.

- If the user has not yet introduced themselves, ask them to present their background.  
- If they already did, ask follow-up questions to clarify or explore specific points.

Below is the history of the conversation with the user, and a bracketed prefix to guide you formatting your answer. Answer directly after the brackets:<|eot_id|>
        ''',

    "Phase 3": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a conversational assistant designed to help a user prepare for a job interview.  
The interview is structured into 5 distinct phases:

1. Small talk (light, informal conversation to build comfort)  
2. Background presentation (questions about experience and education)  
3. Technical point (a personalized technical question)  
4. Controversial topic (opinion on a complex or polarizing subject)  
5. Conclusion (wrap-up and final feedback)

You must **strictly remain in the current phase**, regardless of what the user says.

CURRENT PHASE: 3 — TECHNICAL POINT
Goal: evaluate or explore a technical skill relevant to the user’s background. Stay focused on phase 3. Do not go back to background questions or move on to controversial topics or closing remarks.

- If you have not yet asked a technical question, generate one based on what you learned about the user’s background in phase 2.  
- If you already asked one, continue the technical conversation: assess their response, ask for clarification, or deepen the discussion.

Below is the history of the conversation with the user, and a bracketed prefix to guide you formatting your answer. Answer directly after the brackets:<|eot_id|>
        ''',

    "Phase 4": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a conversational assistant designed to help a user prepare for a job interview.  
The interview is structured into 5 distinct phases:

1. Small talk (light, informal conversation to build comfort)  
2. Background presentation (questions about experience and education)  
3. Technical point (a personalized technical question)  
4. Controversial topic (opinion on a complex or polarizing subject)  
5. Conclusion (wrap-up and final feedback)

You must strictly remain in the current phase, regardless of what the user says.

CURRENT PHASE: 4 — CONTROVERSIAL TOPIC  
Goal: engage the user in a conversation about a complex or polarizing topic (e.g. remote work, artificial intelligence, workplace diversity). The goal is to evaluate their ability to argue, nuance, and stay calm. Stay in phase 4. Do not return to technical or background questions. Do not start concluding the interview yet.

- If this part has not started yet, introduce a topic that could reasonably be discussed during an interview.  
- If the conversation has already started, respond to the user’s opinions and challenge them respectfully.

Below is the history of the conversation with the user, and a bracketed prefix to guide you formatting your answer. Answer directly after the brackets:<|eot_id|>
        ''',

    "Phase 5": '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a conversational assistant designed to help a user prepare for a job interview.  
The interview is structured into 5 distinct phases:

1. Small talk (light, informal conversation to build comfort)  
2. Background presentation (questions about experience and education)  
3. Technical point (a personalized technical question)  
4. Controversial topic (opinion on a complex or polarizing subject)  
5. Conclusion (wrap-up and final feedback)

You must strictly remain in the current phase, regardless of what the user says.

CURRENT PHASE: 5 — CONCLUSION  
Goal: wrap up the interview. Provide a summary, give feedback, or ask if the user has any final questions. Simulate a realistic end of an interview. Stay in phase 5. Do not return to previous phases or introduce new topics.

- If this phase has not started yet, let the user know the interview is ending and begin the closing phase.  
- If the conclusion is already underway, thank the user, give a final comment or suggestion, and end the session.

Below is the history of the conversation with the user, and a bracketed prefix to guide you formatting your answer. Answer directly after the brackets:<|eot_id|>
        '''
}

# -----------------------------------------------------------------------------------------
# General purpose auxiliary functions
# -----------------------------------------------------------------------------------------

def conversations_dir():
    return os.getenv('OL3M_CONVERSATIONS_DIR','../conversations/')

def descriptions_dir():
    return os.getenv('OL3M_DESCRIPTIONS_DIR','../descriptions/')

def rdf_dir():
    return os.getenv('OL3M_RDF_DIR','../rdf/')

def output_dir():
    return os.getenv('OL3M_OUTPUT_DIR','../output/')

def ontologies_dir():
    return os.getenv('OL3M_ONTOLOGIES_DIR','../ontologies/')

def models_dir():
    return os.getenv('OL3M_MODELS_DIR','../models/')

def get_datetime():
    '''
        This function gets the current date time and returns it as a string.

        @returns dt_string (str): current time formatted in a string
    '''
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string


def concatenate(iterable, sep=""):
    sentence = iterable[0]
    for word in iterable[1:]:
        sentence += (sep + word)
    return sentence


def activate_gpu(force_cpu=False):
    '''
        A function to return the right device depending on GPU availability
    '''
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            # print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            # print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            # print('DEVICE = ', colored('CPU', "blue"))
    return device

# -----------------------------------------------------------------------------------------
# Generation utils
# -----------------------------------------------------------------------------------------

def identify_last_user_utterance(user_utterances:list):
    pattern = re.compile(r"^conv_(\d+)\.(\d+)_(\d+)_utt$")

    max_j = -1
    last_utt = user_utterances[0]

    for utt in user_utterances:
        match = pattern.match(str(utt))
        if match:
            i, j = int(match.group(2)), int(match.group(3))
            if j > max_j:
                max_j = j
                last_utt = utt

    return last_utt

def simplify_value(value):
    # Single ontology entity or float/string
    if isinstance(value, list):
        if len(value) == 1:
            return simplify_value(value[0])
        else:
            return [v.name if hasattr(v, "name") else v for v in value]
    else:
        return [value.name if hasattr(value, "name") else value]


def remove_bracketed_text(sentence):
    return re.sub(r'\[.*?\]', '', sentence)

def get_latest_conv_index(lst):
    str_lst = [str(l) for l in lst]
    max_index = -1
    max_value = -1
    for i, item in enumerate(str_lst):
        match = re.search(r'\.(\d+)_conv$', item)
        if match:
            value = int(match.group(1))
            if value > max_value:
                max_value = value
                max_index = i
    return max_index


def get_current_phase_and_characts(indiv_path):
    indiv_world = World()
    individual = indiv_world.get_ontology(indiv_path).load()

    # next characts instructions are on the last utterance
    user_utterances = individual.search(is_a = individual.UserUtterance)
    last_user_utt = identify_last_user_utterance(user_utterances)
    next_characts = last_user_utt.nextCharacteristic
    next_characts = [str(n).split(".")[1] for n in next_characts]

    properties = last_user_utt.get_properties()
    prop_values = {}

    for prop in properties:
        prop_name = prop.name
        raw_value = getattr(last_user_utt, prop_name, None)
        prop_values[prop_name] = simplify_value(raw_value)

    # the current phase information is the output of a conversation property, the last conversation element stored in the ontology
    conversations = individual.search(is_a = individual.Conversation)
    conversation = conversations[get_latest_conv_index(conversations)]
    current_phase = conversation.inPhase
    current_phase = [str(p).split(".")[1] for p in current_phase]

    print(colored(f"Properties of the user utterance: {prop_values}", 'green'))

    return current_phase, prop_values


def format_user_history(user_prompt, experiment):
    path_to_conv_history = f'{conversations_dir()}/{experiment}_whole_history.txt'

    # check if there is already a conversation history
    if os.path.exists(path_to_conv_history):
        with open(path_to_conv_history, 'r') as f:
            conversation_history = f.read().strip()
    else:
        conversation_history = ""

    # format the output for the prompt with or without history
    if conversation_history:
        formatted_history = f"{conversation_history}\n<|start_header_id|>user<|end_header_id|> {user_prompt}<|eot_id|>"
    else:
        formatted_history = f"<|start_header_id|>user<|end_header_id|> {user_prompt}<|eot_id|>"

    return formatted_history


def next_cefr_index_prefix_with_conv_history(classes, cefr_index, user_prompt, experiment):
    prefix = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You will be the history of a user/agent conversation and an instruction between brackets to format your answer properly. Answer directly after the brackets, and start your answer with a word, not a figure. Adapt the length of the answer to the instruction between brackets. <|eot_id|><|start_header_id|>user<|end_header_id|>

        {format_user_history(user_prompt, experiment)} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        [ProficiencyLevel: {classes[cefr_index]}]
        '''

    return prefix, cefr_index


def build_prefix_from_constraints(constraints):
    constraints_mapping = {'non_loaded':"'Load': 0",
                           'loaded':"'Load': 1", 
                           'negative':"'Polarity': 0", 
                           'neutral':"'Polarity': 1", 
                           'positive':"'Polarity': 2", 
                           'A1':"'ProficiencyLevel': 0", 
                           'A2':"'ProficiencyLevel': 1", 
                           'B1':"'ProficiencyLevel': 2", 
                           'B2':"'ProficiencyLevel': 3", 
                           'C1':"'ProficiencyLevel': 4", 
                           'C2':"'ProficiencyLevel': 5",
                        }
    prefix = '['
    print(colored(f"Given constraints: {constraints}", 'red'))
    for const in constraints:
        formatted_const = constraints_mapping[const]
        prefix += formatted_const + ', '

    return prefix[:len(prefix)-2] + ']'

def get_prompt_from_inferred_info(current_phase, characts, user_prompt, experiment):
    phase_mapping = {'phase1': "Phase 1", 'phase2': "Phase 2", 'phase3': "Phase 3", 'phase4': "Phase 4", 'phase5': "Phase 5"}
    history = format_user_history(user_prompt, experiment)
    phase = phase_mapping[current_phase[0]]

    prompt_header = prompt_headers[phase]
    constraints = characts['nextCharacteristic']
    prefix = build_prefix_from_constraints(constraints)

    prompt = f'''
{prompt_header}
{history}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
{prefix}
    '''

    return prompt


def generate_from_next_cefr_index(args, cefr_index, user_prompt, experiment):
    classes = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    tokenizer = args['tokenizer']
    model = args['model']
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    # get prompts and labels for the whole batch
    prompt, label = next_cefr_index_prefix_with_conv_history(classes, cefr_index, user_prompt, experiment)

    # Perform single utterance generation (not batched)
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(args['device'])

    # generate content (and time it)
    tic = time.time()
    outputs = model.generate(
        input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        max_new_tokens=args['max_new_tokens'],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators
    )
    toc = time.time()

    # Decode only the newly generated tokens
    generated_tokens = outputs[:, input_tokens['input_ids'].shape[-1]:] if len(outputs.shape) > 1 else outputs
    results = tokenizer.decode(generated_tokens[0], skip_special_tokens=True) if len(outputs.shape) > 1 else tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Find the part of the result after the special token '<|end_header_id|>'
    generation = results[results.find('<|end_header_id|>')+1:] if '<|end_header_id|>' in results else results
    # remove any prefix between brackets
    cleaned_generation = remove_bracketed_text(generation)[1:] # remove space at the beginning of generated content

    return toc-tic, cleaned_generation

def generate_from_next_characteristics(args, current_phase, characts, user_prompt, experiment):
    tokenizer = args['tokenizer']
    model = args['model']
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # get the appropriate prompt from the current phase and the next characts
    prompt = get_prompt_from_inferred_info(current_phase, characts, user_prompt, experiment)
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(args['device'])

    # generate content (and time it)
    tic = time.time()
    outputs = model.generate(
        input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        max_new_tokens=args['max_new_tokens'],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators
    )
    toc = time.time()

    # Decode only the newly generated tokens
    generated_tokens = outputs[:, input_tokens['input_ids'].shape[-1]:] if len(outputs.shape) > 1 else outputs
    results = tokenizer.decode(generated_tokens[0], skip_special_tokens=True) if len(outputs.shape) > 1 else tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Find the part of the result after the special token '<|end_header_id|>'
    generation = results[results.find('<|end_header_id|>')+1:] if '<|end_header_id|>' in results else results
    # remove any prefix between brackets
    cleaned_generation = remove_bracketed_text(generation)[1:] # remove space at the beginning of generated content

    return toc-tic, cleaned_generation