import torch
import torch.nn as nn
from termcolor import colored
import os
import warnings
import json
from peft import PeftModel
from transformers import RobertaModel, AutoModelForSequenceClassification, RobertaTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse
from utils import *
from build_individuals_and_infer import *

class RobertaWithAdditionalLayers(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(RobertaWithAdditionalLayers, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # freeze pre-trained roberta
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False

        # Additional linear layers with dropout
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, 1)
        
        # Dropout and activation layers
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # CLS token output
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        logits = self.classifier(x)
        output = self.sigmoid(logits)
        
        return output

def test_usual_text_generation(model, base_model_name, prompt):
    """
        This function simply checks if the newly fine-tuned model still performs well in the classical generation task. This would prevent from evaluating a broken generation as it already happened in previous training pipelines. Therefore, this function is not used in the generation pipeline for conversation.

        > IMPORTANT NOTE: This function raises the false alarm that PeftModel cannot be used in text-generation. It is actually possible and the models generates the desired output after this warning. Note that the problem is identified and still being solved by the Huggingface team: https://github.com/huggingface/transformers/issues/29395

        Args:
            model (str or model object): The language model to use for text generation.
            base_model_name (str): The name of the base model for the tokenizer.
            prompt (str): The initial text prompt to start the text generation.

        Returns:
            None
            Prints the generated text to the console.
    """
    messages = [prompt]

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_id("<|eot_id|>")
    ]

    pipe = pipeline(
        "text-generation",
        model = model,
        model_kwargs = {"torch_dtype": torch.bfloat16},
        device_map = "auto",
        tokenizer = AutoTokenizer.from_pretrained(base_model_name),
    )

    outputs = pipe(
        messages,
        max_new_tokens = 256,
        eos_token_id = terminators,
        do_sample = True,
        temperature = 0.6,
        top_p = 0.9,
    )
    print(outputs[0][0]['generated_text'])


def format_generation(generation):
    '''
        Remove the last sentence if unfinished (so that the agent looks more realistic)
    '''
    sentence_pattern = r'(.*?[.!?])'
    sentences = re.findall(sentence_pattern, generation, re.DOTALL)

    if sentences:
        formatted_text = ' '.join(sentences).strip()
    else:
        formatted_text = ''

    return f'Agent > {formatted_text}'


def evaluate_one_iter_conv(args, experiment, it, load_predictor, load_tokenizer, polarity_predictor, polarity_tokenizer):
    '''
        An evaluation step to test the conversation strategy. It is composed of the following steps:
            - get the prompt of the user through an input
            - pass this prompt to the ontology to get its CEFR level
            - deduce the correct prefix to use in such case
            - generate after this prefix

        In this case the batch is 1, we simply interact with the model on the current conversation.
    '''
    file_path = f'{conversations_dir()}/{experiment}_user_utt.txt'
    history_file_path = f'{conversations_dir()}/{experiment}_whole_history.txt'
    # if it is a new conversation, clear the file that store user prompts
    if it == 0:
        open(file_path, 'w').close()

    # get and store the user prompt
    user_prompt = input("User > ")

    with open(file_path, 'a') as f:
        f.write('- ' + user_prompt + '\n')

    # build the ontology individual from all previous user utterances - WORKING
    pellet_runtime = individual_generation(args['world'], args['onto'], args['nlp'], it, file_path, load_predictor, load_tokenizer, polarity_predictor, polarity_tokenizer, experiment)

    # retrieve the detected phase and the next characteristic instructions
    conv_onto_path = f'{rdf_dir()}/{experiment}.rdf'
    current_phase, characts = get_current_phase_and_characts(conv_onto_path)

    # give the right prompt to the model depending on the detected phase and the next characteristic specifications
    llama_runtime, generation = generate_from_next_characteristics(args, current_phase, characts, user_prompt, experiment)

    formatted_generation = format_generation(generation)
    print(formatted_generation)

    # update the conversation history with the new content
    with open(history_file_path, 'a') as f:
        f.write('<|start_header_id|>user<|end_header_id|> ' + user_prompt + '<|eot_id|>\n')
        f.write('<|start_header_id|>assistant<|end_header_id|> ' + generation + '<|eot_id|>\n')

    # initialize json content
    json_content = {
    'prompt': user_prompt,
    'generated_content': generation,
    'current_phase': current_phase,
    'next_characteristics': characts['nextCharacteristic'],
    'id': experiment,
    'it': it,
    'user_utt_features': {
        'cole': characts['hasColeReadabilityScore'],
        'flesch_kincaid': characts['hasFleschKincaidReadabilityScore'],
        'average_word_length': characts['hasAverageWordLength'],
        'mtld': characts['hasMTLDMeasure'],
        'smog': characts['hasSMOGReadabilityScore'],
        'pronoun_density': characts['hasPronounDensity'],
        'characteristics': characts['hasCharacteristic'],
        },
    'runtimes': {
        'pellet_runtime': pellet_runtime,
        'llama_runtime': llama_runtime,
        }
    }

    # print(json_content)

    result = json.dumps(json_content, indent=4)

    # dump dialog turn content into a json file
    with open(f'{output_dir()}/{experiment}_{it}.json', 'w') as f:
        f.write(result)

    return result


def evaluate_conv(args, experiment):
    n = args['eval_iters']

    load_predictor = RobertaWithAdditionalLayers()
    load_predictor.load_state_dict(torch.load('../models/best_load_predictor.pt', weights_only=True))

    polarity_predictor = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    polarity_state_dict = torch.load('../models/best_polarity_predictor.pt')
    polarity_predictor.load_state_dict(polarity_state_dict)

    load_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    polarity_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

    for it in range(n):
        evaluate_one_iter_conv(args, experiment, it, load_predictor, load_tokenizer, polarity_predictor, polarity_tokenizer)


def init_onto_linguo_llm():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    # warning management
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token.As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")

    nlp = spacy.load("en_core_web_sm")

    # load onto in a new, dedicated world
    w = World()
    onto = w.get_ontology(f"{ontologies_dir()}/interview-strategy-merged.rdf").load()


    args = {
    'device': activate_gpu(),
    'eval_iters': 5,
    'max_new_tokens': 100,
    }

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_model_name = "../adapters/gpro_multicontrol_llama3.1/checkpoint-155"  # change adapter model here

    # get the base model
    clm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    # plug adapters into the base model
    clm_model = PeftModel.from_pretrained(clm_model, adapter_model_name)

    args.update({
        'model': clm_model,
        'nlp': nlp,
        'onto': onto,
        'world': w, # store the world to infer on it later on
        'tokenizer': AutoTokenizer.from_pretrained(model_name),
        'eval_iters': 12, # the conversation has 12 user turns by construction
        'max_new_tokens': 100 # the max size of the generated text (in tokens)
    })

    args['tokenizer'].pad_token_id = args['tokenizer'].eos_token_id

    return args

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tiny-model', '-t', help="Whether to use a tiny llama model for testing, or consider llama3 7B which is the intended model for this experiment.", action='store_true')
    argument = parser.parse_args()

    args = init_onto_linguo_llm()

    evaluate_conv(args, experiment=f'conv_{get_datetime()}')
