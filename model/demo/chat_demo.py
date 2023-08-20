from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BertForSequenceClassification
import json
import argparse

def get_score(seeker_persona, supporter_persona, model, tokenizer, max_length=512, device='cuda:0'):
    inputs = '[CLS]' + seeker_persona + '[SEP]' + supporter_persona + '[SEP]'
    input_ids = tokenizer(inputs, max_length=max_length, return_tensors='pt')
    score = model(**input_ids).logits
    score = score[0][0]
    return score

def get_memory(history, info_list, model, tokenizer, max_length=512, device='cuda:0'):
    best_score = 0
    selected_info = ''
    for info in info_list:
        inputs = '[CLS]' + history + '[SEP]' + info + '[SEP]'
        input_ids = tokenizer(inputs, max_length=max_length, return_tensors='pt')
        score = model(**input_ids).logits
        score = score[0][0]
        if score > best_score:
            best_score = score
            selected_info = info
    return selected_info

if __name__ == '__main__':
    device = 'cuda:0'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeker_robot', action='store_true', help='chat with model_seeker')
    parser.add_argument('--supporter_robot', action='store_true', help='chat with model_supporter')
    args = parser.parse_args()

    seeker_robot = args.seeker_robot
    supporter_robot = args.supporter_robot

    seeker_filter_fields = ['tone']
    supporter_filter_fields = ['recent_worry_or_anxiety', 'tone']

    with open('model/demo/mbti_profile_dataset.json', 'r+', encoding='utf-8') as f:
        character_list = [json.loads(_) for _ in f.readlines()]

    if seeker_robot:
        print('loading seeker model...')
        model_path_seeker = 'model/models/model_seeker'
        tokenizer_seeker = AutoTokenizer.from_pretrained(
                model_path_seeker
            )
        config_seeker = AutoConfig.from_pretrained(model_path_seeker)
        model_seeker = AutoModelForCausalLM.from_pretrained(
                model_path_seeker,
                config=config_seeker
            ).half()
        model_seeker.to(device)
        model_seeker.eval()

        seeker_index = input('seeker index=')
        seeker_index = int(seeker_index)
        seeker = character_list[seeker_index]
        seeker_persona = seeker['seeker_statement']
        seeker_profile_trans = seeker['profile_trans']
        seeker_profile_trans = [v for k, v in seeker_profile_trans.items() if k not in seeker_filter_fields]
        print('seeker persona:\n'+seeker_persona)

    else:
        seeker_persona = input('seeker persona:\n')

    if supporter_robot:
        print('loading supporter model...')
        model_path_supporter = 'model/models/model_supporter'
        tokenizer_supporter = AutoTokenizer.from_pretrained(
                model_path_supporter
            )
        config_supporter = AutoConfig.from_pretrained(model_path_supporter)
        model_supporter = AutoModelForCausalLM.from_pretrained(
                model_path_supporter,
                config=config_supporter
            ).half()
        model_supporter.to(device)
        model_supporter.eval()

        supporter_index = input('supporter index=')
        supporter_index = int(supporter_index)
        supporter = character_list[supporter_index]
        supporter_persona = supporter['supporter_statement']
        supporter_profile_trans = supporter['profile_trans']
        supporter_profile_trans = [v for k, v in supporter_profile_trans.items() if k not in supporter_filter_fields]
    else:
        supporter_persona = input('supporter persona:\n')

    print('loading memory selecter...')
    model_path_info_selecter = 'model/models/info_selecter'
    tokenizer_info_selecter = AutoTokenizer.from_pretrained(model_path_info_selecter)
    model_info_selecter = BertForSequenceClassification.from_pretrained(model_path_info_selecter)

    seeker_system = "<SYSTEM>{}".format(seeker_persona)
    supporter_system = "<SYSTEM>{}".format(supporter_persona)
    seeker_info = "<SEEKER>[INFO-{}]"
    supporter_info = "<SUPPORTER>[INFO-{}]"
    history = "<SUPPORTER>Hello, what can i assist you?"
    while True:
        if seeker_robot:
            selected_info = get_memory(history, seeker_profile_trans, model_info_selecter, tokenizer_info_selecter, device=device)
            info_content = seeker_info.format(selected_info)
            inputs = seeker_system + history + info_content
            encoded_inputs = tokenizer_seeker(inputs, return_tensors='pt').to(device)
            outputs = model_seeker.generate(encoded_inputs['input_ids'], max_length=1024, do_sample=True)
            decoded_output = tokenizer_seeker.decode(outputs[0], skip_special_tokens=True)
            decoded_output = decoded_output.replace(inputs, '')
            print('seeker: ', decoded_output)
            history += '<SEEKER>' + decoded_output
        else:
            seeker_utterance = input('seeker: ')
            history += '<SEEKER>' + seeker_utterance
        if supporter_robot:
            selected_info = get_memory(history, supporter_profile_trans, model_info_selecter, tokenizer_info_selecter, device=device)
            info_content = supporter_info.format(selected_info)
            inputs = supporter_system + history + info_content
            encoded_inputs = tokenizer_supporter(inputs, return_tensors='pt').to(device)
            outputs = model_supporter.generate(encoded_inputs['input_ids'], max_length=1024, do_sample=True)
            decoded_output = tokenizer_supporter.decode(outputs[0], skip_special_tokens=True)
            decoded_output = decoded_output.replace(inputs, '')
            print('supporter: ', decoded_output)
            history += '<SUPPORTER>' + decoded_output
        else:
            supporter_utterance = input('supporter: ')
            history += '<SUPPORTER>' + supporter_utterance