import json

import streamlit as st
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BertForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer
import requests


device = 'cuda:0'

st.set_page_config(page_title="Social Support Chat Test", layout='wide')
st.title("Social Support Chat Test")

@st.cache_resource
def init_supporter_model():
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

    return model_supporter, tokenizer_supporter

@st.cache_resource
def init_info_selecter():
    model_path_info_selecter = 'model/models/info_selecter'
    tokenizer_info_selecter = AutoTokenizer.from_pretrained(model_path_info_selecter)
    model_info_selecter = BertForSequenceClassification.from_pretrained(model_path_info_selecter)
    model_info_selecter.to(device)
    model_info_selecter.eval()

    return model_info_selecter, tokenizer_info_selecter

def get_memory(history, info_dict, model, tokenizer, max_length=512, filter_fields=[], device='cuda:0'):
    info_dict = {k:v for k, v in info_dict.items() if k not in filter_fields}
    best_score = 0
    selected_aspect = ''
    selected_info = ''
    for aspect, info in info_dict.items():
        inputs = '[CLS]' + history + '[SEP]' + info + '[SEP]'
        input_ids = tokenizer(inputs, max_length=max_length, return_tensors='pt')
        input_ids.to(device)
        score = model(**input_ids).logits
        score = score[0][0]
        if score > best_score:
            best_score = score
            selected_aspect = aspect
            selected_info = info
    return selected_aspect, selected_info

def count_aspect(aspect_list):
    aspect_num_dict = {}
    for aspect in aspect_list:
        if aspect not in aspect_num_dict:
            aspect_num_dict[aspect] = 0
        aspect_num_dict[aspect] += 1
    filter_fields = [k for k, v in aspect_num_dict.items() if v >= 3]

    return filter_fields

def gen_dial(history, supporter_statement, supporter_profile_trans, model_supporter, tokenizer_supporter, aspect_list, device='cuda:0'):
    supporter_system = "<SYSTEM>{}".format(supporter_statement)
    supporter_info = "<SUPPORTER>[INFO-{}]"
    history_str = ''
    for _ in history:
        if _['speaker'] == 'seeker':
            history_str += '<SEEKER>' + _['utterance']
        elif _['speaker'] == 'supporter':
            history_str += '<SUPPORTER>' + _['utterance']
    filter_fields = count_aspect(aspect_list)
    selected_aspect, selected_info = get_memory(history_str, supporter_profile_trans, model_info_selecter, tokenizer_info_selecter, filter_fields=filter_fields, device=device)
    if random.randint(0, 2) == 0:
        if len(history) > 0:
            aspect_list.append(selected_aspect)
        if len(history) == 0:
            selected_info = ''
    else:
        selected_info = ''
    info_content = supporter_info.format(selected_info)
    inputs = supporter_system + history_str + info_content
    encoded_inputs = tokenizer_supporter(inputs, return_tensors='pt').to(device)
    outputs = model_supporter.generate(encoded_inputs['input_ids'], max_length=2048, do_sample=True)
    decoded_output = tokenizer_supporter.decode(outputs[0], skip_special_tokens=True)
    decoded_output = decoded_output.replace(inputs, '')

    end_flag = False
    if len(decoded_output.split(' ')) <= 2:
        end_flag = True
    return decoded_output, end_flag

##### session #####
if 'character_list' not in st.session_state:
    st.session_state['character_list'] = []
if 'cur_supporter' not in st.session_state:
    st.session_state['cur_supporter'] = {}
if 'cur_supporter_history' not in st.session_state:
    st.session_state['cur_supporter_history'] = []
if 'cur_supporter_aspect_list' not in st.session_state:
    st.session_state['cur_supporter_aspect_list'] = []
if 'cur_supporter_dial_end' not in st.session_state:
    st.session_state['cur_supporter_dial_end'] = False
##### session #####

model_supporter, tokenizer_supporter = init_supporter_model()
model_info_selecter, tokenizer_info_selecter = init_info_selecter()

if st.session_state['character_list'] == []:
    with open('model/demo/mbti_profile_dataset.json', 'r+', encoding='utf-8') as f:
        character_list = [json.loads(_) for _ in f.readlines()]
    st.session_state['character_list'] = character_list

if st.button('Get Random Supporter'):
    selected_index = random.sample([i for i in range(len(st.session_state['character_list']))], 1)[0]
    character_choose = st.session_state['character_list'][selected_index]
    st.session_state['cur_supporter'] = character_choose
    st.session_state['cur_supporter_history'] = []
    st.session_state['cur_supporter_aspect_list'] = []
    st.session_state['cur_supporter_dial_end'] = False
    st.experimental_rerun()
st.write('supporter index: ')
if st.session_state['cur_supporter'] != {}:
    st.write(st.session_state['cur_supporter']['index'])
if st.button('Start Chat'):
    if st.session_state['cur_supporter'] != {}:
        cur_supporter = st.session_state['cur_supporter']
        supporter_field_filter = ['tone', 'recent_worry_or_anxiety']
        profile_trans = cur_supporter['profile_trans']
        profile_trans = {k:v for k, v in profile_trans.items() if k not in supporter_field_filter}
        cur_supporter['profile_trans'] = profile_trans
        st.session_state['cur_supporter'] = cur_supporter
        st.session_state['cur_supporter_history'] = []
        st.session_state['cur_supporter_aspect_list'] = []
        supporter_utterance, end_flag = gen_dial(st.session_state['cur_supporter_history'], st.session_state['cur_supporter']['supporter_statement'], st.session_state['cur_supporter']['profile_trans'], model_supporter, tokenizer_supporter, st.session_state['cur_supporter_aspect_list'], device=device)
        st.session_state['cur_supporter_history'].append({'speaker': 'supporter', 'utterance': supporter_utterance})
        st.session_state['cur_supporter_dial_end'] = False
    else:
        st.error('You Need to Get a Supporter')

dial_content = ''
st.write('Chat History')
for i in range(len(st.session_state['cur_supporter_history'])):
    _ = st.session_state['cur_supporter_history'][i]
    if i == len(st.session_state['cur_supporter_history']) - 1:
        if _['speaker'] == 'seeker':
            st.write('<font color=\'black\'>'+'You: '+_['utterance']+'</font>', unsafe_allow_html=True)
        elif _['speaker'] == 'supporter':
            st.write('<font color=\'black\'>'+'Helper: '+_['utterance']+'</font>', unsafe_allow_html=True)
    else:
        if _['speaker'] == 'seeker':
            st.write('<font color=\'gray\'>'+'You: '+_['utterance']+'</font>', unsafe_allow_html=True)
        elif _['speaker'] == 'supporter':
            st.write('<font color=\'gray\'>'+'Helper: '+_['utterance']+'</font>', unsafe_allow_html=True)

seeker_input = st.text_input('Input', key='cur_supporter_input', disabled=st.session_state['cur_supporter_dial_end'])
if st.session_state['cur_supporter_dial_end']:
    st.warning('The Number of Words / Rounds Has Reached The Upper Limit')
if st.button('Submit', key='cur_supporter_submit', disabled=st.session_state['cur_supporter_dial_end']):
    if st.session_state['cur_supporter'] == {}:
        st.error('You Should Start the Chat First')
    else:
        st.session_state['cur_supporter_history'].append({'speaker': 'seeker', 'utterance': seeker_input})
        supporter_utterance, end_flag = gen_dial(st.session_state['cur_supporter_history'], st.session_state['cur_supporter']['supporter_statement'], st.session_state['cur_supporter']['profile_trans'], model_supporter, tokenizer_supporter, st.session_state['cur_supporter_aspect_list'], device=device)
        if end_flag or len(st.session_state['cur_supporter_history']) > 15:
            st.session_state['cur_supporter_dial_end'] = True
            st.experimental_rerun()
        st.session_state['cur_supporter_history'].append({'speaker': 'supporter', 'utterance': supporter_utterance})
        st.experimental_rerun()