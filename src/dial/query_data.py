import os
import re

import openai
import json
import random
import threading
import datetime
import string
import difflib
import random
import argparse

def get_data(content, mode='', api_key='', max_tries=5):
    n = 0
    while n < max_tries:
        try:
            if mode == 'multi':
                openai.api_key = api_key
                USE_MODEL = "gpt-3.5-turbo-16k-0613"
            response = openai.ChatCompletion.create(
                model=USE_MODEL,
                messages=content
            )
            break
        except Exception as e:
            print(e)
            n += 1
            continue
    if n == max_tries:
        return -1
    return response["choices"][0]["message"]["content"]

def process_content(content, last_sentence=''):
    content = re.sub(r'\n+', '', content)
    content = re.sub(r'^.*?[：:]', '', content)
    content = re.sub(r'^(Hey|Oh|Absolutely)+[,.!?]?', '', content)
    content = re.sub(r'{', '(', content)
    content = re.sub(r'}', ')', content)
    special_character_list = [',', '.', '!', '?']
    for _ in special_character_list:
        content = content.replace(_, _+'#')
    content_list = content.split('#')
    content_list = [_ for _ in content_list if len(_.split(' ')) >= 3]
    
    filter_words = ['Thank you', 'AI']
    new_content = ''
    for _ in content_list:
        flag = False
        for word in filter_words:
            if word in _:
                flag = True
                break
        if not flag:
            new_content += _
    
    if last_sentence != '':
        special_character_list = [',', '.', '!', '?']
        for _ in special_character_list:
            last_sentence = last_sentence.replace(_, _+'#')
            new_content = new_content.replace(_, _+'#')
        last_sentence_list = last_sentence.split('#')
        new_content_list = new_content.split('#')
        s1 = last_sentence_list[0]
        s2 = new_content_list[0]
        score = difflib.SequenceMatcher(None, s1, s2).quick_ratio()
        if score > 0.9:
            new_content_list = new_content_list[1:]
        new_content = ''.join(new_content_list)
    new_content = new_content.strip()
    if new_content != '' and new_content[0] in string.ascii_lowercase:
        character_dict = dict(zip(string.ascii_lowercase, string.ascii_uppercase))
        new_content = character_dict[new_content[0]] + new_content[1:]
    return new_content

def eval_sentence(sentence, api_key=''):
    eval_prompt = """Task: Sentiment Scoring
Task introduction: Given a sentence, evaluate its sentiment according to the scoring criteria. The more negative the sentiment of the sentence, the lower the score, and the more positive the sentiment, the higher the score. The lowest score is 0 points and the highest score is 5 points.
Grading: 
0 points: There is no hope for tomorrow, so what is the use of hard work?
1 point: I can't do anything well, I'm a useless person. No one can help me either.
2 points: I have encountered a lot of difficulties recently, I am under a lot of pressure, and I can’t sleep because of anxiety every day.
3 points: Oh, I've been so busy with work recently, I haven't gone out to play for a long time.
4 points: This problem is really bothering me, but I believe that with hard work, I can solve this problem.
5 points: Exactly! The difficulty is only temporary, I am a talented person, this difficulty will always be solved by me!

Sentiment score the following sentences. Only the rating is output, no additional information is required.
{}
"""
    sentence = re.sub(r'[\(（].*?[\)）]', '', sentence)
    if sentence == '':
        return -1
    eval_prompt = eval_prompt.format(sentence)
    content = [{'role': 'user', 'content': eval_prompt}]
    resp = get_data(content, mode='multi', api_key=api_key)
    if resp == -1:
        resp = ''
    score = re.findall(r'[0-9]+', resp)
    if len(score) == 0:
        return -1
    score = int(score[0])
    if not (score >= 0 and score <= 5):
        return -1
    return score

def check_end_state(score_list):
    if len(score_list) >= 5 and score_list[-2] >= 4 and score_list[-1] >= 4:
        return True
    return False

def get_additional_info(last_sentence, info_list, info_filter=[], api_key=''):
    info_list = [_ for _ in info_list if _ not in info_filter]
    
    prompt = """You are chatting with someone, just now, the other person said:
{}
In order to make the conversation flow more smoothly, you may need some additional information about yourself to respond. Now, you can choose the most suitable aspect from the following list:
{}

Only output the aspect you choose, do not output redundant information. If you don't think you need to select any of them for responding, output "NONE"
"""
    prompt = prompt.format(last_sentence, str(info_list))
    content = [{'role': 'user', 'content': prompt}]
    resp = get_data(content, mode='multi', api_key=api_key)
    if resp == -1:
        resp = ''
    info = [_ for _ in info_list if _ in resp]
    if info:
        info = info[0]
    else:
        info = ''
    return info

def get_reply(last_sentence, raw_reply, field, info, api_key=''):
    prompt = """You are chatting with someone.
Just now, the other person said: {}
And you reply: {}
Now here is the additional character settings about your {}:
{}

Note that this character settings is about yourself, not about someone else, which means {}
For information that does not appear in the character settings, you can use the second person to express.

Please rewrite what you said based on this character settings with emoji to better reply the other person.
When rewriting, you need to mention this character settings.
Control the output length not to exceed 2 sentences and less than 30 words. These sentences are not allowed to be separated by commas.

Output:"""
    statement_dict = {
        'recent_worry_or_anxiety': 'the worry or anxiety is your worry or anxiety. Not mine, not someone else\'s. So when you mention your worry or anxiety in this character settings, use the first person to express (I/My ...).',
        'hobby': 'the hobbies if your hobbies. Not mine, not someone else\'s. So when you mention your worry or anxiety in this character settings, use the first person to express (I/My ...).',
        'growth_experience': 'this is your own and unique grow experience. Not other people\'s experience. So when you mention your grow experience in this character settings, use the first person to express (I/My ...).',
        'family_relationship': 'the family is yours, not other people\'s. So when you mention your family in this character settings, use the first person to express (I/My ...).',
        'working_conditions': 'this is your own working conditions. Not mine, not someone else\'s. So when you mention your working conditions in this character settings, use the first person to express (I/My ...).',
        'social_relationship': 'this is you own and unique social relationship. Those friends are yours, not mine, not someone else\'s. So when you mention your social relationship in this character settings, use the first person to express (I/My ...).',
        'emotional_state': 'this is your emotional state. The partner is yours, not mine, not someone else\'s. So when you mention your emotional state in this character settings, use the first person to express (I/My ...).',
        'living_conditions': 'this is your own living conditions. Not other people\'s living conditions. So when you mention your living conditions in this character settings, use the first person to express (I/My ...).',
        'additional_information': 'this is some additional information about you. So when you mention this character settings, use the first person to express (I/My ...).',
        'advantages_and_disadvantages': 'this is your advantages and disadvantages. Not mine, not someone else\'s. So when you mention your advantages and disadvantages in this character settings, use the first person to express (I/My ...).'
    }
    if field in statement_dict:
        field_statement = statement_dict[field]
    else:
        field_statement = 'use the first person to express (I/My ...).'
    prompt = prompt.format(last_sentence, raw_reply, field, info, field_statement)
    content = [{'role': 'user', 'content': prompt}]
    
    resp = get_data(content, mode='multi', api_key=api_key)
    if resp == -1:
        return raw_reply
    return resp

def get_info_filter(field_list, info, round_num=5, max_num=3):
    if len(field_list) < round_num:
        field_list.append(info)
    else:
        field_list.pop(0)
        field_list.append(info)
    field_nums = {}
    for field in field_list:
        if field not in field_nums:
            field_nums[field] = 1
        else:
            field_nums[field] += 1
    for k, v in field_nums.items():
        if v >= max_num:
            info_filter = [k]
    return info_filter

def gen_dial(seeker_statement, supporter_statement, seeker_profile, supporter_profile, seeker_tone='', supporter_tone='', max_turn=20, api_key=''):
    ########## prompt ##########

    answer_prompt_prefix_supporter = """You are chatting with someone online. The other party has encountered some problems recently, and you will give comfort and help to the other party through chatting.

Now, the other party say:
"""
    answer_prompt_prefix_seeker = """You are chatting with someone online. You have encountered some problems recently and want to have chat with someone. If the other party ask you recent situation, you will tell the other party the issue you encountered in detail.

Now, the other party say:
"""
    answer_prompt_postfix_supporter = """
Your response should meet the following requirements:
1. Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.
2. Talk like a normal person, which means you don't have to be too polite and don't repeat what the other person or yourself have said.
3. Every time you reply, you need to insert emoji in the reply, which should match the emotion and dialogue situation of the reply. Emoji should be rich and avoid duplication with what is already in the chat history.
4. Control the length, only responed one or two sentences at a time, each sentence should be short, just like a normal person talk with someone.

Use everyday language to reply.{}
Here is some additional settings about yourself:
{}

Now, you reply:"""
    answer_prompt_postfix_seeker = """
Your response should meet the following requirements:
1. Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.
2. Talk like a normal person, which means you don't have to be too polite and don't repeat what the other person or yourself have said.
3. Every time you reply, you need to insert emoji in the reply, which should match the emotion and dialogue situation of the reply. Emoji should be rich and avoid duplication with what is already in the chat history.
4. Control the length, only responed one or two sentences at a time, each sentence should be short, just like a normal person talk with someone.

Use everyday language to reply.{}
Here is some additional settings about yourself:
{}

Now, you reply:"""

    tone_prompt = """Deep imitation of a person's speaking tone, which is described as follows:
{}

Generate 5 samples in the format of "When the other party xxx, you should say xxx"
Do not generate redundant information.
"""

    supporter_end_prompt = """You had a chat with the other party, and the other party got your comfort. It's time to say goodbye.
Say goodbye to the other party and end this conversation! However, you'd better think of a better reason to say goodbye based on your character setting. At the same time, the farewell should not be too abrupt, otherwise it will appear impolite.
Reply with one or two sentences, each sentence should be short, just like a normal person talk with someone. Your tone is described as follows:
{}

The other party just say: {}
Now, you reply:
"""

    seeker_end_prompt = """You had a chat with the other party, and you got comfort and advice from the other party after chatting with the other party about the problems you encountered. It's time to say goodbye.
Say goodbye to the other party and end this conversation! However, you'd better think of a better reason to say goodbye based on your character setting. At the same time, the farewell should not be too abrupt, otherwise it will appear impolite.
Reply with one or two sentences, each sentence should be short, just like a normal person talk with someone. Your tone is described as follows:
{}

The other party just say: {}
Now, you reply:
"""

    ########## prompt ##########

    seeker_base_fields = ['name', 'gender', 'age', 'region', 'tone', 'job', 'personality', 'additional_information']
    supporter_base_fields = ['name', 'gender', 'age', 'region', 'tone', 'job', 'personality', 'additional_information', 'recent_worry_or_anxiety']
    seeker_additional_info = [_ for _ in seeker_profile if _ not in seeker_base_fields]
    supporter_additional_info = [_ for _ in supporter_profile if _ not in supporter_base_fields]

    dial = []
    dial_flag = True
    
    seeker_content = []
    supporter_content = []

    ### get tone sentences
    if seeker_tone:
        seeker_tone_prompt = tone_prompt.format(seeker_tone)
        content = [{'role': 'user', 'content': seeker_tone_prompt}]
        seeker_tone_sentences = get_data(content, mode='multi', api_key=api_key)
        if seeker_tone_sentences != -1:
            seeker_statement = seeker_statement + '\nDuring the conversation, you should mimic the tone in the following example:\n[Example tone]\n' + seeker_tone_sentences
        else:
            seeker_tone_sentences = ''

    if supporter_tone:
        supporter_tone_prompt = tone_prompt.format(supporter_tone)
        content = [{'role': 'user', 'content': supporter_tone_prompt}]
        supporter_tone_sentences = get_data(content, mode='multi', api_key=api_key)
        if supporter_tone_sentences != -1:
            supporter_statement = supporter_statement + '\nDuring the conversation, you should mimic the tone in the following example:\n[Example tone]\n' + supporter_tone_sentences
        else:
            supporter_tone_sentences = ''

    # 对seeker和supporter进行角色设定
    seeker_content.append({'role': 'system', 'content': seeker_statement})
    supporter_content.append({'role': 'system', 'content': supporter_statement})

    # 开始对话
    turn_index = 0
    i = 0
    round_num = max_turn
    score_list = []
    stop_flag = False
    initial_dialogue = 'Hello'
    current_speaker = 'seeker'
    field_list = []
    info_filter = []
    while i <= round_num:
        if i > max_turn - 2:
            stop_flag = True

        if current_speaker == 'seeker':
            if i == 0:
                seeker_content.append({'role': 'assistant', 'content': initial_dialogue})
                supporter_content.append({'role': 'user', 'content': initial_dialogue})
                current_speaker = 'supporter'
                dial_item = {
                    'turn': 0,
                    'speaker': 'seeker',
                    'choose_info': {
                        'aspect': '',
                        'content': ''
                        },
                        'utterance': initial_dialogue
                        }
                dial.append(dial_item)
                i += 1
                turn_index += 1
                continue
            
            pre_content = seeker_content[-1]['content']
            if not stop_flag:
                seeker_content[-1]['content'] = answer_prompt_prefix_seeker + pre_content + answer_prompt_postfix_seeker
                seeker_tone_format = ''
                if seeker_tone:
                    seeker_tone_format = 'Your tone is described as follows:\n' + seeker_tone + '\nPlease mimic this tone when replying.'
                
                seeker_info_format = ''
                if i == 2:
                    seeker_info = 'recent_worry_or_anxiety'
                else:
                    seeker_info = get_additional_info(pre_content, seeker_additional_info, info_filter=info_filter, api_key=api_key)
                    if seeker_info == '':
                        t = random.randint(0, 1)
                        if t == 0:
                            seeker_info = 'additional_information'
                    else:
                        info_filter = get_info_filter(field_list, seeker_info)

                if seeker_info and seeker_info in seeker_profile:
                    seeker_info_format = seeker_profile[seeker_info]
                
                try:
                    seeker_content[-1]['content'] = seeker_content[-1]['content'].format(seeker_tone_format, seeker_info_format)
                except Exception as e:
                    print(seeker_content[-1]['content'])
            else:
                seeker_content[-1]['content'] = seeker_end_prompt.format(seeker_tone, pre_content)
            
            resp = get_data(seeker_content, mode='multi', api_key=api_key)
            if resp == -1:
                dial_flag = False
                break
            resp = process_content(resp, pre_content)
            seeker_content[-1]['content'] = pre_content

            if resp == '':
                resp = 'Um, please go on, I\'m listening.'
            seeker_content.append({'role': 'assistant', 'content': resp})
            supporter_content.append({'role': 'user', 'content': resp})
            current_speaker = 'supporter'
            dial_item = {
                'turn': turn_index,
                'speaker': 'seeker',
                'choose_info': {
                    'aspect': seeker_info,
                    'content': seeker_info_format
                },
                'utterance': resp
            }
            dial.append(dial_item)
            if not stop_flag:
                score = eval_sentence(resp, api_key=api_key)
                if score != -1:
                    score_list.append(score)
                stop_flag = check_end_state(score_list)
                if stop_flag:
                    round_num = 4
                    i = 2
            i += 1
            turn_index += 1
            
        elif current_speaker == 'supporter':
            pre_content = supporter_content[-1]['content']
            if not stop_flag:
                supporter_content[-1]['content'] = answer_prompt_prefix_supporter + pre_content + answer_prompt_postfix_supporter
                supporter_tone_format = ''
                if supporter_tone:
                    supporter_tone_format = 'Your tone is described as follows:\n' + supporter_tone + '\nPlease mimic this tone when replying.'
                supporter_info_format = ''
                
                supporter_info = get_additional_info(pre_content, supporter_additional_info, info_filter=info_filter, api_key=api_key)
                if supporter_info == '':
                    t = random.randint(0, 1)
                    if t == 0:
                        supporter_info = 'additional_information'
                else:
                    info_filter = get_info_filter(field_list, supporter_info)
                if supporter_info and supporter_info in supporter_profile:
                    supporter_info_format = supporter_profile[supporter_info]
                
                try:
                    supporter_content[-1]['content'] = supporter_content[-1]['content'].format(supporter_tone_format, supporter_info_format)
                except Exception as e:
                    print(supporter_content[-1]['content'])
            else:
                supporter_content[-1]['content'] = supporter_end_prompt.format(supporter_tone, pre_content)

            resp = get_data(supporter_content, mode='multi', api_key=api_key)
            if resp == -1:
                dial_flag = False
                break
            resp = process_content(resp, pre_content)
            supporter_content[-1]['content'] = pre_content
            
            if resp == '':
                resp = 'Um, please go on, I\'m listening.'
            seeker_content.append({'role': 'user', 'content': resp})
            supporter_content.append({'role': 'assistant', 'content': resp})
            current_speaker = 'seeker'
            # print('supporter: '+resp)
            dial_item = {
                'turn': turn_index,
                'speaker': 'supporter',
                'choose_info': {
                    'aspect': supporter_info,
                    'content': supporter_info_format
                },
                'utterance': resp
            }
            dial.append(dial_item)
            if not stop_flag:
                stop_flag = check_end_state(score_list)
                if stop_flag:
                    round_num = 4
                    i = 2
            i += 1
            turn_index += 1
    
    data = {
        'dial_flag': dial_flag,
        'dial': dial,
        'seeker_tone_sentences': seeker_tone_sentences,
        'supporter_tone_sentences': supporter_tone_sentences
    }
    return data


def thread_query(index, mbti_profile_seeker, mbti_profile_supporter, api_key, key_n, thread_n, base_path):
    seeker_mbti = mbti_profile_seeker['mbti']
    seeker_statement = mbti_profile_seeker['seeker_statement']
    seeker_profile_trans = mbti_profile_seeker['profile_trans']
    seeker_tone = ''
    if 'tone' in seeker_profile_trans:
        seeker_tone = seeker_profile_trans['tone']
    supporter_mbti = mbti_profile_supporter['mbti']
    supporter_statement = mbti_profile_supporter['supporter_statement']
    supporter_profile_trans = mbti_profile_supporter['profile_trans']
    supporter_tone = ''
    if 'tone' in supporter_profile_trans:
        supporter_tone = supporter_profile_trans['tone']
    data = gen_dial(seeker_statement, supporter_statement, seeker_profile_trans, supporter_profile_trans, seeker_tone, supporter_tone, api_key=api_key)
    dial_flag = data['dial_flag']
    dial = data['dial']
    seeker_tone_sentences = data['seeker_tone_sentences']
    supporter_tone_sentences = data['supporter_tone_sentences']
    if dial_flag:
        json_item = {
            'index': index,
            'seeker_info': {
                'seeker_index': mbti_profile_seeker['index'],
                'seeker_mbti': seeker_mbti,
                'seeker_profile': mbti_profile_seeker['profile'],
                'seeker_profile_trans': seeker_profile_trans,
                'seeker_statement': seeker_statement,
                'seeker_tone_sentences': seeker_tone_sentences
            },
            'supporter_info': {
                'supporter_index': mbti_profile_supporter['index'],
                'supporter_mbti': supporter_mbti,
                'supporter_profile': mbti_profile_supporter['profile'],
                'supporter_profile_trans': supporter_profile_trans,
                'supporter_statement': supporter_statement,
                'supporter_tone_sentences': supporter_tone_sentences,
            },
            'dial': dial
        }
        with open(base_path+'key_{}_thread_{}.json'.format(str(key_n), str(thread_n)), 'a', encoding='utf-8') as f:
            f.write(json.dumps(json_item, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/dial_input.json')
    parser.add_argument('--output_dir', type=str, default='./data/dial_output/')
    parser.add_argument('--output_file', type=str, default='./data/dial_output.json')
    parser.add_argument('--api_key_file', type=str, default='./api_keys.txt')
    parser.add_argument('--thread_num', type=int, default=4)
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir
    output_file = args.output_file

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.api_key_file, 'r+', encoding='utf-8') as f:
        api_key_list = [_.strip() for _ in f.readlines()]
    with open(input_file, 'r+', encoding='utf-8') as f:
        data_list = [json.loads(_) for _ in f.readlines()]
    
    use_key_num = len(api_key_list)
    thread_num = args.thread_num

    bucket_size = use_key_num * thread_num
    bucket_num = len(data_list) // bucket_size
    thread_query_list = [data_list[i*bucket_size:(i+1)*bucket_size] for i in range(bucket_num)]
    if bucket_size * bucket_num < len(data_list):
        thread_query_list.append(data_list[-(len(data_list)-bucket_size*bucket_num):])
    
    for query_index in range(len(thread_query_list)):
        with open('./query_log.txt', 'a', encoding='utf-8') as f:
            f.write('query {} start'.format(str(query_index)))
            f.write('\n')
            f.write(str(datetime.datetime.now()))
            f.write('\n')

        index_list = []
        for key in range(use_key_num):
            for thread in range(thread_num):
                index_list.append((key, thread))
        index_list = index_list[:len(thread_query_list[query_index])]

        threads = []
        for index in range(len(index_list)):
            mbti_profile_seeker = thread_query_list[query_index][index][0]
            mbti_profile_supporter = thread_query_list[query_index][index][1]
            key_index = index_list[index][0]
            thread_n = index_list[index][1]
            api_key = api_key_list[key_index]
            threads.append(threading.Thread(target=thread_query, args=(query_index*bucket_size+index, mbti_profile_seeker, mbti_profile_supporter, api_key, key_index, thread_n, input_file)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        with open('./query_log.txt', 'a', encoding='utf-8') as f:
            f.write('query {} end'.format(str(query_index)))
            f.write('\n')
            f.write(str(datetime.datetime.now()))
            f.write('\n')

    file_list = os.listdir(output_dir)
    for file in file_list:
        with open(output_dir+file, 'r+', encoding='utf-8') as fin:
            line = fin.readline()
            with open(output_file, 'a', encoding='utf-8') as fout:
                fout.write(line)