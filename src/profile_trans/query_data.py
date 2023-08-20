import os

import openai
import json
import threading
import datetime
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

def get_statement(profile, api_key):
    prompt = """Task: Transform the following character settings given in JSON format into everyday language.
When rewriting, use the second person to express (you are...), and the expression should be smooth and natural, in line with English speaking habits.
Character settings (JSON format):
{}

Output must be succinct.
"""

    seeker_base_fields = ['name', 'gender', 'age', 'region', 'job', 'personality', 'tone', 'recent_worry_or_anxiety']
    supporter_base_fields = ['name', 'gender', 'age', 'region', 'job', 'personality', 'tone']
    seeker_profile = {k:v for k, v in profile.items() if k in seeker_base_fields}
    supporter_profile = {k:v for k, v in profile.items() if k in supporter_base_fields}
    seeker_prompt = prompt.format(str(seeker_profile))
    supporter_prompt = prompt.format(str(supporter_profile))
    content = [{'role': 'user', 'content': seeker_prompt}]
    seeker_statement = get_data(content, mode='multi', api_key=api_key)
    if seeker_statement == -1:
        seeker_statement = ''
    content = [{'role': 'user', 'content': supporter_prompt}]
    supporter_statement = get_data(content, mode='multi', api_key=api_key)
    if supporter_statement == -1:
        supporter_statement = ''
    return {'seeker_statement': seeker_statement, 'supporter_statement': supporter_statement}

def get_profile_trans(profile, api_key):
    prompt = """Task: Transform the following {} information given in JSON format into everyday language.
When rewriting, use the second person to express (you are...), and the expression should be smooth and natural, in line with English speaking habits.
Character settings (JSON format):
{}

Output must be succinct.
"""
    base_fields = ['name', 'gender', 'age', 'region', 'job']
    if 'name' in profile:
        name = profile['name']
    else:
        name = ''
    for k, v in profile.items():
        try:
            v = str(v)
        except Exception as e:
            continue
        if len(v.split(' ')) >= 20:
                continue
        if name != '':
            if name in v:
                continue
            flag = False
            name_list = name.split(' ')
            for _ in name_list:
                if _ in v:
                    flag = True
                    break
            if flag:
                continue
        if k not in base_fields:
            base_fields.append(k)
    profile_trans = {}
    for k, v in profile.items():
        if k in base_fields:
            profile_trans[k] = v
        else:
            try:
                v = str(v)
            except Exception as e:
                profile_trans[k] = v
                continue
            prompt_trans = prompt.format(k, v)
            content = [{'role': 'user', 'content': prompt_trans}]
            resp = get_data(content, mode='multi', api_key=api_key)
            if resp == -1:
                resp = ''
            profile_trans[k] = resp
    return {'profile_trans': profile_trans}

def get_info(profile, api_key):
    prompt = """You are an outstanding creator, you can construct a variety of characters in the real world.
Now, you get a profile about a character. Based on the given profile, we need you to use your imagination and write some additional information about the character.
You can use your talents as much as possible to boldly Fantasy, but make sure the additional information if in line with the profile.

profile (JSON format):
{}

additional_information must be succinct, less than 100 words, only output new information, do not repeat the existing information in the profile.

additional_information (string format, do not output lists or dictionaries, must be succinct, less than 100 words):
"""
    
    prompt = prompt.format(str(profile))
    content = [{'role': 'user', 'content': prompt}]
    additional_information = get_data(content, mode='multi', api_key=api_key)
    if additional_information == -1:
        additional_information = ''
    return additional_information

def get_info_trans(info, api_key):
    if info == '':
        return info
    prompt = """Task: Rewrite the given information.
When rewriting, use the second person to express (you are...), and the expression should be smooth and natural, in line with English speaking habits.

given formation:
{}

Output must be succinct.
"""

    prompt = prompt.format(str(info))
    content = [{'role': 'user', 'content': prompt}]
    info_trans = get_data(content, mode='multi', api_key=api_key)
    if info_trans == -1:
        info_trans = str(info)
    return info_trans

def thread_query(index, mbti, profile, api_key, key_n, thread_n, base_path):
    statement = get_statement(profile, api_key)
    profile_trans = get_profile_trans(profile, api_key)
    data = {}
    data['index'] = index
    data['mbti'] = mbti
    data['profile'] = profile
    for k, v in statement.items():
        data[k] = v
    for k, v in profile_trans.items():
        data[k] = v
    additional_information = get_info(profile, api_key)
    additional_information_trans = get_info_trans(additional_information, api_key)
    if 'profile' in data:
        data['profile']['additional_information'] = additional_information
        data['profile_trans']['additional_information'] = additional_information_trans
    with open(base_path+'key_{}_thread_{}.json'.format(str(key_n), str(thread_n)), 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))
        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/profile_output.json')
    parser.add_argument('--output_dir', type=str, default='./data/profile_trans_output/')
    parser.add_argument('--output_file', type=str, default='./data/profile_trans_output.json')
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

        threads = []
        for index in range(len(index_list)):
            mbti = thread_query_list[query_index][index]['mbti']
            profile = thread_query_list[query_index][index]['filled_profile']
            key_index = index_list[index][0]
            thread_n = index_list[index][1]
            api_key = api_key_list[key]
            threads.append(threading.Thread(target=thread_query, args=(query_index*bucket_size+index, mbti, profile, api_key, key_index, thread_n, output_dir)))

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
            with open(output_file, 'a', encoding='utf-8') as fout:
                line = fin.readline()
                while line:
                    fout.write(line)
                    line = fin.readline()