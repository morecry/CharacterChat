import re
import openai
import json
import threading
import datetime
import os
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

def get_profile(content):
    profile = re.findall(r'{.*}', content, re.DOTALL)
    if profile:
        profile = profile[0]
    else:
        profile = ''
    return profile

def thread_query(index, mbti, command_list, api_key, key_n, thread_n, base_path):
    content = []
    profile = ''
    filled_profile = ''
    k = 0
    for command in command_list:
        input_text = command
        content.append({"role": "user", "content": input_text})
        response = get_data(content, mode='multi', api_key=api_key)
        if response == -1:
            response = ''
        if k == 0:
            try:
                profile = get_profile(response)
            except Exception as e:
                profile = ''
        if k == 1:
            try:
                filled_profile = get_profile(response)
            except Exception as e:
                filled_profile = ''
            json_item = {
                'index': index,
                'mbti': mbti,
                'profile': profile,
                'filled_profile': filled_profile
                }
            with open(base_path+'key_{}_thread_{}.json'.format(str(key_n), str(thread_n)), 'a', encoding='utf-8') as f:
                f.write(json.dumps(json_item, ensure_ascii=False))
                f.write('\n')
        content.append({"role": "assistant", "content": response})
        k += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/profile_input.json')
    parser.add_argument('--output_dir', type=str, default='./data/profile_output/')
    parser.add_argument('--output_file', type=str, default='./data/profile_output.json')
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
            command_list = thread_query_list[query_index][index]['command_list']
            key_index = index_list[index][0]
            thread_n = index_list[index][1]
            api_key = api_key_list[key]
            threads.append(threading.Thread(target=thread_query, args=(query_index*bucket_size+index, mbti, command_list, api_key, key_index, thread_n, output_dir)))

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
            with open('./1.json', 'a', encoding='utf-8') as fout:
                line = fin.readline()
                while line:
                    fout.write(line)
                    line = fin.readline()
                    