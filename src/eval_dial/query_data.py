import re
import json
import openai
import datetime
import threading
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

def get_dial_score(dial, api_key=''):
    system_prompt = """You are an experienced dialogue quality assessment expert, you are good at analyzing the overall quality of dialogue, and make a notarized, rational and objective evaluation of the quality of dialogue according to the requirements.
    """
    
    prompt = """You will read an emotional dialogue.
    
    The background of the dialogue is as follows:
    Seeker recently encountered some troubles or things that made him feel anxious, so seeker chatted with supporter, hoping to get emotional support.

    Here is the dialogue, please read carefully:
    @dial@

    After reading the dialogue, please evaluate this dialogue from 3 dimensions:
    1. Does the conversation improve the mood of the seeker?
    2. Did the suggestion made by the supporter in the conversation solve Seeker's problem?
    3. Is the seeker actively participating in the conversation?

    For any dimension, use a 1-5 scale rating:
    1-poor
    2-weak
    3-moderate
    4-strong
    5-very strong

    Note that only the rating (1-5) is required, do not output the reason for rating or redundant information

    Output format (JSON):
    {
        "Does the conversation improve the mood of the seeker": , # Rating for this dimension
        "Did the suggestion made by the supporter in the conversation solve Seeker's problem": , # Rating for this dimension
        "Is the seeker actively participating in the conversation": , # Rating for this dimension
        "Rating Reason": , # explain why you rated it this way, no more than 100 words
    }
    """

    flag = True
    prompt = re.sub(r'@dial@', re.escape(str(dial)), prompt)
    content = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
        ]
    resp = get_data(content, mode='multi', api_key=api_key)
    if resp == -1:
        flag = False
        resp = ''
    
    data = {
        'flag': flag,
        'rating': resp
    }
    return data

def thread_query(index, dial, api_key, key_n, thread_n, base_path):
    data = get_dial_score(dial, api_key=api_key)
    flag = data['flag']
    rating = data['rating']
    if flag:
        json_item = {
            'index': index,
            'dial': dial,
            'rating': rating
        }
        with open(base_path+'key_{}_thread_{}.json'.format(str(key_n), str(thread_n)), 'a', encoding='utf-8') as f:
            f.write(json.dumps(json_item, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./data/dial_output.json')
    parser.add_argument('--output_dir', type=str, default='./data/eval_dial_output/')
    parser.add_argument('--output_file', type=str, default='./data/eval_dial_output.json')
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
        post_flag = False
        post_num = 0
        for index in range(len(index_list)):
            idx = thread_query_list[query_index][index]['index']
            dial = thread_query_list[query_index][index]['dial']
            key_index = index_list[index][0]
            thread_n = index_list[index][1]
            api_key = api_key_list[key_index]
            threads.append(threading.Thread(target=thread_query, args=(idx, dial, api_key, key_index, thread_n, input_file)))

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