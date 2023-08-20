import json
import random
import os

def gen_dial_list(num=1024, sample_num=10, output_path='./data/dial_list.json'):
    if os.path.exists(output_path):
        with open(output_path, 'r+', encoding='utf-8') as f:
            dial_list = json.load(f)
        return dial_list
    dial_list = {}
    sample_list = [i for i in range(num)]
    for i in range(num):
        cur_list = sample_list[0:i] + sample_list[i+1:num]
        cur_selected = random.sample(cur_list, sample_num)
        dial_list[str(i)] = cur_selected
    with open(output_path, 'w+', encoding='utf-8') as f:
        json.dump(f, ensure_ascii=False)
    return dial_list

if __name__ == '__main__':
    random.seed(0)

    with open('./profile_trans_output.json', 'r+', encoding='utf-8') as f:
        character_list = [json.loads(_) for _ in f.readlines()]

    num = len(character_list)
    dial_list = gen_dial_list(num)

    data_list = []
    for k, v in dial_list.items():
        k = int(k)
        seeker = character_list[k]
        for _ in v:
            _ = int(_)
            supporter = character_list[_]
            data_list.append((seeker, supporter))

    for _ in data_list:
        with open('./data/dial_input.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(_, ensure_ascii=False))
            f.write('\n')