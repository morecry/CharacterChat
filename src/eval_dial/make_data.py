import json

with open('./data/dial_output.json', 'r+', encoding='utf-8') as f:
    data_list = [json.loads(_) for _ in f.readlines()]

for dial in data_list:
    dial_str = ''
    index = dial['index']
    for _ in dial['dial']:
        dial_str = dial_str + _['speaker'] + ': ' + _['utterance'] + '\n'
    with open('./data/eval_dial_input.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'index': index, 'dial': dial_str}, ensure_ascii=False))
        f.write('\n')