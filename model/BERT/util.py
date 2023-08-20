from datasets import Dataset, load_dataset
import numpy as np
import json

def get_dataset(
    dataset_path
) -> Dataset:
    dataset = load_dataset("json", data_files=dataset_path)['train']
    return dataset

def process_dataset(dataset, tokenizer):
    
    column_names = list(dataset.column_names)

    def process_fun(examples):
        model_inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "labels": []}
        max_length = 512
        for example in examples['conversations']:
            input_ids = []
            token_type_ids = []
            attention_mask = []
            labels = []
            
            history_ids = tokenizer.encode(example['history'], add_special_tokens=False)
            info_ids = tokenizer.encode(example['info'], add_special_tokens=False)
            if len(history_ids) + len(info_ids) > max_length - 3:
                history_ids = history_ids[:max_length-3-len(info_ids)]
            value_ids = tokenizer.build_inputs_with_special_tokens(history_ids, info_ids)
            input_ids += value_ids

            token_type_ids += tokenizer.create_token_type_ids_from_sequences(history_ids, info_ids)
            attention_mask += [1] * len(input_ids)
            if len(input_ids) != len(token_type_ids):
                token_type_ids = [0] * len(input_ids)

            label_ids = [0]
            if example['label'] == 1:
                label_ids[0] = 1.0
            else:
                label_ids[0] = 0.0
            labels += label_ids

            model_inputs['input_ids'].append(input_ids)
            model_inputs['labels'].append(labels)
            model_inputs['attention_mask'].append(attention_mask)
            model_inputs['token_type_ids'].append(token_type_ids)

        return model_inputs
    
    dataset = dataset.map(
        process_fun,
        batched=True,
        remove_columns=column_names,
        num_proc=32
    )

    return dataset