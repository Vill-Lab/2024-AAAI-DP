import jsonlines
import threading
import random
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from annotator import GeneralGenerator

api_keys = ""


def description_annotation_data_list(data_list):
    texts = []

    for data in data_list:        
        text = f'''You are asked to rewrite a sentence describing pedestrians substituting the corresponding attribute (refer to the attribute list). 
        The sentence to be rewritten is as follows:
        {data['base_description']}

        The attribute list: 
        {data['reference_attribute_list']}

        The rewritten sentence:
        '''
        
        texts.append(text)

    return texts


def load_processed_jsonl(file_path):
    data_list = []
    with jsonlines.open(file_path) as reader:
        for data in reader:
            data_list.append(data)

    return data_list


def generate_instruction(task):
    data_list = load_processed_jsonl(task['origin_path'])
    texts = description_annotation_data_list(data_list)
    print("Generated {} instructions.".format(len(texts)))

    return texts


def generator(task):
    data_list = load_processed_jsonl(task['origin_path'])
    texts = description_annotation_data_list(data_list)
    evol = GeneralGenerator(
        model_name=task['model'],
        backend="openai",
        backend_settings={"api_key": api_keys},
    )

    results = evol.ingest(texts[task['start']:task['end']], task['output_path'], strategy=task['strategy'], model_name=task['model'])
    return results


def main(origin_task_list):
    exec_task_list = []
    for task in origin_task_list:
        texts = generate_instruction(task)[task['TEXT_START']:task['TEXT_END']]

        texts_slice_list = []
        start = 0
        end = start + task['SLICE_STEP']
        if end > len(texts):
            end = -1
            texts_slice_list.append({'start':start,'end':end})
        else:
            while(1):
                if start >= len(texts):
                    break
                if end > len(texts):
                    end = -1
                texts_slice_list.append({'start':start,'end':end})
                start += task['SLICE_STEP']
                end += task['SLICE_STEP']

        for slice in texts_slice_list:
            exec_task_list.append(
                {
                    'task_name':task['task_name']+' ('+str(slice['start'])+':'+str(slice['end'])+')',
                    'task_type':task['task_type'],
                    'origin_path':task['origin_path'],
                    'output_path':task['output_path'],
                    'start':slice['start'],
                    'end':slice['end'],
                    'model':task['model'],
                    'strategy':task['strategy'],
                },
            )

    exec_task_count = 1
    for task in exec_task_list:
        print('\n------- task',exec_task_count,'-------')
        for info in task:
            print(info+':',str(task[info]))
        exec_task_count += 1

        t = threading.Thread(target=generator, args=(task, ))
        t.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--origin_path', type=str, help='Path to the original data')
    parser.add_argument('--output_path', type=str, help='Path to save the generated data')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--TEXT_START', type=int, default=0)
    parser.add_argument('--TEXT_END', type=int, default=-1)
    parser.add_argument('--SLICE_STEP', type=int, default=500)
    
    args = parser.parse_args()
    
    origin_task_list = [
    {
        'task_name': 'description_annotation',
        'task_type': 'description_annotation',
        'origin_path': args.origin_path,
        'output_path': args.output_path,
        'model': args.model,
        'strategy':'general_generation',
        'TEXT_START': args.TEXT_START,
        'TEXT_END': args.TEXT_END,
        'SLICE_STEP': args.SLICE_STEP,
    }
]
    
    main(origin_task_list)
