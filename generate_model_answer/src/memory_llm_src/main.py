import os
import json
import random
from utils import *
from configs import model_nicks

os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN'

def main():
    args = arg_parser()
    task_list = json.loads(args.task_list)
    input_ds = parsed_input_data(args.data_path, args.lang_type, task_list)

    dataset_dict = {}

    for task in task_list:
        dataset = input_ds[task]
        fewshot_samples = dataset[:2]
            
        if len(dataset) == 0:
            continue
        
        random.shuffle(dataset)

        dataset_dict[task] = {"fewshot": fewshot_samples, "dataset": dataset}

    inference_worker(dataset_dict, args)

if __name__ == "__main__":
    main()
