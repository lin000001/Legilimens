# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import numpy as np
from prompt_format_utils import LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_2, SafetyCategory
from prompt_format_utils import build_custom_prompt, build_default_prompt, create_conversation, LlamaGuardVersion, AgentType
from typing import List, Tuple
from enum import Enum
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
dtype = torch.bfloat16
import sys
sys.path.append('../../')
from tools.dataset import CommonDataset, OigfeatureDataset, PromptDataset, JailbreakDataset, CustomDataset
from torch.utils.data import Dataset  
import pandas as pd
from tqdm import tqdm
import argparse
import time
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()
    if args.test_dataset == 'beaver':
        # args.dataset_path = './legilimens/test_data_subset/beaver_split_test_2000.csv'
        args.dataset_path = 'XXX/test_data_subset/beaver_5000.csv'
    elif args.test_dataset == 'beaver_adv':
        args.dataset_path = 'XXX/test_data_subset/BEA_adv.csv'
    elif args.test_dataset == 'oig':
        args.dataset_path = 'XXX/test_data_subset/final_test_oig_simple_rebuttal.csv'
    elif args.test_dataset == 'jigsaw':
        args.dataset_path = 'XXX/test_data_subset/jigsaw_test_2000_rebuttal.csv'
    elif args.test_dataset == 'hatexplain':
        args.dataset_path = 'XXX/test_data_subset/hx_2000_rebuttal.csv'
    elif args.test_dataset == 'mhs':
        args.dataset_path = 'XXX/test_data_subset/MHS_test_0dot5_2000_rebuttal.csv'
    elif args.test_dataset == 'bag':
        args.dataset_path = 'XXX/test_data_subset/BAG_2000_rebuttal.csv'
    elif args.test_dataset == 'bag_adv':
        args.dataset_path = 'XXX/test_data_subset/BAG_adv_test_2000_rebuttal.csv'
    else:
        raise NotImplementedError
    return args

class Prompt_ResponseDataset(Dataset):
    def __init__(self, csv_file, labels_column):
        self.df = pd.read_csv(csv_file)
        self.prompts = self.df['prompt'].values
        self.response = self.df['response'].values
        self.labels = self.df[labels_column].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {'prompts': self.prompts[idx], 'response': self.response[idx], 'labels': self.labels[idx] }
        return sample

def to_binary(dataset_name, labels):
    if dataset_name == 'oig':
        bool_labels = labels == 'needs intervention'
    elif dataset_name == 'jigsaw':
        bool_labels = labels != 'innocent'

    binary_labels = np.array(bool_labels, dtype=np.int64)
    return binary_labels

def load_dataset(dataset_name, dataset_path):
    dataset_name = dataset_name.lower()
    if dataset_name == "beaver":  
        trainset = Prompt_ResponseDataset(dataset_path, 'label')
        return trainset.prompts, trainset.response, trainset.labels  
    elif dataset_name == "oig":  
        trainset = OigfeatureDataset(dataset_path, 'input', 'label')  
        binary_label = to_binary(dataset_name, trainset.labels)
        return trainset.features, None, binary_label
    elif dataset_name == "jigsaw":  
        trainset = CustomDataset(dataset_path, 'input', 'label')
        binary_label = to_binary(dataset_name, trainset.labels)
        return trainset.features, None, binary_label
    elif dataset_name == "beaver_adv":
        trainset = JailbreakDataset(dataset_path, 'jailbreak_prompt', 'response','label')
        return trainset.features, trainset.response, trainset.labels 
    elif dataset_name == "bag":
        trainset = Prompt_ResponseDataset(dataset_path, 'label')
        return trainset.prompts, trainset.response, trainset.labels
    elif dataset_name == "bag_adv":
        trainset = JailbreakDataset(dataset_path, 'prompt', 'response','label')
        return trainset.features, trainset.response, trainset.labels
    elif dataset_name == "hatexplain":
        trainset = PromptDataset(dataset_path, 'input', 'label')
        return trainset.features, None, trainset.labels
    elif dataset_name == "mhs":
        trainset = PromptDataset(dataset_path, 'input', 'label')
        return trainset.features, None, trainset.labels
    else:  
        raise NotImplementedError 



def main(
    model_id: str = "./llama-guard2-hf",
    llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2
):
    """
    Entry point for Llama Guard inference sample script.

    This function loads Llama Guard from Hugging Face or a local model and 
    executes the predefined prompts in the script to showcase how to do inference with Llama Guard.

    Args:
        model_id (str): The ID of the pretrained model to use for generation. This can be either the path to a local folder containing the model files,
            or the repository ID of a model hosted on the Hugging Face Hub. Defaults to 'meta-llama/LlamaGuard-7b'.
        llama_guard_version (LlamaGuardVersion): The version of the Llama Guard model to use for formatting prompts. Defaults to LLAMA_GUARD_1.
    """
    args = parse_arguments()
    prompts_ori, responses, labels = load_dataset(args.test_dataset, args.dataset_path)
    # import pdb;pdb.set_trace()
    if args.test_dataset == 'oig':
        prompts_list = [([prompt], AgentType.USER) for prompt in prompts_ori]
        prompts: List[Tuple[List[str], AgentType]] = prompts_list
    elif args.test_dataset == 'jigsaw':
        prompts_list = [([prompt], AgentType.USER) for prompt in prompts_ori]
        prompts: List[Tuple[List[str], AgentType]] = prompts_list
    elif args.test_dataset == 'hatexplain':
        prompts_list = [([prompt], AgentType.USER) for prompt in prompts_ori]
        prompts: List[Tuple[List[str], AgentType]] = prompts_list
    elif args.test_dataset == 'mhs':
        prompts_list = [([prompt], AgentType.USER) for prompt in prompts_ori]
        prompts: List[Tuple[List[str], AgentType]] = prompts_list
    else:
        prompts_list = [([prompt, response], AgentType.AGENT) for prompt, response in zip(prompts_ori, responses)]
        prompts: List[Tuple[List[str], AgentType]] = prompts_list

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
    results_list = []
    y_pre_list = []
    new_time = []
    y_pre = 0
    for prompt in tqdm(prompts):
        formatted_prompt = build_default_prompt(
                prompt[1], 
                create_conversation(prompt[0]),
                llama_guard_version)
        strat_time = time.time()
        input = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0)
        results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        end_time = time.time()
        one_time = end_time - strat_time
        results_list.append(results)
        new_time.append(one_time) 
        
        if 'unsafe' in results:
            y_pre = 1
        elif results == 'safe':
            y_pre = 0
        y_pre_list.append(y_pre)
        with open(f'./predictions_llama_guard_time_{args.test_dataset}_all.json', 'w', encoding='utf-8') as f:
            json.dump({"gt": labels.tolist()}, f, ensure_ascii=False, indent=4)
            json.dump({"predict": results_list}, f, ensure_ascii=False, indent=4)
            json.dump({"y_pre": y_pre_list}, f, ensure_ascii=False, indent=4)
            json.dump({"time": new_time}, f, ensure_ascii=False, indent=4)
            
    average_time = sum(new_time) / len(new_time)
    print('average_time:', average_time)
        # print(prompt[0])
        # print(f"> {results}")
    # i = 0
    
    # if args.test_dataset == 'beaver':
    #     for pre, gt in zip(results_list, labels):
    #         if ('unsafe' in pre and bool(gt == False)) or (pre == 'safe' and bool(gt)):
    #             i += 1
    #     accuracy = i/len(results_list)
    # if args.test_dataset == 'beaver_adv':
    #     for pre, gt in zip(results_list, labels):
    #         if ('unsafe' in pre and bool(gt == False)) or (pre == 'safe' and bool(gt)):
    #             i += 1
    #     accuracy = i/len(results_list)
    # else:
    #     for pre, gt in zip(results_list, labels):
    #         if ('unsafe' in pre and gt == 1) or (pre == 'safe' and gt == 0):
    #             i += 1
    #     accuracy = i/len(results_list)
    # print(f'accuracy_{args.test_dataset}:', accuracy)

if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(e)