# test T5
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import time
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel, T5Config, T5ForConditionalGeneration
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from model import *
from collections import defaultdict

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    
    # PTM model
    parser.add_argument('--check_point_dir', type=str)
    
    # 데이터 관련
    parser.add_argument('--max_length',type= int, default = 512)
    parser.add_argument('--batch_size', default = 8, type=int)
    
    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    args = parser.parse_args()
    return args

def pair_evaluation(args, model, tokenizer, eval_dataloader):
    total_loss = 0.
    model.eval()
    Predict = []
    Actual = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items()}
            sentence_1_score = model.forward(data['sentence_1_input_ids'], data['sentence_1_attention_mask'])['score'] # bs
            sentence_2_score = model.forward(data['sentence_2_input_ids'], data['sentence_2_attention_mask'])['score'] # bs
            predict = torch.cat([sentence_1_score.unsqueeze(1), sentence_2_score.unsqueeze(1)], dim=1).argmax(dim=1).tolist()
            Predict.extend(predict)
            Actual.extend(data['labels'].cpu().tolist())
    acc = []
    for i,j in zip(Predict, Actual):
        acc.append(i==j)
    cnt = len(Predict)
    return dict(acc=sum(acc)/cnt), Predict

def tag_and_statistics(data, predicts):
    for i,j in zip(data, predicts):
        i['predict'] = j
    statistics = defaultdict(list)
    for i in data:
        statistics[i['degree']].append(int(i['predict']==i['label']))
    keys = sorted(list(statistics.keys()))
    for k in keys:
        print(f'{k}-acc')
        print(sum(statistics[k])/len(statistics[k]))
        print(f'{k}-cnt')
        print(len(statistics[k]))
    return data
    
if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)    
    ###########################################################################################
    # tokenizer, config, model
    ###########################################################################################
    config = T5Config.from_pretrained(check_point_args['ptm_path'])
    tokenizer = AutoTokenizer.from_pretrained(check_point_args['ptm_path'])
    if 'enc' in check_point_args['ptm_path']:
        model_type = T5EncoderModel
        model = PairWiseModel(config, 'mean', T5EncoderModel)
    else:
        model_type = T5ForConditionalGeneration
        model = T5PairWiseModel(config, T5ForConditionalGeneration)
    model.load_state_dict(torch.load(os.path.join(check_point_args['output_dir'],'best_model')))
    ###########################################################################################
    # device
    ###########################################################################################
    # TODO
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # multi gpu
    else:
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device) 
        model.to(device)
    ###########################################################################################
    
    ###########################################################################################
    # data
    ###########################################################################################
    test_data = load_jsonl(args.test_data)
    
    # just list wise when evaluates
    test_dataset = PairWiseDataset(test_data, tokenizer, args.max_length)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, sampler = test_sampler, collate_fn = test_dataset.collate_fn)
    
    scores, predicts = pair_evaluation(args, model, tokenizer, test_dataloader)
    print(scores)
    test_data = tag_and_statistics(test_data, predicts)
    save_jsonl(args.output_dir, test_data, 'predicted')
    ###########################################################################################
    